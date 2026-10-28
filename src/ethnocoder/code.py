import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import pymupdf
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
import litellm
from rich import print

from ethnocoder.utils import strip_fences, parse_codings, get_timestamp, ollama_get_loaded_models

# Globals
PROMPT_FILE = Path('.') / "PROMPT.md"


def load_prompt(path: Path) -> str:
    """
    Load the prompt text from the specified file path.

    Args:
        path (Path): The path to the prompt file.

    Returns:
        str: The contents of the prompt file.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_variables(path: Path) -> list[dict]:
    """
    Load variables data from a CSV file.

    Args:
        path (Path): The path to the CSV file containing variables data.

    Returns:
        list[dict]: A list of dictionaries representing the variables.
    """

    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_codes(path: Path) -> dict[str, list[dict]]:
    """
    Load codes data from a CSV file and organize them by variable ID.

    Args:
        path (Path): The path to the CSV file containing codes data.

    Returns:
        dict[str, list[dict]]: A dictionary mapping variable IDs to lists of code dictionaries.
    """

    # Initialise the dictionary to store codes by variable ID
    codes_by_var: dict[str, list[dict]] = {}

    # Open the CSV file and read its contents
    with open(path, encoding="utf-8") as f:

        # Iterate over each row in the CSV and organize codes by variable ID
        for row in csv.DictReader(f):

            # If row["Parameter_ID"] is not already a key in the dictionary, add it with an empty list
            # and then append the current row.
            codes_by_var.setdefault(row["Parameter_ID"], []).append(row)
    
    
    return codes_by_var


def extract_pdf_text(pdf_path: Path, max_chars: int | None = None) -> str:
    """
    Get the text content from a PDF file.

    Args:
        pdf_path (Path): The path to the PDF file.
        max_chars (int | None): Maximum number of characters to return. If None, return all text.

    Returns:
        str: The extracted text from the PDF.
    """

    # Open the PDF file using pymupdf
    doc = pymupdf.open(str(pdf_path))

    # Get page texts as a list (one element per page)
    pages = [f"[page {i}]\n{cast(str, page.get_text())}" for i, page in enumerate(doc, 1)]  # type: ignore[arg-type]

    # Close the PDF document
    doc.close()

    # Combine the page texts into a single string
    text = "\n\n".join(pages)

    # If max_chars is specified and the text exceeds that length, truncate it and print a warning.
    if max_chars and len(text) > max_chars:
        original_len = len(text)
        text = text[:max_chars]
        print(f"Warning: PDF text truncated to {max_chars:,} chars (was {original_len:,})", file=sys.stderr)

    # Return the final text (possibly truncated)
    return text


def build_coding_prompt(variables: list[dict], codes_by_var: dict[str, list]) -> str:
    """
    Build a prompt for coding variables based on their definitions and available codes.

    Args:
        variables (list[dict]): A list of variable dictionaries.
        codes_by_var (dict[str, list]): A dictionary mapping variable IDs to lists of code dictionaries.

    Returns:
        str: The constructed coding prompt.
    """

    # Initialize the list of lines that will make up the prompt
    lines: list[str] = [
        "Code each variable below based on the source document above.",
        "For each, assign the most appropriate code and briefly justify your choice.\n",
    ]

    # Start by simply listing the variables.
    lines.append("Code ONLY the following variables: " + ", ".join(var["ID"] for var in variables) +".\n")

    # Initialise
    current_section = None

    # Iterate over each variable to build the prompt
    for var in variables:

        # Check if the section has changed and add a section header if needed
        # Using the heading "Variable descriptions" as the default.
        section = var.get("Section", "Variable descriptions")
        if section != current_section:
            lines.append(f"[{section}]")
            current_section = section

        # Get basic details for variable
        var_id = var["ID"]
        datatype = var.get("Datatype") or "Option"
        name = var["Name"]
        description = (var.get("Description") or "").strip()

        # Add the variable ID and name to the prompt
        lines.append(f"ID {var_id}: {name}")

        # If there's a description, add it with indentation
        if description:
            lines.append(f"  {description}")

        # Different datatypes require slightly different prompt text.
        # The "Option" type requires listing valid codes and instructing to assign one code from the list.
        if datatype == "Option":

            # Get the possible codes for this variable
            var_codes = sorted(codes_by_var.get(var_id, []), key=lambda c: c["Name"])
            if var_codes:

                # Format the codes as "Name=Description" strings, truncating descriptions to 100 characters.
                code_strs = [
                    f"{c['Name']}={c['Description'][:100].rstrip()}"
                    for c in var_codes
                ]

                # Append the formatted valid codes to the prompt.
                lines.append("  Valid codes: " + " | ".join(code_strs))

            # Instruct to assign one code from the list above.
            lines.append("  Assign one code from the list above.")
        
        elif datatype == "Int":
            # For integer variables, instruct to assign an integer value.
            lines.append("  Assign an integer value.")

        elif datatype == "Float":
            # For float variables, instruct to assign a numeric value.
            lines.append("  Assign a numeric value.")

        else:
            # For other datatypes, instruct to provide a brief text value.
            lines.append("  Provide a brief text value.")

        lines.append("")

    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """
    A rough estimate of the number of tokens in the given text.

    Args:
        text (str): The input text to estimate token count for.

    Returns:
        int: Estimated number of tokens in the text.
    """

    return len(text) // 4


def pack_batches(
    items: list,
    sizes: list[int],
    prefix_tokens: int,
    context_budget: int | None,
    response_reserve_per_var: int,
) -> list[list]:
    """
    Pack items into batches based on their token sizes and context budget.

    Args:
        items (list): List of items to be batched.
        sizes (list[int]): List of token sizes corresponding to each item.
        prefix_tokens (int): Number of tokens reserved for the prefix.
        context_budget (int | None): Maximum allowed tokens for a batch, including prefix.
        response_reserve_per_var (int): Tokens reserved per variable for the response.

    Returns:
        list[list]: List of batches, each containing a subset of items.
    """

    # If no context budget is specified, return all items as a single batch.
    if context_budget is None:

        # The return type is a list of lists.
        return [list(items)] if items else []

    # Initialize the list of batches and the current batch.
    batches: list[list] = []
    current: list = []

    # Initialise current tokens: the prefix size.
    current_tokens = prefix_tokens

    # For each item and its corresponding size, determine if it fits in the current batch 
    # or if a new batch is needed.
    for item, size in zip(items, sizes):

        # Calculate the total cost of adding this item, including the response reserve.
        cost = size + response_reserve_per_var

        # If the total cost exceeds the budget per batch...
        if prefix_tokens + cost > context_budget:

            # ...raise an error. This item cannot fit within the context budget.
            raise ValueError(
                f"A single variable needs {cost} tokens on top of a {prefix_tokens}-token "
                f"prefix, exceeding the {context_budget}-token budget. "
                "Increase --context-budget or shrink the PDF with --max-chars."
            )

        # If adding this item would exceed the context budget for the current batch, start a new batch.
        if current and current_tokens + cost > context_budget:

            # Append the current batch to the list of batches and start a new batch.
            batches.append(current)
            current = []

            # Reset the current tokens to account for the prefix in the new batch.
            current_tokens = prefix_tokens

        # Add the current item to the batch and update the token count.
        current.append(item)
        current_tokens += cost

    # Append the last batch if it contains any items.
    if current:
        batches.append(current)
        
    # Return the list of batches.
    return batches


def clean_codings(codings: list[dict]) -> list[dict]:
    """
    Clean the codings by removing any internal keys that start with an underscore.

    Args:
        codings (list[dict]): List of coding dictionaries.

    Returns:
        list[dict]: List of cleaned coding dictionaries.
    """

    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in codings]


def validate_option_codes(
    codings: list[dict],
    codes_by_var: dict[str, list[dict]],
    variables: list[dict],
) -> list[dict]:
    """
    Validate that the codes assigned to option variables are among the allowed codes.
    Marks codings with "_invalid" and "_valid_codes" if the code is not valid.

    Args:
        codings (list[dict]): List of coding dictionaries.
        codes_by_var (dict[str, list[dict]]): Dictionary mapping variable IDs to their allowed codes.
        variables (list[dict]): List of variable dictionaries.

    Returns:
        list[dict]: List of validated coding dictionaries with "_invalid" and "_valid_codes" keys added where applicable.
    """

    # Set option variables
    option_vars = {v["ID"] for v in variables if v.get("Datatype", "Option") == "Option"}

    # Build a dictionary of valid codes for each option variable
    valid: dict[str, set[str]] = {
        vid: {c["Name"] for c in codes}
        for vid, codes in codes_by_var.items()
        if vid in option_vars
    }

    # Initialize the result list to store validated codings
    result = []
    for c in codings:

        # Get the variable ID and the assigned code.
        vid = str(c.get("id", ""))
        code = c.get("code")
        if code is None or code == "":
            result.append(c)  # no code assigned (evidence absent) — nothing to validate
            continue

        # If the code is not in the list of valid codes, mark it as invalid and store the valid codes.
        if vid in valid and str(code) not in valid[vid]:
            c = dict(c, _invalid=True, _valid_codes=sorted(valid[vid]))

        # If the coded variable is not even one of the requested variables, mark an error.
        # Then the review prompt will tell the LLM it shouldn't have included this variable.
        if vid not in valid:

            # Leave _valid_codes empty to indicate no valid codes are available for this variable
            # i.e. this variable should not have been included.
            c = dict(c, _invalid=True, _valid_codes=[])

        # Append to the output.
        result.append(c)

    return result


def build_review_message(codings: list[dict]) -> tuple[str | None, int, int]:
    """
    Prepare a review message for the codings, highlighting invalid codes and low confidence assignments.

    Args:
        codings (list[dict]): List of coding dictionaries.

    Returns:
        tuple[str | None, int, int]: A tuple containing the review message (or None if no issues), the number of invalid codings, and the number of low confidence codings.
    """

    # Find the invalid and low confidence codings.
    invalid = [c for c in codings if c.get("_invalid")]
    low_conf = [
        c for c in codings
        if (c.get("confidence") in ("low", "absent") or c.get("evidence") == "absent")
        and not c.get("_invalid")
    ]

    # If everything looks good, return now.
    if not invalid and not low_conf:
        return None, 0, 0

    # Prepare the review message parts.
    parts: list[str] = []
    if invalid:

        # List all invalid codings with explanations.
        lines = ["The following variables should be omitted or have invalid codes that must be corrected:"]
        
        for c in invalid:

            # Check if the variable should not have been included at all.
            if len(c['_valid_codes']) == 0:
                lines.append(
                    f"  ID {c['id']}: you coded '{c['code']}' but this variable should not have been included. DO NOT CODE THIS VARIABLE."
                )
            else:

                # The variable is correct, but the code provided is invalid.
                # Remind the LLM of the valid codes.
                lines.append(
                    f"  ID {c['id']}: you coded '{c['code']}' but valid codes are: {', '.join(c['_valid_codes'])}"
                )

        # Append the invalid coding lines to the review message parts.
        parts.append("\n".join(lines))

    # Low confidence codings.
    if low_conf:
        ids = ", ".join(str(c["id"]) for c in low_conf)

        # Ask the LLM to re-evaluate the low confidence codings.
        parts.append(
            f"For variables {ids} you assigned low or absent confidence. "
            "Re-read the document carefully for any overlooked evidence and correct these if possible."
        )

    # Combine all parts into the final review message.
    review_msg = "\n\n".join(parts) + (
        "\n\nReturn the complete corrected codings for ALL variables in the same JSON format."
    )
    return review_msg, len(invalid), len(low_conf)


def llm_stream(
    messages: list[dict],
    model: str,
    is_anthropic: bool,
    api_base: str | None,
    bool_log: bool = False,
) -> tuple[str, dict]:
    """
    Stream the LLM response for the given messages and return the final text.

    Args:
        messages: The list of messages to send to the LLM.
        model: The model to use for the LLM.
        is_anthropic: Whether the model is an Anthropic model.
        api_base: The base URL for the API, if any.
        bool_log: Whether to log the messages to a file.
    
    Returns:
        final_text: The final text response from the LLM.
        meta: Metadata about the LLM call, including elapsed time and model load status.
    """

    # Initialise. We will use litellm with the streaming option enabled.
    kwargs: dict = {"stream": True}

    # Add response_format option if using a non-Anthropic model that is not lm_studio.
    if not is_anthropic and not model.startswith("lm_studio/"):
        kwargs["response_format"] = {"type": "json_object"}

    # Set api base url if provided.
    if api_base:
        kwargs["api_base"] = api_base

    # If it's ollama, get the list of loaded models.
    if model.startswith("ollama/"):
        loaded_models = ollama_get_loaded_models()

        if model[len("ollama/"):] not in loaded_models:
            model_loaded = False
        else:
            model_loaded = True
    else:
        model_loaded = None

    # Initialise the list to collect streamed chunks.
    chunks: list[str] = []

    # Log messages now
    if bool_log:

        os.makedirs("logs", exist_ok=True)

        # Output the exact messages to a timestamped log
        ts = get_timestamp()
        with open(f"logs/{ts}_messages.log", "w", encoding="utf-8") as f:
            for message in messages:
                f.write(json.dumps(message, ensure_ascii=False, indent=2))
                f.write("\n")

    # Estimate token usage now.
    prompt_tokens = litellm.token_counter(
        model=model,       # e.g. "ollama/qwen3.5:35b"
        messages=messages,
    )

    if prompt_tokens:
        print(f"[blue]Estimated prompt tokens: {prompt_tokens}[/blue]")

    # Start timer for call
    started = time.perf_counter()

    # Send the messages to the LLM and stream the response.
    response = cast(
        Any, 
        litellm.completion(
            model=model,
            messages=messages,
            stream_options={"include_usage": True},
            extra_body={
                "keep_alive": "24h",
                "options": {"num_ctx": 16384}, # TODO: set this properly; chunk things; use RAG
            },
            **kwargs
        )
    )

    usage = None

    # Stream the response and collect chunks.
    for chunk in response:
        if getattr(chunk, "usage", None):
            usage = chunk.usage

        delta = chunk.choices[0].delta.content or ""
        if delta:
            chunks.append(delta)

    # Combine the collected chunks into the final response text.
    # Strip code fences just in case.
    final_text = strip_fences("".join(chunks))

    # Print the elapsed time for the LLM call
    elapsed = time.perf_counter() - started

    # # Create full response object for metadata reporting.
    # response_from_chunks = litellm.stream_chunk_builder(chunks, messages=messages)

    # Create metadata dictionary for the LLM call.
    meta = {
        "elapsed": f"{elapsed:.2f}",
        "model_loaded": model_loaded,
        # "prompt_tokens": response_from_chunks.usage.prompt_tokens,
        # "completion_tokens": response_from_chunks.usage.completion_tokens
    }

    if usage:
        meta["prompt_tokens"] = usage.prompt_tokens
        meta["completion_tokens"] = usage.completion_tokens
        meta["total_tokens"] = usage.total_tokens

        print(f"[green]Actual prompt tokens: {usage.prompt_tokens}, Completion tokens: {usage.completion_tokens}, Total tokens: {usage.total_tokens}[/green]")

    return final_text, meta


def code_batch(
    variables: list[dict],
    codes_by_var: dict[str, list[dict]],
    model: str,
    api_base: str | None,
    is_anthropic: bool,
    system_msg: dict,
    pdf_full: str,
) -> tuple[list[dict], str]:
    """
    Code one batch of variables in a fresh conversation.

    The prefix [system_msg, pdf_full] is identical across batches so a local
    server can reuse its KV cache for the PDF instead of reprocessing it.
    
    Args:
        variables: A list of variable dictionaries to be coded.
        codes_by_var: A dictionary mapping variable IDs to their corresponding list of code dictionaries.
        model: The model name to use for the LLM.
        api_base: The base URL for the API, if any.
        is_anthropic: A boolean indicating if the model is Anthropic.
        system_msg: The system message dictionary for the conversation.
        pdf_full: The entire PDF text, prefaced by the PDF document name.

    Returns:
        A tuple containing the list of cleaned coding dictionaries and the final raw response text.
    """

    # Create the coding prompt for the current batch of variables.
    # This is the user prompt.
    coding_prompt = build_coding_prompt(variables, codes_by_var)
    user_content = f"{pdf_full}\n\n---\n\n{coding_prompt}" if pdf_full else coding_prompt

    # Create the initial message list by combining the system message and the user prompt.
    messages = [system_msg, {"role": "user", "content": user_content}]

    # Stream the LLM response for the current batch of variables.
    text, meta = llm_stream(messages, model, is_anthropic, api_base)

    # Parse the response text into coding dictionaries.
    codings = parse_codings(text) or []

    # Validate the option codes against the provided codes by variable and the current batch of variables.
    codings = validate_option_codes(codings, codes_by_var, variables)

    # Check if we need a review pass based on the validity and confidence of the codings.
    review_msg, n_invalid, n_low = build_review_message(codings)

    # If no review is needed, return the cleaned codings and the original response text.
    if review_msg is None:
        return clean_codings(codings), text, meta

    # We will do a review pass.
    print(
        f"[magenta]  → Review pass ({n_invalid} invalid code(s), {n_low} low-confidence)…[/magenta]",
        file=sys.stderr,
    )

    # Append the previous response along with the review message.
    messages.append({"role": "assistant", "content": json.dumps({"codings": clean_codings(codings)}, indent=2)})
    messages.append({"role": "user", "content": review_msg})

    # Ask the LLM to do better.
    text2, meta2 = llm_stream(messages, model, is_anthropic, api_base)

    # combine meta and meta2 into a single dictionary
    combined_meta = {
        "elapsed_1": meta.get("elapsed", 0),
        "model_loaded_1": meta.get("model_loaded", False),
        "elapsed_2": meta2.get("elapsed", 0),
        "model_loaded_2": meta2.get("model_loaded", False)
    }

    # We won't try for another review, just return the cleaned codings from this pass.
    return clean_codings(parse_codings(text2) or []), text2, combined_meta


def code_document(
    pdf_stem: str,
    variables: list[dict],
    codes_by_var: dict[str, list[dict]],
    model: str,
    api_base: str | None,
    is_anthropic: bool,
    messages: list[dict],
    pdf_full: str = "",
    out_dir: Path = Path("."),
    context_budget: int | None = None,
    response_reserve_per_var: int = 250,
) -> list[dict]:
    """
    Code a PDF document by processing it in batches according to the context budget and response reserve.
    Returns a list of coding dictionaries for all variables.

    Args:
        pdf_stem: The stem of the PDF file name.
        variables: A list of variable dictionaries to code.
        codes_by_var: A dictionary mapping variable IDs to their possible codes.
        model: The language model to use for coding.
        api_base: The base URL for the API, if applicable.
        is_anthropic: Whether the model is an Anthropic model.
        messages: The initial system and user messages for the LLM.
        pdf_full: The actual PDF content, with the PDF filename.
        out_dir: The directory to save output files.
        context_budget: The maximum number of tokens to use per batch.
        response_reserve_per_var: The number of tokens to reserve for each variable's response.
    Returns:
        A list of coding dictionaries for all variables.
    """

    # Get the system message: it's the first in the list.
    system_msg = messages[0]

    # Estimate the number of tokens used by the system message and PDF file.
    prefix_tokens = estimate_tokens(system_msg["content"]) + estimate_tokens(pdf_full)

    # Get a dict storing estimated number of tokens per variable definition
    var_size = {
        v["ID"]: estimate_tokens(build_coding_prompt([v], codes_by_var))
        for v in variables
    }

    # Get a list of estimated token sizes for each variable
    sizes = [var_size[v["ID"]] for v in variables]

    # Pack variables into batches based on their estimated token sizes and the context budget
    try:
        batches = pack_batches(
            variables, sizes, prefix_tokens, context_budget, response_reserve_per_var
        )
    except ValueError as e:
        sys.exit(f"Error: {e}")

    # Prepare output paths
    txt_path = out_dir / f"{pdf_stem}.txt"
    json_path = out_dir / f"{pdf_stem}.json"
    meta_path = out_dir / f"{pdf_stem}.log"

    # Initialise output for codings and raw text parts
    all_codings: list[dict] = []
    raw_parts: list[str] = []

    # Are there multiple batches?
    multi = len(batches) > 1

    # Loop over each batch and code it
    for i, batch in enumerate(batches, 1):

        # If there are multiple batches, print batch information
        if multi:
            batch_tokens = prefix_tokens + sum(var_size[v["ID"]] for v in batch)
            print(
                f"  [blue]Batch {i}/{len(batches)} ({len(batch)} vars, ~{batch_tokens // 1000}k tok)…[/blue]",
                file=sys.stderr,
            )

        # Code this batch
        clean, text, meta = code_batch(
            batch, codes_by_var, model, api_base, is_anthropic, system_msg, pdf_full
        )

        # Add results from this batch to the overall output
        all_codings.extend(clean)

        # Append the raw text for this batch to the list of raw parts, with a separator stating which batch it is
        raw_parts.append(f"=== batch {i} ===\n{text}" if multi else text)

        # Output raw txt file and json file
        txt_path.write_text("\n\n".join(raw_parts), encoding="utf-8")
        json_path.write_text(json.dumps({"codings": all_codings}, indent=2), encoding="utf-8")

        # Output meta information for this batch
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


        # Inform the user
        print(f"[green]JSON saved → {json_path}[/green]", file=sys.stderr)

    return all_codings


def model_dirname(model: str) -> str:
    """
    Return the directory name for storing model outputs.

    The directory is constructed as "output/{model_name}/{os}", where colons in the model name are replaced with underscores.

    Args:
        model: The name of the model.

    Returns:
        The directory name for storing model outputs.
    """

    # Get the operating system: windows, macos, linux, or unknown_os
    if sys.platform.startswith("win"):
        os_name = "windows"
    elif sys.platform.startswith("darwin"):
        os_name = "macos"
    elif sys.platform.startswith("linux"):
        os_name = "linux"
    else:
        os_name = "unknown_os"

    return os.path.join("output", model.split("/")[-1].replace(":", "_"), os_name)


def resolve_model_config(model: str, api_base: str | None) -> tuple[bool, str | None]:
    """
    Determine the model configuration based on the model name and API base.

    Args:
        model: The name of the model.
        api_base: The API base URL, if any.

    Returns:
        A tuple containing:
        - is_anthropic: True if the model is an Anthropic Claude model, False otherwise.
        - api_base: The resolved API base URL, if applicable.
    """

    # It's anthropic if the model name contains "claude" and doesn't start with "openai/"
    is_anthropic = "claude" in model.lower() and not model.startswith("openai/")

    # If the API base is not provided and the model is an lm_studio model, use the default local API base
    if api_base is None and model.startswith("lm_studio/"):
        api_base = "http://localhost:1234/v1"

    return is_anthropic, api_base


def code_pdf(
    pdf_path: Path,
    variables: list[dict],
    codes_by_var: dict[str, list[dict]],
    model: str | None,
    api_base: str | None = None,
    max_chars: int | None = None,
    context_budget: int | None = None,
    response_reserve_per_var: int = 250,
) -> list[dict]:
    """
    Code a PDF for trait/feature variables using a local or remote LLM.

    Args:
        pdf_path: Path to the PDF file.
        variables: List of variable dictionaries.
        codes_by_var: Dictionary mapping variable IDs to lists of code dictionaries.
        model: Model string (required).
        api_base: API base URL, if any.
        max_chars: Maximum number of characters to extract from the PDF.
        context_budget: Token budget per request.
        response_reserve_per_var: Reserved tokens per variable in the response.

    Returns:
        List of coding results as dictionaries.
    """

    # Error if no model supplied
    if model is None:
        print("[red]Error: --model is required[/red]", file=sys.stderr)
        sys.exit(1)

    # Inform the user
    print(f"[blue]Extracting text from {pdf_path.name} ({pdf_path.stat().st_size / 1e6:.1f} MB)…[/blue]", file=sys.stderr)

    # Get PDF text
    pdf_text = extract_pdf_text(pdf_path, max_chars=max_chars)

    # Get model config
    is_anthropic, api_base = resolve_model_config(model, api_base)

    # Set up inputs for LLM
    pdf_full = f"Source document: {pdf_path.stem}\n\n{pdf_text}"
    messages = [{"role": "system", "content": load_prompt(PROMPT_FILE)}]

    # Prepare output directory for coded results
    out_dir = Path(model_dirname(model))
    print(f"[blue]Preparing output directory {out_dir}…[/blue]", file=sys.stderr)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Inform the user and code the document
    print(f"[blue]Coding [magenta]{pdf_path.name}[/magenta] ({len(variables)} variables) with [green bold]{model}[/green bold]…[/blue]", file=sys.stderr)
    return code_document(
        pdf_path.stem, variables, codes_by_var, model, api_base, is_anthropic,
        messages, pdf_full=pdf_full, out_dir=out_dir,
        context_budget=context_budget, response_reserve_per_var=response_reserve_per_var,
    )


def add_subparser(subparsers) -> None:
    """
    Add the "code" subparser for coding a PDF with a local or remote LLM.

    Args:
        subparsers: The argparse subparsers object to which the "code" subparser will be added.
    """

    # Add the "code" subparser to the provided subparsers object
    p = subparsers.add_parser(
        "code",
        description="Code a PDF for trait/feature variables using a local or remote LLM",
        help="Code a PDF for trait/feature variables using a local or remote LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add arguments for the "code" subparser
    # Positional argument for the PDF file to code
    p.add_argument("pdf", help="PDF file to code")

    # model: the LLM to use for coding the PDF
    p.add_argument("--model", "-m", default=None, help="LiteLLM model string (required)")

    # variables: CSV file containing the variables to code
    p.add_argument("--variables", default="parameters.csv", help="Variables CSV (default: parameters.csv)")

    # codes: CSV file containing the codes for the variables
    p.add_argument("--codes", default="codes.csv", help="Codes CSV (default: codes.csv)")

    # ids: Comma-separated list of variable IDs to code
    p.add_argument("--ids", help="Comma-separated list of variable IDs to code (e.g. 2,3,5)")

    # api-base: Override API base URL for the LLM
    p.add_argument("--api-base", default=None, help="Override API base URL")

    # max-chars: Truncate PDF text to this many characters
    p.add_argument("--max-chars", type=int, default=None, help="Truncate PDF text to this many characters")
    
    # context-budget: Token budget per request
    p.add_argument(
        "--context-budget",
        type=int,
        default=None,
        help="Token budget per request. When set, variables are coded in batches that fit "
        "this budget (reusing a cached PDF prefix). Omit for a single request with all variables.",
    )

    # response-reserve-per-var: Estimated output tokens reserved per variable when packing batches
    p.add_argument(
        "--response-reserve-per-var",
        type=int,
        default=250,
        help="Estimated output tokens reserved per variable when packing batches (default: 250)",
    )

    # print-prompt: Print the full prompt and exit
    p.add_argument("--print-prompt", action="store_true", help="Print the full prompt and exit")

    # dump: Dump messages JSON and exit
    p.add_argument("--dump", action="store_true", help="Dump messages JSON and exit")

    # Set the default function to run when this subparser is used
    p.set_defaults(func=_run)


def _run(args) -> None:
    """
    Code the specified PDF using the provided variables and codes.
    This is what is run when calling `ethnocoder code` from the command line.

    Args:
        args: Command-line arguments parsed by argparse.
    """

    # Get the PDF path from the command-line arguments and check if it exists
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[red]Error: {pdf_path} not found[/red]", file=sys.stderr)
        sys.exit(1)

    # Get variables and codes from the provided paths
    all_variables = load_variables(Path(args.variables))
    codes_by_var = load_codes(Path(args.codes))

    # Subset the variables if necessary
    variables = all_variables
    if args.ids:
        id_set = {i.strip() for i in args.ids.split(",")}
        variables = [v for v in variables if v["ID"] in id_set]

        # Error if no variables match the provided IDs
        if not variables:
            print(f"[red]Error: No variables found with IDs: {args.ids}[/red]", file=sys.stderr)
            sys.exit(1)

    # Count the number of option-type variables for reporting
    option_count = sum(1 for v in variables if v.get("Datatype", "Option") == "Option")
    print(
        f"[blue]Loaded {len(variables)} variables ({option_count} option-type) from {args.variables}[/blue]",
        file=sys.stderr,
    )

    # If we are printing the prompt, do so and return
    if args.print_prompt:
        coding_prompt = build_coding_prompt(variables, codes_by_var)
        print("=" * 60, "SYSTEM PROMPT", "=" * 60)
        print(load_prompt(PROMPT_FILE))
        print("=" * 60, "CODING PROMPT", "=" * 60)
        print(coding_prompt)
        return

    # If we are dumping the messages, do so and return
    if args.dump:
        pdf_text = extract_pdf_text(pdf_path, max_chars=args.max_chars)
        pdf_full = f"Source document: {pdf_path.stem}\n\n{pdf_text}"
        messages: list[dict] = [{"role": "system", "content": load_prompt(PROMPT_FILE)}]
        coding_prompt = build_coding_prompt(variables, codes_by_var)
        messages.append({"role": "user", "content": f"{pdf_full}\n\n---\n\n{coding_prompt}"})
        print(json.dumps(messages, indent=2))
        return

    # Code the PDF using the provided variables and codes
    codings = code_pdf(
        pdf_path,
        variables,
        codes_by_var,
        model=args.model,
        api_base=args.api_base,
        max_chars=args.max_chars,
        context_budget=args.context_budget,
        response_reserve_per_var=args.response_reserve_per_var,
    )

    # # Save the codings to a JSON file and report the location
    # json_path = Path(model_dirname(args.model)) / f"{pdf_path.stem}.json"
    # json_path.write_text(json.dumps({"codings": codings}, indent=2), encoding="utf-8")
    # print(f"[green]JSON saved → {json_path}[/green]", file=sys.stderr)
