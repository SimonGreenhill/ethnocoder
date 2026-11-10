# The general flow is:
# _run -> code_pdf -> code_document -> code_batch -> llm_stream().

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import cast
import itertools

import pymupdf
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
from rich import print

from ethnocoder.models import llm_stream, get_max_context
from ethnocoder.utils import parse_codings, get_timestamp

# Globals
PROMPT_FILE = Path('.') / "PROMPT.md"

class Coder():
    def __init__(self, args = None):
        
        # Set args as class attribute for easy access.
        self.args = args

        # Set useful attributes from args for convenience.
        # Arguments to be passed to the LLM.
        if hasattr(args, "temperature"):
            self.temperature = args.temperature
        if hasattr(args, "top_p"):
            self.top_p = args.top_p

    def run(self):
        """
        Code the specified PDF using the provided variables and codes.
        This is what is run when calling `ethnocoder code` from the command line.
        """

        # Get the PDF path from the command-line arguments and check if it exists
        pdf_path = Path(self.args.pdf)
        if not pdf_path.exists():
            print(f"[red]Error: {pdf_path} not found[/red]", file=sys.stderr)
            sys.exit(1)
    
        # Get variables and codes from the provided paths
        all_variables = self.load_variables(Path(self.args.variables))
        codes_by_var = self.load_codes(Path(self.args.codes))
    
        # Subset the variables if necessary
        variables = all_variables
        if self.args.ids:
            id_set = {i.strip() for i in self.args.ids.split(",")}
            variables = [v for v in variables if v["ID"] in id_set]
    
            # Error if no variables match the provided IDs
            if not variables:
                print(f"[red]Error: No variables found with IDs: {self.args.ids}[/red]", file=sys.stderr)
                sys.exit(1)
    
        # Count the number of option-type variables for reporting
        option_count = sum(1 for v in variables if v.get("Datatype", "Option") == "Option")
        print(
            f"[blue]Loaded {len(variables)} variables ({option_count} option-type) from {self.args.variables}[/blue]",
            file=sys.stderr,
        )
    
        # If we are printing the prompt, do so and return
        if self.args.print_prompt:
            coding_prompt = self.build_coding_prompt(variables, codes_by_var)
            print("=" * 60, "SYSTEM PROMPT", "=" * 60)
            print(self.load_prompt(PROMPT_FILE))
            print("=" * 60, "CODING PROMPT", "=" * 60)
            print(coding_prompt)
            return
    
        # If we are dumping the messages, do so and return
        if self.args.dump:
            # pdf_text = extract_pdf_text(pdf_path, max_chars=args.max_chars)
            # pdf_full = f"Source document: {pdf_path.stem}\n\n{pdf_text}"
            # messages: list[dict] = [{"role": "system", "content": load_prompt(PROMPT_FILE)}]
            # coding_prompt = build_coding_prompt(variables, codes_by_var)
            # messages.append({"role": "user", "content": f"{pdf_full}\n\n---\n\n{coding_prompt}"})
            # print(json.dumps(messages, indent=2))
            print("[yellow]Dumping messages is currently disabled. Need to factorise prompt creation first.[/yellow]", file=sys.stderr)
            return
    
        if self.args.pdf_size:
            pdf_text = self.extract_pdf_text(pdf_path, max_chars=self.args.max_chars)
            token_estimate = self.estimate_tokens(pdf_text)
            print(f"[blue]Estimated token count of PDF file for {pdf_path.stem}: {token_estimate}[/blue]", file=sys.stderr)
            return
    
        # Code the PDF using the provided variables and codes
        codings = self.code_pdf(
            pdf_path,
            variables,
            codes_by_var,
            model=self.args.model,
            api_base=self.args.api_base,
            max_chars=self.args.max_chars,
            context_budget=self.args.context_budget,
            response_reserve_per_var=self.args.response_reserve_per_var,
            bool_review=self.args.bool_review,
            bool_thinking=self.args.bool_thinking,
            bool_debug_batches=self.args.bool_debug_batches,
        )

    def load_prompt(self,path: Path) -> str:
        """
        Load the prompt text from the specified file path.

        Args:
            path (Path): The path to the prompt file.

        Returns:
            str: The contents of the prompt file.
        """
        with open(path, encoding="utf-8") as f:
            return f.read()

    def load_variables(self, path: Path) -> list[dict]:
        """
        Load variables data from a CSV file.

        Args:
            path (Path): The path to the CSV file containing variables data.

        Returns:
            list[dict]: A list of dictionaries representing the variables.
        """

        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def load_codes(
            self,
            path: Path,
            bool_use_question_mark: bool = True,
        ) -> dict[str, list[dict]]:
        """
        Load codes data from a CSV file and organize them by variable ID.

        Args:
            path (Path): The path to the CSV file containing codes data.
            bool_use_question_mark (bool): Whether to include the "?" code as valid, to mean "no evidence found". Defaults to True.

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

        # If bool_use_question_mark is True, add the "?" code to each variable's list of codes if it's not already present.
        if bool_use_question_mark:
            for var_id, codes in codes_by_var.items():
                if not any(code["Name"] == "?" for code in codes):
                    codes.append({"ID": f"{var_id}-?", "Parameter_ID": var_id, "Name": "?", "Description": "No evidence found"})

        return codes_by_var

    def extract_pdf_text(self, pdf_path: Path, max_chars: int | None = None) -> str:
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

    def build_coding_prompt(self, variables: list[dict], codes_by_var: dict[str, list]) -> str:
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
                lines.append("") # Newline after section heading.
                current_section = section

            # Get basic details for variable
            var_id = var["ID"]
            datatype = var.get("Datatype") or "Option"
            name = var["Name"]
            description = (var.get("Description") or "").strip()

            # Add the variable ID and name to the prompt
            lines.append(f"# {var_id}")
            lines.append(name)
            lines.append("") # Newline after variable name.

            # Add valid codes.
            lines.append("## Valid codes")

            # Different datatypes require slightly different prompt text.
            # The "Option" type requires listing valid codes and instructing to assign one code from the list.
            if datatype == "Option":

                # Get the possible codes for this variable
                var_codes = sorted(codes_by_var.get(var_id, []), key=lambda c: c["Name"])
                if var_codes:

                    # Format the codes as "Name=Description" strings, truncating descriptions to 100 characters.
                    for c in var_codes:
                        lines.append(f"{c['Name']} = {c['Description'][:100].rstrip()}")

                    # Newline after listing valid codes.
                    lines.append("")

                else:

                    # Variable should have codes but they weren't found.
                    print(f"[red]Error: No valid codes found for variable {var_id}")
                    sys.exit(1)
            
            elif datatype == "Int":
                # For integer variables, instruct to assign an integer value.
                lines.append("Assign an integer value.")

            elif datatype == "Float":
                # For float variables, instruct to assign a numeric value.
                lines.append("Assign a numeric value.")

            else:
                # For other datatypes, instruct to provide a brief text value.
                lines.append("Provide a brief text value.")

            # If there's a description, add it below the valid codes section.
            if description:
                lines.append(description)

            lines.append("")

        return "\n".join(lines)

    def estimate_tokens(self, text: str) -> int:
        """
        A rough estimate of the number of tokens in the given text.

        Args:
            text (str): The input text to estimate token count for.

        Returns:
            int: Estimated number of tokens in the text.
        """

        return len(text) // 4

    def pack_batches(
        self, 
        variables: list,
        var_size: dict,
        prefix_tokens: int,
        context_budget: int,
        response_reserve_per_var: int,
    ) -> list[list]:
        """
        Pack items into batches based on their token sizes and context budget.

        Args:
            variables (list): List of variables to be batched.
            var_size (dict): Dictionary mapping variable IDs to their estimated token sizes.
            prefix_tokens (int): Number of tokens reserved for the prefix.
            context_budget (int): Maximum allowed tokens for a batch, including prefix.
            response_reserve_per_var (int): Tokens reserved per variable for the response.

        Returns:
            list[list]: List of batches, each containing a subset of variables.
        """

        # Initialize the list of batches and the current batch.
        batches: list[list] = []
        current: list = []

        # Initialise current tokens: the prefix size.
        current_tokens = prefix_tokens

        # Load the feature grouping csv.
        feature_grouping = {}
        with open("feature_grouping_for_analysis.csv", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # feature_grouping[row["Feature_ID"]] = row["Finer_grouping"] # TODO: toggle switch these
                feature_grouping[row["Feature_ID"]] = row["Main_domain"]

        # Any Feature_ID's that end in a letter, put them together and ensure they have the same Main_domain.
        grouped_features = {}
        for feature_id, main_domain in feature_grouping.items():

            # If the feature ID ends in a letter...
            if feature_id[-1].isalpha():

                # ...get the base ID by removing the trailing letter...
                base_id = feature_id[:-1]

                # ...and it to the master dict...
                if base_id not in grouped_features:
                    grouped_features[base_id] = main_domain

                # ...but if it exists and has a different main domain, raise an error.
                elif grouped_features[base_id] != main_domain:
                    raise ValueError(
                        f"Feature_IDs ending with a letter must have the same Main_domain. Conflict for base ID {base_id}."
                    )
            else:

                # Normal ID, add it to the master list.
                grouped_features[feature_id] = main_domain

        # For items that are not one of the keys in grouped_features, add them with main domain "OTHER_MISC".
        sum_misc = 0
        for variable in variables:
            if variable["ID"] not in grouped_features:
                grouped_features[variable["ID"]] = "OTHER_MISC"
                sum_misc += 1

        if sum_misc > 0:
            print(f"[blue]Found {sum_misc} uncategorised variables, adding to 'OTHER_MISC' group.[/blue]")

        # Sort grouped_features by main domain, then by feature ID.
        grouped_features = dict(sorted(grouped_features.items(), key=lambda x: (x[1], x[0])))

        # Get the variable indices for quick lookup.
        variable_indices = {variable["ID"]: i for i, variable in enumerate(variables)}

        # Remove variables from grouped_features that are not present in the variables list.
        for variable_id in list(grouped_features.keys()):
            if variable_id not in variable_indices:
                del grouped_features[variable_id]

        # Loop at each group.
        for main_domain, group_items in itertools.groupby(grouped_features.items(), key=lambda x: x[1]):

            # Loop at each item in the current group.
            for variable_id, main_domain in group_items:

                # Get the size of this variable from the var_size dictionary.
                size = var_size[variable_id]

            # # For each item and its corresponding size, determine if it fits in the current batch 
            # # or if a new batch is needed.
            # for variable, size in zip(variables, [var_size[i["ID"]] for i in variables]):

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
                # We add the entry of <variables> associated with this <variable_id>.
                # A bit clunky for backwards compatibility.
                current.append(variables[variable_indices[variable_id]])
                current_tokens += cost

            # Append the current batch to the list of batches if it contains any items.
            if current:
                batches.append(current)

            # Reset the current batch and token count for the next main domain.
            current = []
            current_tokens = prefix_tokens

        # Append the last batch if it contains any items.
        if current:
            batches.append(current)
            
        # Return the list of batches.
        return batches

    def clean_codings(self, codings: list[dict]) -> list[dict]:
        """
        Clean the codings by removing any internal keys that start with an underscore.

        Args:
            codings (list[dict]): List of coding dictionaries.

        Returns:
            list[dict]: List of cleaned coding dictionaries.
        """

        return [{k: v for k, v in c.items() if not k.startswith("_")} for c in codings]

    def validate_response(
        self,
        codings: list[dict],
        codes_by_var: dict[str, list[dict]],
        variables: list[dict],
        pdf_full: str,
    ) -> list[dict]:
        """
        Validate the response from the LLM.
        1. Check that the codes assigned to option variables are among the allowed codes.
        Marks codings with "_invalid" and "_valid_codes" if the code is not valid.
        2. Check that all variables requested are in the response, and that no extra variables are included.

        Args:
            codings (list[dict]): List of coding dictionaries.
            codes_by_var (dict[str, list[dict]]): Dictionary mapping variable IDs to their allowed codes.
            variables (list[dict]): List of variable dictionaries.
            pdf_full (str): Full text of the PDF being coded.

        Returns:
            list[dict]: List of validated coding dictionaries with "_invalid" and "_valid_codes" keys added where applicable.
        """

        # If the JSON was invalid, simply tell the LLM to repeat the task.
        if len(codings) == 1 and "message" in codings[0] and "Invalid JSON" in codings[0]["message"]:
            return [{"invalid_json": True}]

        # # Replace all runs of whitespace, newlines etc. Replace each with a single space.
        # # Note that this is just for quote-searching, not for sending to the LLM;
        # # that alteration is done slightly differently to retain some of the newline structure.
        # pdf_full_normalized = " ".join(pdf_full.split())

        # Set option variables
        option_vars = {v["ID"] for v in variables if v.get("Datatype", "Option") == "Option"}

        # Build a dictionary of valid codes for each option variable
        valid: dict[str, set[str]] = {
            vid: {c["Name"] for c in codes}
            for vid, codes in codes_by_var.items()
            if vid in option_vars
        }

        # Initialise a list of requested variable IDs, and those returned.
        variables_requested = set(v["ID"] for v in variables)
        variables_returned = set()

        # Initialize the result list to store validated codings
        result = []
        for c in codings:

            # Get the variable ID and the assigned code.
            vid = c.get("id", None)
            code = c.get("code")

            if vid is None:
                continue # Skip codings with no variable ID. Possibly system message that will be dealt with elsewhere.

            # Convert ID to string.
            vid = str(vid)

            # Was this variable requested?
            if vid not in variables_requested:
                c = dict(c, _superfluous=True)
                result.append(c)
                continue

            # Has this variable been returned before? If so, it is a duplicate.
            if vid in variables_returned:
                c = dict(c, _duplicate=True)
                result.append(c)
                continue

            # Add this variable to the set of requested variables, as it has been returned.
            variables_returned.add(vid)

            # If the code is not in the list of valid codes, mark it as invalid and store the valid codes.
            if vid in valid and str(code) not in valid[vid]:
                c = dict(c, _invalid=True, _valid_codes=sorted(valid[vid]))

            # # Check whether the quote exists in the PDF text.
            # quote = c.get("quote")

            # # If quote is the string "null", treat it as None.
            # if quote == "null":
            #     quote = c["quote"] = None

            # # If the quote is not None, and doesn't contain ellipses, check that it exists in the PDF text.
            # if quote is not None and "..." not in quote:

            #     # Normalise the quote before searching in the PDF text.
            #     quote = " ".join(quote.split())

            #     # Look for the normalized quote in the normalized PDF text.
            #     if pdf_full_normalized.find(quote) == -1:
            #         c = dict(c, _quote_missing=True)

            # If "pages" exists and is the string "null", change it to None.
            if "pages" in c and c["pages"] == "null":
                c["pages"] = None

            # Append to the output.
            result.append(c)

        # Get any variables from variables_requested that were not returned.
        missing_variables = variables_requested - variables_returned
        for vid in missing_variables:
            result.append({"id": vid, "_missing": True})

        return result

    def build_review_message(
            self,
            codings: list[dict],
        ) -> tuple[str | None, dict]:
        """
        Prepare a review message for the codings, highlighting invalid codes and low confidence assignments.
        Possible errors are:
            - Invalid codes (_invalid, comes with _valid_codes)
            - Missing variables (_missing)
            - Duplicate variables (_duplicate)
            - Superfluous variables (_superfluous)

        Args:
            codings (list[dict]): List of coding dictionaries.

        Returns:
            tuple[str | None, dict]: A tuple containing the review message (or None if no issues), and a dictionary with counts of different types of codings.
            The dictionary has the following keys:
            - n_missing: Number of missing variables.
            - n_superfluous: Number of superfluous variables.
            - n_duplicate: Number of duplicate variables.
            - n_invalid: Number of invalid codes.
            - n_low_confidence: Number of low confidence codings.
            - n_quote_missing: Number of codings with quotes missing in the PDF text.
        """

        # Initialise return dictionary
        review_stats = {
            "n_missing": 0,
            "n_superfluous": 0,
            "n_duplicate": 0,
            "n_invalid": 0,
            "n_low_confidence": 0,
            "n_quote_missing": 0,
        }

        # If codings is a single list with entry {"invalid_json": True}, return a special message.
        if len(codings) == 1 and codings[0].get("invalid_json"):
            return "Invalid JSON detected. Please resubmit the coding.", review_stats

        # Find the missing codings.
        missing = [c for c in codings if c.get("_missing")]
        review_stats["n_missing"] = len(missing)

        # Find the superfluous codings.
        superfluous = [c for c in codings if c.get("_superfluous")]
        review_stats["n_superfluous"] = len(superfluous)

        # Find the duplicate codings.
        duplicate = [c for c in codings if c.get("_duplicate")]
        review_stats["n_duplicate"] = len(duplicate)

        # Find the invalid codings.
        invalid = [c for c in codings if c.get("_invalid")]
        review_stats["n_invalid"] = len(invalid)

        # Find the low confidence codings.
        low_conf = [
            c for c in codings
            if (c.get("confidence") == "low")
            and not c.get("_invalid") and not c.get("_missing") and not c.get("_superfluous") and not c.get("_duplicate")
        ]
        review_stats["n_low_confidence"] = len(low_conf)

        # Find the codings with quotes that were not found in the PDF text.
        quote_missing = [c for c in codings if c.get("_quote_missing")]
        review_stats["n_quote_missing"] = len(quote_missing)

        # If everything looks good, return now.
        if not missing and not superfluous and not duplicate and not invalid and not low_conf and not quote_missing:
            return None, review_stats

        # Prepare the review message parts.
        parts: list[str] = []

        # Missing variables.
        if missing:

            # Add a line listing all missing variable IDs.
            lines = ["The following variables are missing. If there is no textual evidence for them in the document, set their values to '?':"]
            lines.append(",".join(str(c['id']) for c in missing))

            # Newline
            parts.append("\n".join(lines))

        # Superfluous variables.
        if superfluous:

            # Add a line listing all superfluous variable IDs.
            lines = ["The following variables are superfluous and should be removed:"]
            lines.append(",".join(str(c['id']) for c in superfluous))
            
            # Newline
            parts.append("\n".join(lines))

        # Duplicate variables.
        if duplicate:

            # Add a line listing all duplicate variable IDs.
            lines = ["The following variables are duplicated and should be corrected:"]
            lines.append(",".join(str(c['id']) for c in duplicate))
            
            # Newline
            parts.append("\n".join(lines))

        # Invalid codings.
        if invalid:

            # List all invalid codings with explanations.
            lines = ["The following variables have invalid codes that must be corrected:"]
            
            for c in invalid:

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
                f"For variables {ids} you assigned low confidence. "
                "Re-read the document carefully for any overlooked evidence and correct these if possible."
            )

        # Quote missing codings.
        if quote_missing:
            ids = ", ".join(str(c["id"]) for c in quote_missing)
            parts.append(
                f"The following variables have quotes that were not found in the PDF text: {ids}"
            )

        # Combine all parts into the final review message.
        review_msg = "\n\n".join(parts) + (
            "\n\nReturn the complete corrected codings for ALL variables in the same JSON format."
        )

        # Return.
        return review_msg, review_stats

    def code_batch(
        self,
        variables: list[dict],
        codes_by_var: dict[str, list[dict]],
        model: str,
        api_base: str | None,
        is_anthropic: bool,
        system_msg: dict,
        pdf_full: str,
        context_budget: int,
        batch_index: int,
        bool_review: bool = False,
        bool_thinking: bool = False,
    ) -> tuple[list[dict], str, dict, str, str | None]:
        """
        Code one batch of variables in a fresh conversation.

        The prefix [system_msg] is identical across batches so a local
        server can reuse its KV cache for the PDF instead of reprocessing it.
        
        Args:
            variables: A list of variable dictionaries to be coded.
            codes_by_var: A dictionary mapping variable IDs to their corresponding list of code dictionaries.
            model: The model name to use for the LLM.
            api_base: The base URL for the API, if any.
            is_anthropic: A boolean indicating if the model is Anthropic.
            system_msg: The system message dictionary for the conversation.
            pdf_full: The entire PDF text, prefaced by the PDF document name.
            context_budget: The token budget for the LLM context.
            batch_index: The index of the current batch being processed.
            bool_review: A boolean indicating if a review pass should be performed when necessary.
            bool_thinking: A boolean indicating if the LLM should display its thinking process.

        Returns:
            A tuple:
            - list of cleaned coding dictionaries
            - the final raw response text
            - the metadata dictionary
            - the user content
            - the review message (or None if no review was performed)
            The metadata dictionary contains information about the LLM response, including elapsed time and model load status for both the initial and review passes.
        """

        # Create the coding prompt for the current batch of variables.
        # This is the user prompt.
        coding_prompt = self.build_coding_prompt(variables, codes_by_var)
        user_content = coding_prompt
        # user_content = f"{pdf_full}\n\n---\n\n{coding_prompt}" if pdf_full else coding_prompt

        # Create the initial message list by combining the system message and the user prompt.
        messages = [system_msg, {"role": "user", "content": user_content}]

        # Stream the LLM response for the current batch of variables.
        text, meta = llm_stream(
            messages = messages,
            model = model,
            is_anthropic = is_anthropic,
            api_base = api_base,
            context_budget = context_budget,
            bool_thinking = bool_thinking,
            temperature = self.temperature,
            top_p = self.top_p,
            bool_check_demand = True if batch_index == 1 else False, # check demand on first call only
        )

        # Add requested variables to meta
        meta["variables"] = [var["ID"] for var in variables]

        # Parse the response text into coding dictionaries.
        codings = parse_codings(text) or []

        # Validate the response against the provided codes by variable and the current batch of variables.
        codings = self.validate_response(
            codings = codings,
            codes_by_var = codes_by_var,
            variables = variables,
            pdf_full = pdf_full,
        )

        # Review if requested
        review_msg = None

        if bool_review:

            # Check if we need a review pass based on the validity and confidence of the codings.
            review_msg, review_stats = self.build_review_message(codings)

        # If no review is needed, return the cleaned codings and the original response text.
        if review_msg is None:
            return self.clean_codings(codings), text, meta, user_content, None

        # We will do a review pass.
        print(
            f"[magenta]  → Review pass [/magenta]",
            file=sys.stderr,
        )

        # State the problem(s) that have triggered the review.
        # Invalid JSON
        if codings and codings[0].get("invalid_json"):
            print(
                f"[magenta]  → Invalid JSON detected[/magenta]",
                file=sys.stderr,
            )

        # Missing codes
        if review_stats['n_missing']:
            print(
                f"[magenta]  → {review_stats['n_missing']} missing codings[/magenta]",
                file=sys.stderr,
            )
        # Superfluous codings
        if review_stats['n_superfluous']:
            print(
                f"[magenta]  → {review_stats['n_superfluous']} superfluous codings[/magenta]",
                file=sys.stderr,
            )
        # Duplicate codings
        if review_stats['n_duplicate']:
            print(
                f"[magenta]  → {review_stats['n_duplicate']} duplicate codings[/magenta]",
                file=sys.stderr,
            )
        # Invalid codings
        if review_stats['n_invalid']:
            print(
                f"[magenta]  → {review_stats['n_invalid']} invalid codings[/magenta]",
                file=sys.stderr,
            )
        # Low confidence codings
        if review_stats['n_low_confidence']:
            print(
                f"[magenta]  → {review_stats['n_low_confidence']} low confidence codings[/magenta]",
                file=sys.stderr,
            )
        # Codings with quotes not found in the PDF text
        if review_stats['n_quote_missing']:
            print(
                f"[magenta]  → {review_stats['n_quote_missing']} codings with quotes not found in the PDF text[/magenta]",
                file=sys.stderr,
            )

        # Append the previous response (or the invalid JSON error message) along with the review message.
        messages.append({"role": "assistant", "content": json.dumps({"codings": self.clean_codings(codings)}, indent=2)})
        messages.append({"role": "user", "content": review_msg})

        # Ask the LLM to do better.
        text2, meta2 = llm_stream(
            messages = messages,
            model = model,
            is_anthropic = is_anthropic,
            api_base = api_base,
            context_budget = context_budget,
            bool_thinking = bool_thinking,
            temperature = self.temperature,
            top_p = self.top_p,
        )

        # combine meta and meta2 into a single dictionary
        combined_meta = {
            **meta,
            "elapsed_2": meta2.get("elapsed", 0),
            "model_loaded_2": meta2.get("model_loaded", False)
        }

        # We won't try for another review, just return the cleaned codings from this pass.
        return self.clean_codings(parse_codings(text2) or []), text2, combined_meta, user_content, review_msg

    def code_document(
        self,
        pdf_stem: str,
        variables: list[dict],
        codes_by_var: dict[str, list[dict]],
        model: str,
        api_base: str | None,
        is_anthropic: bool,
        messages: list[dict],
        context_budget: int,
        pdf_full: str = "",
        out_dir: Path = Path("."),
        response_reserve_per_var: int = 250,
        bool_review: bool = False,
        bool_thinking: bool = False,
        bool_debug_batches: bool = False,
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
            bool_review: Whether to perform a review pass when necessary.
            bool_thinking: Whether to enable thinking mode for the LLM.
            bool_debug_batches: Whether to output debug information for batches without coding them.
        Returns:
            A list of coding dictionaries for all variables.
        """

        # Get the system message: it's the first in the list.
        system_msg = messages[0]

        # Estimate the number of tokens used by the system message and PDF file.
        prefix_tokens = self.estimate_tokens(system_msg["content"]) # + estimate_tokens(pdf_full)

        # Get a dict storing estimated number of tokens per variable definition
        var_size = {
            v["ID"]: self.estimate_tokens(self.build_coding_prompt([v], codes_by_var))
            for v in variables
        }

        # Pack variables into batches based on their estimated token sizes and the context budget
        try:
            batches = self.pack_batches(
                variables = variables,
                var_size = var_size,
                prefix_tokens = prefix_tokens,
                context_budget = context_budget * 0.9, # Reserve 10% of the context budget for safety margin
                response_reserve_per_var = response_reserve_per_var,
            )
        except ValueError as e:
            sys.exit(f"Error: {e}")

        # Initialise output for codings and raw text parts
        all_codings: list[dict] = []
        raw_parts: list[str] = []

        # Are there multiple batches?
        multi = len(batches) > 1

        # Get the size of the numeral representing the total number of batches.
        # This helps filename zero padding for batch indices below.
        width = len(str(len(batches)))

        # Output the system message: it should NOT change across batches.
        path_system_message = out_dir / f"{pdf_stem}_system_message.md"
        path_system_message.write_text(system_msg["content"], encoding="utf-8")

        # Loop over each batch and code it
        for i, batch in enumerate(batches, 1):

            # If there are multiple batches, print batch information
            if multi:
                batch_tokens = prefix_tokens + sum(var_size[v["ID"]] for v in batch)
                print(
                    f"[blue]Batch {i}/{len(batches)} ({len(batch)} vars, ~{batch_tokens // 1000}k tok)…[/blue]",
                    file=sys.stderr,
                )

            # If we are debugging batches, output batch info and continue.
            if bool_debug_batches:

                # Create output path for batch debug information
                batch_debug_path = out_dir / f"{pdf_stem}_{i:0{width}}_batch_debug.txt"

                # Output batch info
                batch_debug_path.write_text(
                    f"Batch {i} of {len(batches)}\nVariables: {[v['ID'] for v in batch]}\n",
                    encoding="utf-8"
                )

                print(f"    [green]DEBUG BATCHES: output file {batch_debug_path}[/green]")

                # Continue: when debugging batches, we do not code the batch, just output debug info.
                continue

            # Code this batch
            clean, text, meta, user_content, review_msg = self.code_batch(
                variables = batch, 
                codes_by_var = codes_by_var,
                model = model,
                api_base = api_base,
                is_anthropic = is_anthropic,
                system_msg = system_msg,
                pdf_full = pdf_full,
                context_budget = context_budget,
                bool_review = bool_review,
                bool_thinking = bool_thinking,
                batch_index = i,
            )

            # Add results from this batch to the overall output
            all_codings.extend(clean)

            # Append the raw text for this batch to the list of raw parts, with a separator stating which batch it is
            raw_parts.append(f"=== batch {i} ===\n{text}" if multi else text)

            # Prepare output paths, zero-padded with batch index
            txt_path = out_dir / f"{pdf_stem}_{i:0{width}}.txt"
            json_path = out_dir / f"{pdf_stem}_{i:0{width}}.json"
            meta_path = out_dir / f"{pdf_stem}_{i:0{width}}.log"
            prompt_path = out_dir / f"{pdf_stem}_{i:0{width}}_prompt.md"
            review_path = out_dir / f"{pdf_stem}_{i:0{width}}_review.md"

            # Output raw txt file and json file
            txt_path.write_text("\n\n".join(raw_parts), encoding="utf-8")
            json_path.write_text(json.dumps({"codings": all_codings}, indent=2), encoding="utf-8")

            # Output meta information for this batch
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            # Output prompt and review files if available
            if prompt_path and user_content:
                prompt_path.write_text(user_content, encoding="utf-8")
            if review_path and review_msg:
                review_path.write_text(review_msg, encoding="utf-8")

            # Inform the user
            print(f"  [green]JSON saved → {json_path}[/green]", file=sys.stderr)

            # Figure out from meta how long it took.
            # If less than 5s and api_base begins "openai/", pause for 5s because we're allowed 2 requests per 10 seconds.
            try:
                if api_base and api_base.startswith("openai/") and "elapsed" in meta and float(meta["elapsed"]) < 5:
                    print(f"  [yellow]Pausing for 5s to respect provider rate limit...[/yellow]", file=sys.stderr)
                    time.sleep(5)
            except Exception as e:
                print(f"  [yellow]Warning: Failed to enforce provider rate limit: {e}[/yellow]", file=sys.stderr)

        return all_codings

    def model_dirname(self, model: str) -> str:
        """
        Return the directory name for storing model outputs.

        The directory is constructed as "output/{timestamp}/{model_name}/{os}", where colons in the model name are replaced with underscores.

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

        # Get timestamp
        ts = get_timestamp()

        ts_date, ts_time = ts.split("_")

        out_dir = os.path.join(
            "output",
            ts_date,
            f"{ts_time}_{model.split('/')[-1].replace(':', '_')}_{os_name}",
        )

        return out_dir

    def resolve_model_config(self, model: str, api_base: str | None) -> tuple[bool, str | None]:
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

    def validate_request(self, model: str | None, context_budget: int) -> None:
        """
        Validate the request parameters for coding a PDF.

        Args:
            model: The model string.
            context_budget: Token budget per request.

        Raises:
            SystemExit: If validation fails.
        """

        if model is None:
            print("[red]Error: --model is required[/red]", file=sys.stderr)
            sys.exit(1)

        if context_budget <= 0:
            print("[red]Error: context_budget must be positive[/red]", file=sys.stderr)
            sys.exit(1)

        # Check if context_budget exceeds the maximum context for the model
        max_context = get_max_context(model)
        if max_context is None:
            print(f"[yellow]Warning: Unable to determine the maximum context for the model[/yellow]", file=sys.stderr)
        elif context_budget > max_context:
            print(f"[red]Error: context_budget {context_budget} exceeds the maximum context for the model ({max_context})[/red]", file=sys.stderr)
            sys.exit(1)

        # Inform the user that the budget is OK
        print(f"[blue]Using context budget of {context_budget:,} tokens; model context window is {max_context:,}[/blue]", file=sys.stderr)

    def code_pdf(
        self,
        pdf_path: Path,
        variables: list[dict],
        codes_by_var: dict[str, list[dict]],
        model: str | None,
        context_budget: int,
        api_base: str | None = None,
        max_chars: int | None = None,
        response_reserve_per_var: int = 250,
        bool_review: bool = False,
        bool_thinking: bool = False,
        bool_debug_batches: bool = False,
    ) -> list[dict]:
        """
        Code a PDF for trait/feature variables using a local or remote LLM.

        Args:
            pdf_path: Path to the PDF file.
            variables: List of variable dictionaries.
            codes_by_var: Dictionary mapping variable IDs to lists of code dictionaries.
            model: Model string (required).
            context_budget: Token budget per request.
            api_base: API base URL, if any.
            max_chars: Maximum number of characters to extract from the PDF.
            response_reserve_per_var: Reserved tokens per variable in the response.
            bool_review: Whether to perform a review of the coding results, if there are invalid or low-confidence codings.
            bool_thinking: Whether to enable thinking mode for the LLM.
            bool_debug_batches: Whether to output debug information for batches without coding them.

        Returns:
            List of coding results as dictionaries.
        """

        # Perform basic validation
        self.validate_request(
            model = model,
            context_budget = context_budget,
        )

        # Error if no model supplied
        if model is None:
            print("[red]Error: --model is required[/red]", file=sys.stderr)
            sys.exit(1)

        # Inform the user
        print(f"[blue]Extracting text from {pdf_path.name} ({pdf_path.stat().st_size / 1e6:.1f} MB)…[/blue]", file=sys.stderr)

        # Get PDF text
        pdf_text = self.extract_pdf_text(pdf_path, max_chars=max_chars)

        # Get model config
        is_anthropic, api_base = self.resolve_model_config(model, api_base)

        # Set up inputs for LLM
        pdf_full = f"{pdf_path.stem}\n\n{pdf_text}"
        # pdf_full = f"Source document: {pdf_path.stem}\n\n{pdf_text}"
        messages = [{"role": "system", "content": self.load_prompt(PROMPT_FILE)}]

        # The system message includes the full PDF content at the end
        messages[0]["content"] += f"\n\n{pdf_full}"

        # Prepare output directory for coded results
        out_dir = Path(self.model_dirname(model))
        print(f"[blue]Preparing output directory {out_dir}…[/blue]", file=sys.stderr)
        out_dir.mkdir(exist_ok=True, parents=True)

        # Inform the user and code the document
        infotext = f"[blue]Coding [magenta]{pdf_path.name}[/magenta] ({len(variables)} variables) with [green bold]{model}[/green bold]"
        if bool_review:
            infotext += " and[/blue] [magenta]review pass[/magenta] [blue]enabled"
        print(f"{infotext}...[/blue]", file=sys.stderr)
        return self.code_document(
            pdf_stem = pdf_path.stem,
            variables = variables,
            codes_by_var = codes_by_var,
            model = model,
            api_base = api_base,
            is_anthropic = is_anthropic,
            messages = messages,
            pdf_full=pdf_full,
            out_dir=out_dir,
            context_budget=context_budget, 
            response_reserve_per_var=response_reserve_per_var,
            bool_review = bool_review,
            bool_thinking = bool_thinking,
            bool_debug_batches = bool_debug_batches,
        )

# Methods for module
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
        default=65536,
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

    # bool-review: Perform a review of the coding results where necessary
    p.add_argument("--bool-review", action="store_true", help="Perform a review of the coding results where necessary")

    # bool-thinking: Enable thinking mode for the LLM, if allowed by the model
    p.add_argument("--bool-thinking", action="store_true", help="Enable thinking mode for the LLM, if allowed by the model")

    # pdf-size: Estimate tokens of PDF
    p.add_argument("--pdf-size", action="store_true", help="Estimate tokens of PDF")

    # bool-debug-batches: Output debug information for batches without coding them
    p.add_argument("--bool-debug-batches", action="store_true", help="Output debug information for batches without coding them")

    # temperature: Set the temperature for the LLM
    # Default is currently 0.4 because we want more deterministic rather than more creative.
    p.add_argument("--temperature", type=float, default=0.4, help="Set the temperature for the LLM")

    # top-p: Set the top-p (nucleus sampling) for the LLM
    # Default is 0.95, recommended by SAIA for GLM 5.3 Flash.
    p.add_argument("--top-p", type=float, default=0.95, help="Set the top-p (nucleus sampling) for the LLM")

    # Set the default function to run when this subparser is used
    p.set_defaults(func=_run)

def _run(args) -> None:
    """
    Code the specified PDF using the provided variables and codes.
    This is what is run when calling `ethnocoder code` from the command line.

    Args:
        args: Command-line arguments parsed by argparse.
    """

    # Create coder
    coder = Coder(args)

    # Run the coding process
    coder.run()
