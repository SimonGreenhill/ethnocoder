#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "litellm",
#   "pymupdf",
# ]
# ///
"""
Code a PDF source document for cultural traits using an LLM.

Reads variables from ./traits/variables.csv and valid codes from ./traits/codes.csv,
then prompts the model to assign the most appropriate code for each variable based
on the content of a PDF source document. Raw JSON response is always saved to
<pdf_stem>.json for later parsing.

LiteLLM is used for provider-agnostic LLM calls. The --model flag accepts any
LiteLLM model string, e.g.:
    ollama/llama3.2          (local via Ollama)
    ollama/gemma3:27b        (local via Ollama)
    anthropic/claude-opus-4-6
    openai/gpt-4o

Usage:
    python code_traits.py docs/buck1952.pdf --model claude-opus-4-8
    python code_traits.py docs/buck1952.pdf --model ollama/llama3.2
    python code_traits.py docs/buck1952.pdf --model lm_studio/gemma-4-e4b --api-base http://localhost:1234/v1
    python code_traits.py docs/buck1952.pdf --model claude-opus-4-8 --section "Supernatural Beings"
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pymupdf
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
import litellm

PROMPT_FILE = Path('.') / "PROMPT.md"
VARIABLES_CSV = Path('.') / "variables.csv"
CODES_CSV = Path('.') / "codes.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prompt(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_variables(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_codes(path: Path) -> dict[str, list[dict]]:
    """Returns valid codes grouped by variable ID (Parameter_ID)."""
    codes_by_var: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes_by_var.setdefault(row["Parameter_ID"], []).append(row)
    return codes_by_var


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path, max_chars: int | None = None) -> str:
    """Extract plain text from all pages of a PDF, optionally truncated."""
    doc = pymupdf.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    text = "\n\n".join(pages)
    if max_chars and len(text) > max_chars:
        original_len = len(text)
        text = text[:max_chars]
        print(f"Warning: PDF text truncated to {max_chars:,} chars (was {original_len:,})", file=sys.stderr)
    return text


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_coding_prompt(variables: list[dict], codes_by_var: dict[str, list]) -> str:
    lines: list[str] = [
        "Code each variable below based on the source document above.",
        "For each, assign the most appropriate code and briefly justify your choice.",
        "Use confidence 'absent' when the document contains no relevant evidence.\n",
    ]

    current_section = None
    for var in variables:
        section = var.get("Section", "")
        if section != current_section:
            lines.append(f"[{section}]")
            current_section = section

        var_id = var["ID"]
        datatype = var["Datatype"]
        name = var["Name"]
        description = (var.get("Description") or "").strip()

        lines.append(f"ID {var_id}: {name}")

        if description:
            lines.append(f"  {description}")

        if datatype == "Option":
            var_codes = sorted(codes_by_var.get(var_id, []), key=lambda c: c["Name"])
            if var_codes:
                code_strs = [
                    f"{c['Name']}={c['Description'][:100].rstrip()}"
                    for c in var_codes
                ]
                lines.append("  Valid codes: " + " | ".join(code_strs))
            lines.append("  Assign one code from the list above.")
        elif datatype == "Int":
            lines.append("  Assign an integer value.")
        elif datatype == "Float":
            lines.append("  Assign a numeric value.")
        else:  # Text
            lines.append("  Provide a brief text value.")

        lines.append("")

    return "\n".join(lines)


def clean_codings(codings: list[dict]) -> list[dict]:
    """Remove internal annotation keys (those starting with '_') from codings."""
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in codings]


def strip_fences(text: str) -> str:
    """Strip markdown code fences that models add despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]  # remove opening fence line
    if text.endswith("```"):
        text = text[:text.rindex("```")]
    return text.strip()


def parse_codings(text: str) -> list[dict]:
    """Parse an LLM response into a list of coding dicts."""
    if text.startswith("{{"):
        text = text[1:]
    raw = json.loads(text)
    if isinstance(raw, dict) and "raw_response" in raw and "codings" not in raw:
        raw = json.loads(strip_fences(raw["raw_response"]))
    return raw if isinstance(raw, list) else raw.get("codings", [])


def validate_option_codes(
    codings: list[dict],
    codes_by_var: dict[str, list[dict]],
    variables: list[dict],
) -> list[dict]:
    """Return codings with invalid option codes annotated (adds '_invalid' and '_valid_codes' keys)."""
    option_vars = {v["ID"] for v in variables if v["Datatype"] == "Option"}
    valid: dict[str, set[str]] = {
        vid: {c["Name"] for c in codes}
        for vid, codes in codes_by_var.items()
        if vid in option_vars
    }
    result = []
    for c in codings:
        vid = str(c.get("id", ""))
        if vid in valid and str(c.get("code", "")) not in valid[vid]:
            c = dict(c, _invalid=True, _valid_codes=sorted(valid[vid]))
        result.append(c)
    return result


def build_review_message(
    codings: list[dict],
) -> tuple[str | None, int, int]:
    """
    Build a review prompt for invalid or low-confidence codings.

    Returns (review_message, n_invalid, n_low_confidence).
    
    Returns (None, 0, 0) if no review is needed.
    """
    invalid = [c for c in codings if c.get("_invalid")]
    low_conf = [c for c in codings if c.get("confidence") in ("low", "absent") and not c.get("_invalid")]

    if not invalid and not low_conf:
        return None, 0, 0

    parts: list[str] = []
    if invalid:
        lines = ["The following variables have invalid codes that must be corrected:"]
        for c in invalid:
            lines.append(f"  ID {c['id']}: you coded '{c['code']}' but valid codes are: {', '.join(c['_valid_codes'])}")
        parts.append("\n".join(lines))
    if low_conf:
        ids = ", ".join(str(c["id"]) for c in low_conf)
        parts.append(
            f"For variables {ids} you assigned low or absent confidence. "
            "Re-read the document carefully for any overlooked evidence and correct these if possible."
        )
    review_msg = "\n\n".join(parts) + (
        "\n\nReturn the complete corrected codings for ALL variables in the same JSON format."
    )
    return review_msg, len(invalid), len(low_conf)


def llm_stream(
    messages: list[dict],
    model: str,
    is_anthropic: bool,
    api_base: str | None,
) -> str:
    """Make a streaming LLM call and return stripped response text."""
    msgs = messages[:]

    kwargs: dict = {"stream": True}
    if not is_anthropic and not model.startswith("lm_studio/"):
        kwargs["response_format"] = {"type": "json_object"}
    if api_base:
        kwargs["api_base"] = api_base

    chunks: list[str] = []
    response = litellm.completion(model=model, messages=msgs, **kwargs)
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            chunks.append(delta)
            #print(delta, end="", flush=True)
    return strip_fences("".join(chunks))


def code_section(
    pdf_stem: str,
    variables: list[dict],
    codes_by_var: dict[str, list[dict]],
    model: str,
    api_base: str | None,
    is_anthropic: bool,
    messages: list[dict],
    pdf_prefix: str = "",
    out_dir: Path = Path("."),
) -> list[dict]:
    """
    Code one batch of variables, with validation and review pass.

    Appends turns to `messages` in place. Pass `pdf_prefix` (the document text
    header) for the first call only; subsequent calls in a multi-turn session
    leave it empty so the PDF is not re-sent.
    """
    coding_prompt = build_coding_prompt(variables, codes_by_var)
    user_content = f"{pdf_prefix}\n\n---\n\n{coding_prompt}" if pdf_prefix else coding_prompt
    messages.append({"role": "user", "content": user_content})

    # Initial coding
    text = llm_stream(messages, model, is_anthropic, api_base)
    (out_dir / f"{pdf_stem}.txt").write_text(text, encoding="utf-8")
    codings = parse_codings(text) or []

    # Validate option codes and identify low-confidence items
    codings = validate_option_codes(codings, codes_by_var, variables)
    review_msg, n_invalid, n_low = build_review_message(codings)

    if review_msg is None:
        clean = clean_codings(codings)
        messages.append({"role": "assistant", "content": json.dumps({"codings": clean}, indent=2)})
        return clean

    print(
        f"  → Review pass ({n_invalid} invalid code(s), {n_low} low-confidence)…",
        file=sys.stderr,
    )
    messages.append({"role": "assistant", "content": json.dumps({"codings": clean_codings(codings)}, indent=2)})
    messages.append({"role": "user", "content": review_msg})
    text2 = llm_stream(messages, model, is_anthropic, api_base)
    (out_dir / f"{pdf_stem}.txt").write_text(text2, encoding="utf-8")
    codings = parse_codings(text2) or []

    clean = clean_codings(codings)
    messages.append({"role": "assistant", "content": json.dumps({"codings": clean}, indent=2)})
    return clean


def model_dirname(model: str) -> str:
    """Convert a model string to a safe directory name.

    'ollama/llama3.2'           → 'llama3.2'
    'anthropic/claude-opus-4-6' → 'claude-opus-4-6'
    'claude-opus-4-6'           → 'claude-opus-4-6'
    """
    return model.split("/")[-1]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def resolve_model_config(model: str, api_base: str | None) -> tuple[bool, str | None]:
    """Resolve provider-specific configuration from a model string.

    Returns (is_anthropic, api_base).
    """
    is_anthropic = "claude" in model.lower() and not model.startswith("openai/")
    if api_base is None and model.startswith("lm_studio/"):
        api_base = "http://localhost:1234/v1"
    return is_anthropic, api_base


def code_pdf(
    pdf_path: Path,
    variables: list[dict],
    codes_by_var: dict[str, list[dict]],
    model: str | None = None,
    api_base: str | None = None,
    max_chars: int | None = None,
    by_section: bool = False,
) -> list[dict]:
    """Extract PDF text and code variables, returning a list of coding dicts."""
    if model is None:
        sys.exit("Error: --model is required")
    print(f"Extracting text from {pdf_path.name} ({pdf_path.stat().st_size / 1e6:.1f} MB)…", file=sys.stderr)
    pdf_text = extract_pdf_text(pdf_path, max_chars=max_chars)
    is_anthropic, api_base = resolve_model_config(model, api_base)

    pdf_prefix = f"Source document: {pdf_path.stem}\n\n{pdf_text}"
    messages = [{"role": "system", "content": load_prompt(PROMPT_FILE)}]
    out_dir = Path(model_dirname(model))
    out_dir.mkdir(exist_ok=True)

    if by_section:
        sections: dict[str, list[dict]] = {}
        for v in variables:
            sections.setdefault(v.get("Section", ""), []).append(v)
        codings_by_id: dict[str, dict] = {}
        first = True
        for section_name, section_vars in sections.items():
            print(f"  [{section_name}] — {len(section_vars)} variables…", file=sys.stderr)
            for c in code_section(
                pdf_path.stem, section_vars, codes_by_var, model, api_base, is_anthropic,
                messages, pdf_prefix=pdf_prefix if first else "", out_dir=out_dir,
            ):
                codings_by_id[str(c.get("id", ""))] = c
            first = False
        return list(codings_by_id.values())
    else:
        print(f"Coding {pdf_path.name} ({len(variables)} variables) with {model}…", file=sys.stderr)
        return code_section(
            pdf_path.stem, variables, codes_by_var, model, api_base, is_anthropic,
            messages, pdf_prefix=pdf_prefix, out_dir=out_dir,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Code a PDF for cultural traits using a local or remote LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf", help="PDF file to code")
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LiteLLM model string",
    )
    parser.add_argument(
        "--variables",
        default=str(Path('.') / "variables.csv"),
        help=f"Variables CSV (default: {VARIABLES_CSV})",
    )
    parser.add_argument(
        "--codes",
        default=str(Path('.') / "codes.csv"),
        help=f"Codes CSV (default: {CODES_CSV})",
    )
    parser.add_argument(
        "--section",
        help="Only code variables in this section (substring match)",
    )
    parser.add_argument(
        "--ids",
        help="Comma-separated list of variable IDs to code (e.g. 2,3,5)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override API base URL (e.g. http://localhost:1234/v1 for LM Studio)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Truncate PDF text to this many characters (useful for small context windows)",
    )
    parser.add_argument(
        "--by-section",
        action="store_true",
        help="Code variables section by section (separate LLM call per section, more focused)",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the system prompt and coding prompt to stdout, then exit",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"Error: {pdf_path} not found")

    all_variables = load_variables(Path(args.variables))
    codes_by_var = load_codes(Path(args.codes))

    # Apply filters
    variables = all_variables
    if args.section:
        variables = [v for v in variables if args.section.lower() in v.get("Section", "").lower()]
        if not variables:
            sys.exit(f"No variables found in section matching '{args.section}'")
    if args.ids:
        id_set = {i.strip() for i in args.ids.split(",")}
        variables = [v for v in variables if v["ID"] in id_set]
        if not variables:
            sys.exit(f"No variables found with IDs: {args.ids}")

    option_count = sum(1 for v in variables if v["Datatype"] == "Option")
    print(
        f"Loaded {len(variables)} variables ({option_count} option-type) "
        f"from {args.variables}",
        file=sys.stderr,
    )

    if args.print_prompt:
        coding_prompt = build_coding_prompt(variables, codes_by_var)
        print("=" * 60, "SYSTEM PROMPT", "=" * 60)
        print(load_prompt(PROMPT_FILE))
        print("=" * 60, "CODING PROMPT", "=" * 60)
        print(coding_prompt)
        return

    codings = code_pdf(
        pdf_path,
        variables,
        codes_by_var,
        model=args.model,
        api_base=args.api_base,
        max_chars=args.max_chars,
        by_section=args.by_section,
    )

    json_path = Path(model_dirname(args.model)) / f"{pdf_path.stem}.json"
    json_path.write_text(json.dumps({"codings": codings}, indent=2), encoding="utf-8")
    print(f"JSON saved → {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
