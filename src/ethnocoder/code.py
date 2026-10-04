import argparse
import csv
import json
import sys
from pathlib import Path

import pymupdf
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
import litellm

from ethnocoder.utils import strip_fences, parse_codings

PROMPT_FILE = Path('.') / "PROMPT.md"
PARAMETERS_CSV = Path('.') / "parameters.csv"
CODES_CSV = Path('.') / "codes.csv"


def load_prompt(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_variables(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_codes(path: Path) -> dict[str, list[dict]]:
    codes_by_var: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes_by_var.setdefault(row["Parameter_ID"], []).append(row)
    return codes_by_var


def extract_pdf_text(pdf_path: Path, max_chars: int | None = None) -> str:
    doc = pymupdf.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    text = "\n\n".join(pages)
    if max_chars and len(text) > max_chars:
        original_len = len(text)
        text = text[:max_chars]
        print(f"Warning: PDF text truncated to {max_chars:,} chars (was {original_len:,})", file=sys.stderr)
    return text


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
        datatype = var.get("Datatype") or "Option"
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
        else:
            lines.append("  Provide a brief text value.")

        lines.append("")

    return "\n".join(lines)


def clean_codings(codings: list[dict]) -> list[dict]:
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in codings]


def validate_option_codes(
    codings: list[dict],
    codes_by_var: dict[str, list[dict]],
    variables: list[dict],
) -> list[dict]:
    option_vars = {v["ID"] for v in variables if v.get("Datatype", "Option") == "Option"}
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


def build_review_message(codings: list[dict]) -> tuple[str | None, int, int]:
    invalid = [c for c in codings if c.get("_invalid")]
    low_conf = [c for c in codings if c.get("confidence") in ("low", "absent") and not c.get("_invalid")]

    if not invalid and not low_conf:
        return None, 0, 0

    parts: list[str] = []
    if invalid:
        lines = ["The following variables have invalid codes that must be corrected:"]
        for c in invalid:
            lines.append(
                f"  ID {c['id']}: you coded '{c['code']}' but valid codes are: {', '.join(c['_valid_codes'])}"
            )
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
    kwargs: dict = {"stream": True}
    if not is_anthropic and not model.startswith("lm_studio/"):
        kwargs["response_format"] = {"type": "json_object"}
    if api_base:
        kwargs["api_base"] = api_base

    chunks: list[str] = []
    response = litellm.completion(model=model, messages=messages, **kwargs)
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            chunks.append(delta)
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
    coding_prompt = build_coding_prompt(variables, codes_by_var)
    user_content = f"{pdf_prefix}\n\n---\n\n{coding_prompt}" if pdf_prefix else coding_prompt
    messages.append({"role": "user", "content": user_content})

    text = llm_stream(messages, model, is_anthropic, api_base)
    (out_dir / f"{pdf_stem}.txt").write_text(text, encoding="utf-8")
    codings = parse_codings(text) or []

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
    return model.split("/")[-1]


def resolve_model_config(model: str, api_base: str | None) -> tuple[bool, str | None]:
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


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "code",
        description="Code a PDF for cultural traits using a local or remote LLM",
        help="Code a PDF for cultural traits using a local or remote LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pdf", help="PDF file to code")
    p.add_argument("--model", "-m", default=None, help="LiteLLM model string (required)")
    p.add_argument("--variables", default="parameters.csv", help="Variables CSV (default: parameters.csv)")
    p.add_argument("--codes", default="codes.csv", help="Codes CSV (default: codes.csv)")
    p.add_argument("--section", help="Only code variables in this section (substring match)")
    p.add_argument("--ids", help="Comma-separated list of variable IDs to code (e.g. 2,3,5)")
    p.add_argument("--api-base", default=None, help="Override API base URL")
    p.add_argument("--max-chars", type=int, default=None, help="Truncate PDF text to this many characters")
    p.add_argument("--by-section", action="store_true", help="Code variables section by section")
    p.add_argument("--print-prompt", action="store_true", help="Print the full prompt and exit")
    p.add_argument("--dump", action="store_true", help="Dump messages JSON and exit")
    p.set_defaults(func=_run)


def _run(args) -> None:
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"Error: {pdf_path} not found")

    all_variables = load_variables(Path(args.variables))
    codes_by_var = load_codes(Path(args.codes))

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

    option_count = sum(1 for v in variables if v.get("Datatype", "Option") == "Option")
    print(
        f"Loaded {len(variables)} variables ({option_count} option-type) from {args.variables}",
        file=sys.stderr,
    )

    if args.print_prompt:
        coding_prompt = build_coding_prompt(variables, codes_by_var)
        print("=" * 60, "SYSTEM PROMPT", "=" * 60)
        print(load_prompt(PROMPT_FILE))
        print("=" * 60, "CODING PROMPT", "=" * 60)
        print(coding_prompt)
        return

    if args.dump:
        pdf_text = extract_pdf_text(pdf_path, max_chars=args.max_chars)
        pdf_prefix = f"Source document: {pdf_path.stem}\n\n{pdf_text}"
        messages: list[dict] = [{"role": "system", "content": load_prompt(PROMPT_FILE)}]
        if args.by_section:
            sections: dict[str, list[dict]] = {}
            for v in variables:
                sections.setdefault(v.get("Section", ""), []).append(v)
            first = True
            for section_vars in sections.values():
                coding_prompt = build_coding_prompt(section_vars, codes_by_var)
                user_content = f"{pdf_prefix}\n\n---\n\n{coding_prompt}" if first else coding_prompt
                messages.append({"role": "user", "content": user_content})
                first = False
        else:
            coding_prompt = build_coding_prompt(variables, codes_by_var)
            messages.append({"role": "user", "content": f"{pdf_prefix}\n\n---\n\n{coding_prompt}"})
        print(json.dumps(messages, indent=2))
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
