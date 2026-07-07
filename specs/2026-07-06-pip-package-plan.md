# Ethnocoder pip-installable package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor six flat scripts into a `src/ethnocoder/` package installable via `pip install -e .`, exposing a single `ethnocoder <command>` CLI.

**Architecture:** Each script becomes a module under `src/ethnocoder/`. Each module exposes an `add_subparser(subparsers)` function and a `_run(args)` handler; `cli.py` assembles them under one `argparse` top-level parser. All project-level data paths (`parameters.csv`, `codes.csv`, `gold/`, `docs/`) remain CWD-relative — the package is a project tool, not a library.

**Tech Stack:** Python 3.12, hatchling, argparse, litellm, pycldf, pymupdf, rich, importlib.resources (stdlib)

**Spec:** `specs/2026-07-06-pip-package-design.md`

---

## File Map

| Action | Path | From |
|--------|------|------|
| Create | `pyproject.toml` | replaces `requirements.txt` (already deleted) |
| Create | `src/ethnocoder/__init__.py` | new |
| Create | `src/ethnocoder/cli.py` | new |
| Create | `src/ethnocoder/utils.py` | `utils.py` (already exists) |
| Create | `src/ethnocoder/code.py` | `code_traits.py` |
| Create | `src/ethnocoder/batch.py` | `run_batch.py` |
| Create | `src/ethnocoder/evaluate.py` | `evaluate.py` |
| Create | `src/ethnocoder/summarise.py` | `summarise.py` |
| Create | `src/ethnocoder/setup.py` | `setup_dataset.py` |
| Create | `src/ethnocoder/check.py` | `check_pdf.py` |
| Create | `src/ethnocoder/init.py` | new |
| Create | `src/ethnocoder/data/__init__.py` | new |
| Create | `src/ethnocoder/data/PROMPT.md` | moved from `PROMPT.md` |
| Modify | `tests.py` | update import paths |
| Delete | `code_traits.py`, `run_batch.py`, `evaluate.py`, `summarise.py`, `setup_dataset.py`, `check_pdf.py`, `utils.py` | after tests pass |

---

## Task 1: Create pyproject.toml and package scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/ethnocoder/__init__.py`
- Create: `src/ethnocoder/data/__init__.py`
- Create: `src/ethnocoder/data/PROMPT.md` (copy from `PROMPT.md`)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ethnocoder"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["litellm", "pycldf", "pymupdf", "rich"]

[project.scripts]
ethnocoder = "ethnocoder.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/ethnocoder"]

[project.optional-dependencies]
dev = ["pytest"]
```

- [ ] **Step 2: Create empty `src/ethnocoder/__init__.py`**

```python
```

(Empty file — just needs to exist.)

- [ ] **Step 3: Create empty `src/ethnocoder/data/__init__.py`**

```python
```

(Empty file — makes `ethnocoder.data` a package so `importlib.resources.files()` can locate it.)

- [ ] **Step 4: Copy `PROMPT.md` into the package**

```bash
cp PROMPT.md src/ethnocoder/data/PROMPT.md
```

- [ ] **Step 5: Install the package in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: no errors, `hatchling` builds successfully.

- [ ] **Step 6: Verify the entry point exists (will fail until cli.py is created — that's fine)**

```bash
which ethnocoder
```

Expected: prints a path (e.g. `.venv/bin/ethnocoder`). The command will error until `cli.py` exists — that's expected.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/
git commit -m "feat: scaffold src/ethnocoder package with pyproject.toml"
```

---

## Task 2: Create `src/ethnocoder/utils.py`

Shared utilities used by both `code.py` and `evaluate.py`. Already exists as `utils.py` in the root — just move it into the package.

**Files:**
- Create: `src/ethnocoder/utils.py` (move from `utils.py`)

- [ ] **Step 1: Copy `utils.py` into the package**

```bash
cp utils.py src/ethnocoder/utils.py
```

- [ ] **Step 2: Commit**

```bash
git add src/ethnocoder/utils.py
git commit -m "feat: add ethnocoder.utils (strip_fences, parse_codings)"
```

---

## Task 3: Create `src/ethnocoder/code.py`

Move all logic from `code_traits.py`. Replace `main()` with `add_subparser()` + `_run()`.

**Files:**
- Create: `src/ethnocoder/code.py`

- [ ] **Step 1: Create `src/ethnocoder/code.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/ethnocoder/code.py
git commit -m "feat: add ethnocoder.code (from code_traits.py)"
```

---

## Task 3: Create `src/ethnocoder/evaluate.py`

Move all logic from `evaluate.py`. Replace `main()` with `add_subparser()` + `_run()`.

**Files:**
- Create: `src/ethnocoder/evaluate.py`

- [ ] **Step 1: Create `src/ethnocoder/evaluate.py`**

```python
import argparse
import csv
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ethnocoder.utils import strip_fences, parse_codings

console = Console()

CONF_STYLE = {
    "high":   "green",
    "medium": "yellow",
    "low":    "red",
    "absent": "red",
}

PARAMETERS_CSV = Path("./parameters.csv")
GOLD_DIR = Path("./gold")


def normalize_code(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, OverflowError):
        return s


def load_codings(path: Path) -> list[dict]:
    return parse_codings(strip_fences(path.read_text(encoding="utf-8")))


def load_codings_as_dict(path: Path) -> dict[str, str]:
    return {
        str(r.get("id") or r.get("variable")): normalize_code(r.get("code"))
        for r in load_codings(path)
        if r.get("code") is not None
    }


def eval_codings(gold_path: Path, coded_path: Path, variables_path: Path) -> None:
    gold_raw = load_codings(gold_path)
    coded_list = load_codings(coded_path)

    var_names: dict[str, str] = {}
    with open(variables_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            var_names[row["ID"]] = row["Name"]

    gold: dict[str, str] = {
        str(r["id"]): normalize_code(r["code"])
        for r in gold_raw
        if r.get("code") is not None
    }

    coded: dict[str, str] = {}
    coded_confidence: dict[str, str] = {}
    coded_justification: dict[str, str] = {}
    for r in coded_list:
        vid = str(r.get("id") or r.get("variable"))
        coded[vid] = normalize_code(r.get("code"))
        if r.get("confidence"):
            coded_confidence[vid] = r["confidence"]
        if r.get("justification"):
            coded_justification[vid] = r["justification"]

    matches = mismatches = missing = 0

    table = Table(show_header=True, header_style="dim", box=None, pad_edge=False)
    table.add_column("ID", width=6)
    table.add_column("Variable", width=35)
    table.add_column("Gold", width=10)
    table.add_column("Coded", width=10)
    table.add_column("Conf", width=8)
    table.add_column("Match", width=5)

    for var_id, gold_code in sorted(gold.items(), key=lambda x: int(x[0])):
        name = var_names.get(var_id, "?")[:35]
        confidence = coded_confidence.get(var_id, "")
        justification = coded_justification.get(var_id, "")
        conf_style = CONF_STYLE.get(confidence, "")
        conf_text = Text(confidence, style=conf_style) if confidence else Text("")

        if var_id not in coded:
            coded_code = "(missing)"
            mark = Text("?", style="yellow")
            missing += 1
        else:
            coded_code = coded[var_id]
            if gold_code == coded_code:
                mark = Text("✓", style="green")
                matches += 1
            else:
                mark = Text("✗", style="red")
                mismatches += 1

        table.add_row(var_id, name, gold_code, coded_code, conf_text, mark)
        if justification:
            table.add_row("", Text(justification, style="dim"), "", "", "", "")

    console.print(table)

    total = matches + mismatches + missing
    pct = 100 * matches / (matches + mismatches) if (matches + mismatches) > 0 else 0
    style = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
    console.print()
    console.print(f"Correct:  [{style}]{matches}/{matches + mismatches} ({pct:.1f}%)[/{style}]")
    if missing:
        console.print(f"[yellow]Missing:  {missing} variables present in gold but absent from coded output[/yellow]")
    console.print(f"[dim]Skipped:  {len(gold_raw) - total} variables with null gold code[/dim]")


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "evaluate",
        description="Compare gold and coded JSON files for cultural trait codings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("doc", help="Coded JSON file")
    p.add_argument(
        "--variables",
        default=str(PARAMETERS_CSV),
        help=f"Variables CSV for names (default: {PARAMETERS_CSV})",
    )
    p.set_defaults(func=_run)


def _run(args) -> None:
    coded = Path(args.doc)
    gold = GOLD_DIR / coded.name
    for p in (coded, gold):
        if not p.exists():
            sys.exit(f"Error: {p} not found")
    eval_codings(gold, coded, Path(args.variables))
```

- [ ] **Step 2: Commit**

```bash
git add src/ethnocoder/evaluate.py
git commit -m "feat: add ethnocoder.evaluate (from evaluate.py)"
```

---

## Task 4: Create `src/ethnocoder/setup.py`

Move all logic from `setup_dataset.py`.

**Files:**
- Create: `src/ethnocoder/setup.py`

- [ ] **Step 1: Create `src/ethnocoder/setup.py`**

```python
import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pycldf

DATASET_DIR = Path("./dataset")


def strip_pages(source_key: str) -> str:
    return re.sub(r"\[.*?\]", "", source_key).strip()


def find_metadata(dataset_dir: Path) -> Path:
    matches = list(dataset_dir.rglob("*-metadata.json"))
    if not matches:
        sys.exit(f"Error: no CLDF *-metadata.json found under {dataset_dir}")
    cldf_matches = [m for m in matches if m.parent.name == "cldf"]
    return cldf_matches[0] if cldf_matches else matches[0]


def build_source_index(ds: pycldf.Dataset) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    for row in ds["ValueTable"]:
        for src in row["Source"]:
            key = strip_pages(src)
            if key:
                index[key].append(row)
    return index


def codings_for_source(
    rows: list[dict],
    all_variables: list[dict],
    code_names: dict[str, str],
) -> list[dict]:
    by_param: dict[str, str | None] = {}
    for row in rows:
        param_id = row["Parameter_ID"]
        if row["Code_ID"]:
            by_param[param_id] = code_names.get(row["Code_ID"], row["Code_ID"])
        elif row["Value"] not in ("", None):
            by_param[param_id] = str(row["Value"])
        else:
            by_param[param_id] = None

    return [
        {"id": var["ID"], "code": by_param.get(var["ID"])}
        for var in all_variables
    ]


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "setup",
        description="Set up working files from a CLDF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", type=Path, default=DATASET_DIR,
        help=f"Path to dataset root (default: {DATASET_DIR})")
    p.add_argument(
        "-o", "--output", default="gold",
        help="Output directory for gold files (default: gold/)")
    p.add_argument(
        "--list", action="store_true",
        help="List all source keys and exit")
    p.set_defaults(func=_run)


def _run(args) -> None:
    metadata_path = find_metadata(args.dataset)
    ds = pycldf.Dataset.from_metadata(metadata_path)

    if not args.list:
        for table, dst_name in (
            (ds["ParameterTable"], "parameters.csv"),
            (ds["CodeTable"], "codes.csv"),
        ):
            src = ds.directory / table.url.string
            dst = Path(dst_name)
            shutil.copy2(src, dst)
            print(f"Copied {src} → {dst}")

    all_variables = list(ds["ParameterTable"])
    code_names = {row["ID"]: row["Name"] for row in ds["CodeTable"]}
    source_index = build_source_index(ds)

    if args.list:
        for key in sorted(source_index):
            rows = source_index[key]
            n_params = len({r["Parameter_ID"] for r in rows})
            n_societies = len({r["Language_ID"] for r in rows})
            print(f"{key:40s}  {n_params:3d} params  {n_societies:3d} societies")
        print(f"\nTotal: {len(source_index)} sources", file=sys.stderr)
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in sorted(source_index):
        rows = source_index[key]
        codings = codings_for_source(rows, all_variables, code_names)
        out_path = output_dir / f"{key}.json"
        out_path.write_text(json.dumps(codings, indent=2), encoding="utf-8")
        n_coded = sum(1 for c in codings if c["code"] is not None)
        print(f"{key:40s}  {n_coded:3d}/{len(all_variables)} coded → {out_path}")

    print(f"\nDone. {len(source_index)} gold file(s) written to {output_dir}/",
          file=sys.stderr)
```

- [ ] **Step 2: Commit**

```bash
git add src/ethnocoder/setup.py
git commit -m "feat: add ethnocoder.setup (from setup_dataset.py)"
```

---

## Task 5: Update `tests.py` and verify tests pass

The three import lines at the top of `tests.py` point at the old script names. Update them to use the new package paths.

**Files:**
- Modify: `tests.py`

- [ ] **Step 1: Confirm existing tests fail with new import paths (expected)**

Run the tests as-is to see the baseline:

```bash
python -m pytest tests.py -v 2>&1 | head -20
```

Expected: tests **pass** (they still import from the old flat scripts which still exist).

- [ ] **Step 2: Update imports in `tests.py`**

Replace lines 20–29 in `tests.py`:

Old:
```python
from code_traits import (
    build_coding_prompt,
    build_review_message,
    model_dirname,
    parse_codings,
    strip_fences,
    validate_option_codes,
)
from evaluate import load_codings, load_codings_as_dict, normalize_code
from setup_dataset import strip_pages
```

New:
```python
from ethnocoder.code import (
    build_coding_prompt,
    build_review_message,
    model_dirname,
    parse_codings,
    strip_fences,
    validate_option_codes,
)
from ethnocoder.evaluate import load_codings, load_codings_as_dict, normalize_code
from ethnocoder.setup import strip_pages
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests.py -v
```

Expected: all tests pass (same count as before).

- [ ] **Step 4: Commit**

```bash
git add tests.py
git commit -m "fix: update tests.py imports to use ethnocoder package"
```

---

## Task 6: Create `src/ethnocoder/summarise.py`

Move logic from `summarise.py`. Fix the cross-module import.

**Files:**
- Create: `src/ethnocoder/summarise.py`

- [ ] **Step 1: Create `src/ethnocoder/summarise.py`**

```python
import argparse
from pathlib import Path

from ethnocoder.evaluate import load_codings_as_dict

GOLD_DIR = Path("gold")


def compare(gold: dict[str, str], coded: dict[str, str]) -> tuple[int, int]:
    same, diff = 0, 0
    for vid, gold_code in gold.items():
        if vid not in coded or gold_code != coded[vid]:
            diff += 1
        else:
            same += 1
    return (same, same + diff)


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "summarise",
        description="Summarise accuracy across all documents for a model",
    )
    p.add_argument("modeldir", type=Path, help="Directory of coded JSON outputs")
    p.set_defaults(func=_run)


def _run(args) -> None:
    overall = {"same": 0, "total": 0}
    for p in sorted(args.modeldir.glob("*.json")):
        coded = load_codings_as_dict(p)
        gold = load_codings_as_dict(GOLD_DIR / p.name)
        same, total = compare(gold, coded)
        m = same / total if total else 0
        print(f"{p.stem:20s}\t{same:5d}\t{total:5d}\t{m:0.4f}")
        overall["same"] += same
        overall["total"] += total

    if overall["total"]:
        pc = (overall["same"] / overall["total"]) * 100
        print(f"\n{overall['same']} / {overall['total']} = {pc:0.2f}%")
    else:
        print("\nNo comparable variables found.")
```

- [ ] **Step 2: Commit**

```bash
git add src/ethnocoder/summarise.py
git commit -m "feat: add ethnocoder.summarise (from summarise.py)"
```

---

## Task 7: Create `src/ethnocoder/batch.py`

Move logic from `run_batch.py`. Replace the `subprocess.run` call with a direct call to `code_pdf()`.

**Files:**
- Create: `src/ethnocoder/batch.py`

- [ ] **Step 1: Create `src/ethnocoder/batch.py`**

```python
import argparse
import json
import sys
from pathlib import Path

from ethnocoder.code import code_pdf, load_codes, load_variables, model_dirname

DOCS_DIR = Path("docs")
PARAMETERS_CSV = Path("parameters.csv")
CODES_CSV = Path("codes.csv")


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "batch",
        description="Run ethnocoder code on all PDFs in docs/ under a size limit",
    )
    p.add_argument("model", help="LiteLLM model string")
    p.add_argument("--max-mb", type=float, default=1.0, help="Max file size in MB (default: 1.0)")
    p.add_argument("--dry-run", action="store_true", help="Print what would run without running it")
    p.add_argument("--force", action="store_true", help="Re-run even if output already exists")
    p.set_defaults(func=_run)


def _run(args) -> None:
    max_bytes = args.max_mb * 1_000_000
    out_dir = Path(model_dirname(args.model))

    pdfs = sorted(DOCS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_size)
    eligible = [p for p in pdfs if p.stat().st_size <= max_bytes]
    skipped_size = len(pdfs) - len(eligible)

    already_done = {p.stem for p in out_dir.glob("*.json")} if out_dir.exists() else set()
    todo = [p for p in eligible if args.force or p.stem not in already_done]
    skipped_done = len(eligible) - len(todo)

    print(f"PDFs in {DOCS_DIR}/: {len(pdfs)} total, {len(eligible)} under {args.max_mb}MB")
    print(f"Already coded: {skipped_done}  |  To run: {len(todo)}  |  Over size limit: {skipped_size}")
    print(f"Output dir: {out_dir}/")
    print()

    if not todo:
        print("Nothing to do.")
        return

    variables = load_variables(PARAMETERS_CSV)
    codes_by_var = load_codes(CODES_CSV)

    ok = 0
    failed = []
    for i, pdf in enumerate(todo, 1):
        size_mb = pdf.stat().st_size / 1_000_000
        print(f"[{i}/{len(todo)}] {pdf.name} ({size_mb:.2f}MB)", flush=True)
        if args.dry_run:
            print(f"  → would run: ethnocoder code --model {args.model} {pdf}")
            continue
        try:
            codings = code_pdf(pdf, variables, codes_by_var, model=args.model)
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{pdf.stem}.json"
            out_path.write_text(json.dumps({"codings": codings}, indent=2), encoding="utf-8")
            print(f"  JSON saved → {out_path}", file=sys.stderr)
            ok += 1
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed.append(pdf.name)

    if not args.dry_run:
        print()
        print(f"Done: {ok}/{len(todo)} succeeded")
        if failed:
            print(f"Failed: {', '.join(failed)}")
            raise SystemExit(1)
```

- [ ] **Step 2: Commit**

```bash
git add src/ethnocoder/batch.py
git commit -m "feat: add ethnocoder.batch (from run_batch.py, no subprocess)"
```

---

## Task 8: Create `src/ethnocoder/check.py`

Move all logic from `check_pdf.py`.

**Files:**
- Create: `src/ethnocoder/check.py`

- [ ] **Step 1: Create `src/ethnocoder/check.py`**

```python
import argparse
import json
from pathlib import Path

import pymupdf
from rich.console import Console
from rich.table import Table

console = Console()


def pdf_stats(pdf_path: Path) -> dict:
    doc = pymupdf.open(str(pdf_path))
    pages = len(doc)
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return {"pages": pages, "chars": len(text)}


def gold_stats(gold_path: Path) -> dict:
    o = json.loads(gold_path.read_text())
    if "codings" in o:
        o = o["codings"]
    coded = sum(1 for x in o if x.get("code") is not None)
    return {"coded": coded}


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "check",
        description="Print per-document statistics from the docs/ directory",
    )
    p.add_argument("--docs-dir", type=Path, default=Path("docs"))
    p.add_argument("--gold-dir", type=Path, default=Path("gold"))
    p.set_defaults(func=_run)


def _run(args) -> None:
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Source", min_width=30)
    table.add_column("Pages", justify="right")
    table.add_column("Chars", justify="right")
    table.add_column("Coded", justify="right")

    rows = []
    for gold_file in sorted(args.gold_dir.glob("*.json")):
        stem = gold_file.stem
        pdf_path = args.docs_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            continue
        ps = pdf_stats(pdf_path)
        gs = gold_stats(gold_file)
        rows.append({"stem": stem, **ps, **gs})
        table.add_row(stem, str(ps["pages"]), f"{ps['chars']:,d}", str(gs["coded"]))

    if rows:
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        mn = lambda k: min(r[k] for r in rows)
        mx = lambda k: max(r[k] for r in rows)
        table.add_section()
        table.add_row("[dim]mean[/dim]", f"[dim]{avg('pages'):.0f}[/dim]", f"[dim]{avg('chars'):,.0f}[/dim]", f"[dim]{avg('coded'):.1f}[/dim]")
        table.add_row("[dim]min[/dim]", f"[dim]{mn('pages')}[/dim]", f"[dim]{mn('chars'):,d}[/dim]", f"[dim]{mn('coded')}[/dim]")
        table.add_row("[dim]max[/dim]", f"[dim]{mx('pages')}[/dim]", f"[dim]{mx('chars'):,d}[/dim]", f"[dim]{mx('coded')}[/dim]")

    console.print(table)
    if rows:
        console.print(f"\n{n} documents")
```

- [ ] **Step 2: Commit**

```bash
git add src/ethnocoder/check.py
git commit -m "feat: add ethnocoder.check (from check_pdf.py)"
```

---

## Task 9: Create `src/ethnocoder/init.py` with test

New module: copies the bundled `PROMPT.md` into the CWD. Uses `importlib.resources` (Python 3.12 stdlib).

**Files:**
- Create: `src/ethnocoder/init.py`
- Modify: `tests.py`

- [ ] **Step 1: Write the failing tests for `init`**

Add to the end of `tests.py` (before `if __name__ == "__main__":`):

```python
# ---------------------------------------------------------------------------
# init: scaffold project
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_prompt_md(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import argparse
        from ethnocoder.init import _run
        _run(argparse.Namespace())
        assert (tmp_path / "PROMPT.md").exists()
        assert len((tmp_path / "PROMPT.md").read_text()) > 0

    def test_refuses_to_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "PROMPT.md").write_text("custom prompt")
        import argparse
        from ethnocoder.init import _run
        with pytest.raises(SystemExit):
            _run(argparse.Namespace())
        assert (tmp_path / "PROMPT.md").read_text() == "custom prompt"
```

- [ ] **Step 2: Run tests to confirm the new tests fail**

```bash
python -m pytest tests.py::TestInit -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ethnocoder.init'`

- [ ] **Step 3: Create `src/ethnocoder/init.py`**

```python
import argparse
import sys
from importlib.resources import files
from pathlib import Path


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "init",
        description="Initialise a new ethnocoder project in the current directory",
    )
    p.set_defaults(func=_run)


def _run(args) -> None:
    dest = Path("PROMPT.md")
    if dest.exists():
        sys.exit(f"Error: {dest} already exists. Remove it first if you want to reset to the default prompt.")
    content = files("ethnocoder.data").joinpath("PROMPT.md").read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest}")
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests.py::TestInit -v
```

Expected: both tests pass.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ethnocoder/init.py tests.py
git commit -m "feat: add ethnocoder.init command with tests"
```

---

## Task 10: Create `src/ethnocoder/cli.py`

Wire all subcommands under a single `ethnocoder` entry point.

**Files:**
- Create: `src/ethnocoder/cli.py`

- [ ] **Step 1: Create `src/ethnocoder/cli.py`**

```python
import argparse

from ethnocoder import batch, check, code, evaluate, init, setup, summarise


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ethnocoder",
        description="Automated coding of cultural trait variables from PDF source documents",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    code.add_subparser(subparsers)
    batch.add_subparser(subparsers)
    evaluate.add_subparser(subparsers)
    summarise.add_subparser(subparsers)
    setup.add_subparser(subparsers)
    check.add_subparser(subparsers)
    init.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)
```

- [ ] **Step 2: Verify top-level help works**

```bash
ethnocoder --help
```

Expected output (approximately):
```
usage: ethnocoder [-h] command ...

Automated coding of cultural trait variables from PDF source documents

positional arguments:
  command
    code        Code a PDF for cultural traits using a local or remote LLM
    batch       Run ethnocoder code on all PDFs in docs/ under a size limit
    evaluate    Compare gold and coded JSON files for cultural trait codings
    summarise   Summarise accuracy across all documents for a model
    setup       Set up working files from a CLDF dataset
    check       Print per-document statistics from the docs/ directory
    init        Initialise a new ethnocoder project in the current directory
```

- [ ] **Step 3: Verify each subcommand's help**

```bash
ethnocoder code --help
ethnocoder batch --help
ethnocoder evaluate --help
ethnocoder summarise --help
ethnocoder setup --help
ethnocoder check --help
ethnocoder init --help
```

Expected: each prints its own help without errors.

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ethnocoder/cli.py
git commit -m "feat: add ethnocoder.cli entrypoint, wire all subcommands"
```

---

## Task 11: Delete old flat scripts

Only delete after all tests pass and the CLI is verified.

**Files:**
- Delete: `code_traits.py`, `run_batch.py`, `evaluate.py`, `summarise.py`, `setup_dataset.py`, `check_pdf.py`, `utils.py`

- [ ] **Step 1: Run full test suite one last time before deleting**

```bash
python -m pytest tests.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Delete old scripts**

```bash
git rm code_traits.py run_batch.py evaluate.py summarise.py setup_dataset.py check_pdf.py utils.py
```

- [ ] **Step 3: Run tests again to confirm nothing broke**

```bash
python -m pytest tests.py -v
```

Expected: all tests still pass (tests now import only from `ethnocoder.*`).

- [ ] **Step 4: Verify CLI still works**

```bash
ethnocoder --help
```

Expected: same output as before.

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: remove old flat scripts, superseded by src/ethnocoder package"
```

---

## Task 12: Update README

The README still references the old `python code_traits.py` invocations.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README.md**

Replace the Setup section's step 1:

Old:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

New:
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
# or include dev dependencies:
pip install -e ".[dev]"
```

Replace all `python code_traits.py` invocations with `ethnocoder code`:
- `python code_traits.py docs/example.pdf --model anthropic/claude-opus-4-8` → `ethnocoder code docs/example.pdf --model anthropic/claude-opus-4-8`
- `python code_traits.py docs/example.pdf --model ollama/llama3.2` → `ethnocoder code docs/example.pdf --model ollama/llama3.2`
- etc.

Replace `python run_batch.py` → `ethnocoder batch`:
- `python run_batch.py anthropic/claude-opus-4-8 --max-mb 2` → `ethnocoder batch anthropic/claude-opus-4-8 --max-mb 2`
- `python run_batch.py ollama/llama3.2 --max-mb 5 --dry-run` → `ethnocoder batch ollama/llama3.2 --max-mb 5 --dry-run`

Replace `python evaluate.py` → `ethnocoder evaluate`:
- `python evaluate.py claude-opus-4-8/example.json` → `ethnocoder evaluate claude-opus-4-8/example.json`

Replace `python summarise.py` → `ethnocoder summarise`:
- `python summarise.py claude-opus-4-8/` → `ethnocoder summarise claude-opus-4-8/`

Replace `python check_pdf.py` → `ethnocoder check`.

Replace `python setup_dataset.py` → `ethnocoder setup`.

Add `ethnocoder init` to the setup section (step 3 or 4):
```
ethnocoder init   # creates PROMPT.md in the current directory
```

Update the Project structure table to reflect the new layout:
```
src/ethnocoder/     Installable package source
  cli.py            Entry point — dispatches subcommands
  code.py           PDF extraction, prompt building, LLM calls
  batch.py          Batch runner for all PDFs in docs/
  evaluate.py       Per-variable comparison vs gold standard
  summarise.py      Aggregate accuracy summary across documents
  setup.py          Copies variables/codes from CLDF, extracts gold
  check.py          Document statistics (pages, chars, coded variables)
  init.py           Project scaffolding (creates PROMPT.md)
  data/PROMPT.md    Bundled default system prompt
```

Remove the `requirements.txt` reference; mention `pyproject.toml` instead.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for pip-installable package"
```
