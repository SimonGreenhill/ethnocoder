# Ethnocoder: pip-installable package refactor

**Date:** 2026-07-06  
**Status:** Approved

## Goal

Refactor the project from a collection of flat scripts into a pip-installable Python package exposing a single `ethnocoder <command> ...` CLI. Install with `pip install -e .` for local use; no PyPI publish planned.

## File Layout

```
ethnocoder/                   ← repo root (unchanged)
├── pyproject.toml            ← replaces requirements.txt
├── src/
│   └── ethnocoder/
│       ├── __init__.py
│       ├── cli.py            ← main entrypoint, argparse subparsers
│       ├── code.py           ← from code_traits.py
│       ├── batch.py          ← from run_batch.py
│       ├── evaluate.py       ← from evaluate.py
│       ├── summarise.py      ← from summarise.py
│       ├── setup.py          ← from setup_dataset.py
│       ├── check.py          ← from check_pdf.py
│       ├── init.py           ← new: project scaffolding
│       └── data/
│           ├── __init__.py
│           └── PROMPT.md     ← bundled default prompt
└── tests.py                  ← unchanged
```

All project-level user data (`docs/`, `gold/`, `parameters.csv`, `codes.csv`, `dataset/`) is entirely user-managed — not created or assumed by the package at install time.

The six flat scripts (`code_traits.py`, `run_batch.py`, `evaluate.py`, `summarise.py`, `setup_dataset.py`, `check_pdf.py`) are deleted once their logic moves into `src/ethnocoder/`.

## CLI Commands

| Command | Old script | Notes |
|---|---|---|
| `ethnocoder code <pdf> --model ...` | `code_traits.py` | all flags preserved |
| `ethnocoder batch <model> --max-mb ...` | `run_batch.py` | calls `code_pdf()` directly, no subprocess |
| `ethnocoder evaluate <doc>` | `evaluate.py` | all flags preserved |
| `ethnocoder summarise <modeldir>` | `summarise.py` | |
| `ethnocoder setup --dataset ...` | `setup_dataset.py` | all flags preserved |
| `ethnocoder check` | `check_pdf.py` | |
| `ethnocoder init` | *(new)* | copies bundled `PROMPT.md` to CWD |

## Key Implementation Changes

1. **`batch` no longer uses subprocess.** Currently `run_batch.py` shells out to `code_traits.py` via `subprocess.run`. In the package, `ethnocoder.batch` imports and calls `code_pdf()` from `ethnocoder.code` directly.

2. **`summarise` import fix.** `summarise.py` does `from evaluate import load_codings_as_dict`. Becomes `from ethnocoder.evaluate import load_codings_as_dict`.

3. **`PROMPT.md` is bundled.** Moves to `src/ethnocoder/data/PROMPT.md` as a package resource. `ethnocoder init` copies it into the CWD. `ethnocoder code` still reads from `./PROMPT.md` (CWD), unchanged.

4. **All other `Path('.')` references are preserved.** `ethnocoder` is a project tool — all data paths (`parameters.csv`, `codes.csv`, `gold/`, `docs/`) correctly reference the working directory.

## `pyproject.toml`

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

## `ethnocoder init`

Copies the bundled `src/ethnocoder/data/PROMPT.md` into the CWD. Exits with an error if `PROMPT.md` already exists (avoids clobbering a customised prompt).

## Testing

`tests.py` is unchanged. Run with `python -m pytest tests.py -v` as before. The import paths inside tests will need updating from `from code_traits import ...` to `from ethnocoder.code import ...` etc.
