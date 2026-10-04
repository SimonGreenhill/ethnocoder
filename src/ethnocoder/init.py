import sys
from importlib.resources import files
from pathlib import Path


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "init",
        description="Initialise a new ethnocoder project in the current directory",
        help="Initialise a new ethnocoder project in the current directory",
    )
    p.set_defaults(func=_run)


def _run(args) -> None:
    dest = Path("PROMPT.md")
    if dest.exists():
        sys.exit(f"Error: {dest} already exists. Remove it first if you want to reset to the default prompt.")
    content = files("ethnocoder.data").joinpath("PROMPT.md").read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest}")
