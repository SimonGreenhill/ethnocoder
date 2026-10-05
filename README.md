# Ethnocoder

Automated coding of trait and feature variables from PDF source documents using
LLMs. Given a CLDF dataset, a PDF and a set of variable definitions, the system
prompts an LLM to assign standardised codes for each variable, then evaluates
accuracy against gold-standard codings.

## Requirements

- Python 3.12+
- An LLM provider: Anthropic API key, OpenAI API key, local [Ollama](https://ollama.com/) instance, or [LM Studio](https://lmstudio.ai/)

## Setup

1. Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
pip install -e .
# or include dev dependencies:
pip install -e ".[dev]"
```

2. Set API keys as needed (for Claude or ChatGPT):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

3. Initialise a working directory from a CLDF dataset (make sure `<dataset>/cldf/*-metadata.json` exists):

```bash
ethnocoder init /path/to/dataset .
```

This will, in the working directory:
- Copy the ParameterTable → `parameters.csv`
- Copy the CodeTable → `codes.csv`
- Create `gold/` and write one gold JSON file per source document
- Write `PROMPT.md` (the default system prompt) if it does not already exist
- Create an empty `docs/` for PDF source documents

Edit `parameters.csv` or `codes.csv` to remove variables or codes, and `PROMPT.md` to customise the LLM system prompt.

4. Place PDF source documents in `docs/`.

NOTE: These need to be named by bibtex citation key i.e. "hv_vanderVeen_B30.pdf" or "s_Peckham_Mairasi_2000.pdf"

### Inspecting or re-extracting a dataset

`init` is a convenience wrapper over `dataset`. To inspect or extract individual pieces:

```bash
ethnocoder dataset /path/to/dataset                          # summary: parameter/code/source counts
ethnocoder dataset /path/to/dataset --list                   # list source keys and stats
ethnocoder dataset /path/to/dataset --extract gold -o .      # (re)write gold/ only
ethnocoder dataset /path/to/dataset --extract parameters codes -o .
```


## Usage

### Code a single document

```bash
ethnocoder code docs/example.pdf --model anthropic/claude-opus-4-8
ethnocoder code docs/example.pdf --model ollama/llama3.2
ethnocoder code docs/example.pdf --model lm_studio/gemma-4-e4b --api-base http://localhost:1234/v1
```

Results are saved to `<model_name>/<pdf_stem>.json`.

The `--model` string must include a LiteLLM **provider prefix** (`anthropic/`,
`openai/`, `ollama/`, `lm_studio/`, …). A bare name like `gemma-4-12b-qat` fails
with `LLM Provider NOT provided`, because `--api-base` sets only the URL, not the
request format. For LM Studio use the `lm_studio/` prefix — it defaults
`--api-base` to `http://localhost:1234/v1`, so you can omit the flag unless your
port differs:

```bash
ethnocoder code docs/example.pdf --model lm_studio/gemma-4-12b-qat
```

#### Options

| Flag | Description |
|------|-------------|
| `--model`, `-m` | LiteLLM model string (required) |
| `--variables` | Variables CSV (default: `parameters.csv`) |
| `--codes` | Codes CSV (default: `codes.csv`) |
| `--ids` | Comma-separated variable IDs to code (e.g. `2,3,5`) |
| `--max-chars` | Truncate PDF text (useful for small context windows) |
| `--api-base` | Override API base URL |
| `--context-budget` | Token budget per request — codes variables in batches that fit it (see below) |
| `--response-reserve-per-var` | Output tokens reserved per variable when packing batches (default: 250) |
| `--print-prompt` | Print the full prompt and exit without calling the LLM |

#### Batching for small context windows

Sending every variable in one request can exceed a local model's context window
(e.g. 195 Grambank variables ≈ 240k tokens, mostly from long variable descriptions).
Set `--context-budget` to the model's context size to code variables in batches that
fit:

```bash
ethnocoder code docs/example.pdf --model lm_studio/gemma-4-12b-qat --context-budget 100000
```

Each batch is a fresh request with an identical `[system prompt, PDF]` prefix, so a
local server (LM Studio / llama.cpp) reuses its cached KV for that prefix and the PDF
is processed once rather than per batch. Results are merged and written incrementally
after each batch, so an interrupted run keeps completed batches. Without
`--context-budget`, all variables go in a single request (fine for large-context API
models). If the PDF alone is too big to leave room for even one variable, the run
stops with a message to raise the budget or use `--max-chars`.


### Batch coding

Run all PDFs under a size limit:

```bash
ethnocoder batch anthropic/claude-opus-4-8 --max-mb 2
ethnocoder batch ollama/llama3.2 --max-mb 5 --dry-run
```

Already-coded documents are skipped unless `--force` is passed.

### Evaluate against gold standard

Compare a single model output against the gold codings:

```bash
ethnocoder evaluate claude-opus-4-8/example.json
```

Summarise accuracy across all documents for a model:

```bash
ethnocoder summarise claude-opus-4-8/
```

### Inspect document statistics

```bash
ethnocoder check
```

Prints page count, character count, and number of gold-coded variables for each PDF.

## Tests

```bash
python -m pytest
```

## Project structure

```
src/ethnocoder/     Installable package source
  cli.py            Entry point — dispatches subcommands
  code.py           PDF extraction, prompt building, LLM calls
  batch.py          Batch runner for all PDFs in docs/
  evaluate.py       Per-variable comparison of coded output vs gold standard
  summarise.py      Aggregate accuracy summary across documents for a model
  cldf.py           CLDF dataset reading / extraction helpers
  dataset.py        Inspect/extract a CLDF dataset (--list, --extract)
  check.py          Document statistics (pages, chars, coded variables)
  init.py           Bootstrap a working directory from a CLDF dataset
  data/PROMPT.md    Bundled default system prompt
pyproject.toml      Package metadata and dependencies
parameters.csv      Variable definitions
codes.csv           Valid code values for option-type variables
gold/               Gold-standard codings (one JSON per source document)
docs/               PDF source documents
```
