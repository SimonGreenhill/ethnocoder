# Ethnocoder

Automated coding of cultural trait variables from PDF source documents using
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

3. Initialise

```
ethnocoder init <DIR> <DATASET>



```

3. Place PDF source documents in `docs/`. 

NOTE: These need to be named by bibtex citation key i.e. "hv_vanderVeen_B30.pdf" or "s_Peckham_Mairasi_2000.pdf"

4. Put CLDF dataset into `./dataset` (make sure `./dataset/cldf/*-metadata.json` exists).

5. Run the setup command to copy variable/code definitions and extract gold-standard codings:

```bash
ethnocoder setup
```

This will:
- Copy the ParameterTable → `parameters.csv`
- Copy the CodeTable → `codes.csv`
- Create `gold/` and write one gold JSON file per source document

Edit parameters.csv or codes.csv if you want to remove certain variables or codes etc.

6. Initialise the project prompt:

```bash
ethnocoder init   # creates PROMPT.md in the current directory
```

Edit `PROMPT.md` to customise the system prompt for the LLM.


## Usage

### Code a single document

```bash
ethnocoder code docs/example.pdf --model anthropic/claude-opus-4-8
ethnocoder code docs/example.pdf --model ollama/llama3.2
ethnocoder code docs/example.pdf --model lm_studio/gemma-4-e4b --api-base http://localhost:1234/v1
```

Results are saved to `<model_name>/<pdf_stem>.json`.

#### Options

| Flag | Description |
|------|-------------|
| `--model`, `-m` | LiteLLM model string (required) |
| `--ids` | Comma-separated variable IDs to code (e.g. `2,3,5`) |
| `--max-chars` | Truncate PDF text (useful for small context windows) |
| `--api-base` | Override API base URL |
| `--print-prompt` | Print the full prompt and exit without calling the LLM |


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
python -m pytest tests.py -v
```

## Project structure

```
src/ethnocoder/     Installable package source
  cli.py            Entry point — dispatches subcommands
  code.py           PDF extraction, prompt building, LLM calls
  batch.py          Batch runner for all PDFs in docs/
  evaluate.py       Per-variable comparison of coded output vs gold standard
  summarise.py      Aggregate accuracy summary across documents for a model
  setup.py          Copies variables/codes from CLDF, extracts gold codings
  check.py          Document statistics (pages, chars, coded variables)
  init.py           Project scaffolding (creates PROMPT.md)
  data/PROMPT.md    Bundled default system prompt
pyproject.toml      Package metadata and dependencies
parameters.csv      Variable definitions
codes.csv           Valid code values for option-type variables
gold/               Gold-standard codings (one JSON per source document)
docs/               PDF source documents
dataset/            CLDF dataset
```
