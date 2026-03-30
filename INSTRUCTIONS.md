# INSTRUCTIONS.md — AI Workflow Guide
> This file is intended to be ingested by an LLM (Claude, GPT, Gemini, etc.) to understand how to build, run, and test this project end-to-end.

---

## Project Overview

**Project Name:** LIKE — Obsolescence in Stock Market QA Systems  
**Goal:** Investigate how quickly financial question-answering systems become outdated as stock market conditions change.  
**Approach:** Compare two QA systems:
1. **Baseline QA** — uses static historical stock data + LLM knowledge only (no live data)
2. **RAG-based QA** *(planned)* — retrieves fresh financial data at query time to reduce knowledge staleness

**Research Question:** How does LLM knowledge obsolescence manifest in financial QA, and can RAG mitigate it?

---

## Repository Structure

The repository is organized with a dedicated folder per system. Currently the baseline system is implemented; a `rag/` folder is planned for the next phase.

```
Financial-LIKE-Project/
├── baseline/
│   ├── data/                  # Raw CSV stock data (downloaded by data_collection.py)
│   ├── sample_data/           # Sample CSVs for testing without running data collection
│   ├── processed_data/        # Output of prepocess_data.py (auto-created at runtime)
│   ├── data_collection.py     # Downloads stock data from Yahoo Finance via yfinance
│   ├── prepocess_data.py      # Cleans and normalizes raw CSVs → processed_data/
│   ├── baseline_qa.py         # Baseline QA system — supports OpenAI (GPT-4) or Ollama
│   ├── config.py              # Configuration: tickers, date ranges, LLM backend choice
│   └── requirements.txt       # Python dependencies for the baseline system
├── rag/                       # (Planned) RAG-based QA system
├── INSTRUCTIONS.md            # This file
└── README.md                  # Project summary
```

> All commands in this guide should be run from inside the `baseline/` directory unless otherwise noted.

---

## Prerequisites

- **Python:** 3.9 or higher
- **Internet access:** Required for `data_collection.py` to pull from Yahoo Finance
- **LLM Backend (choose one):**

| Option | When to use | What's needed |
|---|---|---|
| **OpenAI** | You have an API key and want a hosted model (GPT-4) | `OPENAI_API_KEY` environment variable |
| **Ollama** | You want to run a model locally for free, no API key needed | [Ollama](https://ollama.com) installed and a model pulled (e.g. `llama3`) |

Both backends are supported by `baseline_qa.py`. The active backend is configured in `config.py`.

---

## Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/chankal/Financial-LIKE-Project.git
cd Financial-LIKE-Project/baseline

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### LLM Backend Setup

**Option A — OpenAI (hosted, requires API key):**
```bash
export OPENAI_API_KEY=your_key_here   # macOS/Linux
set OPENAI_API_KEY=your_key_here      # Windows
```
Then set `LLM_BACKEND = "openai"` in `config.py`.

**Option B — Ollama (local, no API key required):**
1. Install Ollama from https://ollama.com
2. Pull a model: `ollama pull llama3`
3. Start the Ollama server: `ollama serve` (runs on `http://localhost:11434` by default)
4. Set `LLM_BACKEND = "ollama"` in `config.py` and set `OLLAMA_MODEL` to your chosen model name

---

## Running the Project (Step-by-Step)

### Step 1 — Collect Stock Data

Downloads historical OHLCV data from Yahoo Finance and saves it to `baseline/data/`.

```bash
# Run from inside the baseline/ directory
python data_collection.py
```

- Output: CSV files in `data/` named by ticker (e.g., `data/AAPL.csv`)
- Tickers and date ranges are configured in `config.py`
- **Skip this step** if using the provided `sample_data/` folder (see below)

---

### Step 2 — Preprocess Data

Cleans and normalizes raw CSVs, keeping only `Date, Open, High, Low, Close, Volume` columns.

```bash
python prepocess_data.py
```

- Reads from: `data/`
- Writes to: `processed_data/` (auto-created if it doesn't exist)
- Output files are named `{TICKER}_processed.csv` (e.g., `processed_data/AAPL_processed.csv`)

> ⚠️ Note: The script filename is `prepocess_data.py` (one 'r' — this is a known typo in the repo).

---

### Step 3 — Run the Baseline QA System

Loads the last 30 rows of processed stock data for a given ticker and answers a natural-language question using your configured LLM backend (OpenAI or Ollama).

```bash
python baseline_qa.py
```

You will be prompted for:
1. A stock ticker (e.g., `AAPL`, `MSFT`, `TSLA`)
2. A natural language question about that stock

**Example interaction:**
```
Enter stock ticker (AAPL, MSFT, etc): AAPL
Enter your question: What was Apple's price trend last month?
```

- The script reads `processed_data/AAPL_processed.csv`
- Passes the last 30 rows as context to the LLM
- Prints the model's answer to stdout

**With OpenAI:** uses the `openai` Python client and calls GPT-4  
**With Ollama:** sends requests to the local Ollama server at `http://localhost:11434`; make sure `ollama serve` is running before executing this script

---

## Using Sample Data (No Internet Required)

If you want to test without running `data_collection.py`, copy the sample files:

```bash
cp sample_data/*.csv data/
python prepocess_data.py
python baseline_qa.py
```

---

## Configuration

Edit `config.py` to change:
- **Tickers** to collect/process
- **Date range** for historical data
- **File paths** for data and processed output directories

---

## Key Files for an LLM to Understand

All paths are relative to the `baseline/` directory.

| File | Purpose |
|---|---|
| `baseline_qa.py` | Core QA logic — builds context from CSV data and calls the configured LLM backend |
| `prepocess_data.py` | Data pipeline — cleans raw CSVs into the format expected by `baseline_qa.py` |
| `data_collection.py` | Data source — fetches OHLCV data from Yahoo Finance via `yfinance` |
| `config.py` | All configurable parameters: tickers, paths, dates, and LLM backend selection |
| `requirements.txt` | All Python package dependencies |

---

## Testing

There are no automated tests at this stage. To manually verify each component works:

```bash
# Run from inside baseline/

# Verify preprocessing runs cleanly
python prepocess_data.py
ls processed_data/   # Should show *_processed.csv files

# Verify baseline QA produces a response
python baseline_qa.py
# Enter: AAPL
# Enter: What was the closing price trend over the last 30 days?
```

Expected: A natural-language answer based on the last 30 rows of OHLCV data, generated by whichever backend is active in `config.py`.

---

## Common Issues

| Problem | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: processed_data/AAPL_processed.csv` | Preprocessing not run yet | Run `python prepocess_data.py` first |
| `AuthenticationError` from OpenAI | Missing or invalid API key | Set `OPENAI_API_KEY` env variable |
| `Connection refused` on `localhost:11434` | Ollama server not running | Run `ollama serve` in a separate terminal |
| `model not found` from Ollama | Model not pulled yet | Run `ollama pull <model_name>` (e.g. `ollama pull llama3`) |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` from inside `baseline/` |
| `KeyError: 'Date'` in preprocessing | Raw CSV format unexpected | Check that `data_collection.py` ran successfully and produced valid CSVs |

---

## Planned Extensions (RAG System)

The next phase of this project will introduce a RAG-based QA system that:
- Retrieves real-time or recent stock data at query time
- Augments the LLM prompt with fresh context
- Allows direct comparison of answer quality vs. the static baseline

When added, it will follow the same pattern: a single runnable Python script that prompts for a ticker and question.

---

## Dependencies (from baseline/requirements.txt)

Key libraries used:
- `yfinance` — Yahoo Finance data download
- `pandas` — Data manipulation and CSV handling
- `openai` — OpenAI API client (used when `LLM_BACKEND = "openai"`)
- `ollama` or `requests` — Local model inference (used when `LLM_BACKEND = "ollama"`)

Install all with: `pip install -r requirements.txt` (from inside the `baseline/` directory)