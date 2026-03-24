# LIKE Project – AI Workflow Instructions

This file enables a seamless AI workflow. An LLM (Claude, GPT-4, Gemini, etc.)
can ingest this document to fully understand how to build, run, test, and extend
the LIKE – Obsolescence in Stock Market QA Systems project.

---

## Project Goal

Compare a **Baseline QA system** (GPT-4 with static historical data) against a
**RAG-based QA system** (GPT-4 with live-retrieved stock data) to quantify how
quickly financial question-answering systems become factually obsolete.

---

## Repository Structure

```
Financial-LIKE-Project/
├── data/                    # Raw CSV files downloaded from Yahoo Finance
│   ├── AAPL.csv
│   ├── MSFT.csv
│   ├── GOOGL.csv
│   └── AMZN.csv
├── processed_data/          # Cleaned CSVs produced by prepocess_data.py (auto-created)
│   └── <TICKER>_processed.csv
├── sample_data/             # Small sample CSVs for testing without an API key
├── baseline_results.jsonl   # Auto-generated log of baseline QA runs
├── eval_baseline_<TICKER>.csv  # Auto-generated batch evaluation results
├── evaluation_comparison.csv   # Auto-generated comparison report
├── data_collection.py       # Downloads historical stock data via yfinance
├── prepocess_data.py        # Cleans and standardises raw CSVs
├── baseline_qa.py           # Baseline QA system + built-in evaluation metrics
├── evaluate.py              # Standalone evaluation / comparison framework
├── config.py                # Centralised configuration (tickers, date range, etc.)
├── requirements.txt         # Python dependencies
├── INSTRUCTIONS.md          # This file
└── README.md
```

---

## Prerequisites

| Requirement | Version  |
|-------------|----------|
| Python      | 3.10+    |
| pip         | any      |
| OpenAI key  | required |

```bash
# 1. Clone
git clone https://github.com/chankal/Financial-LIKE-Project.git
cd Financial-LIKE-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."   # Linux / macOS
# setx OPENAI_API_KEY "sk-..."   # Windows
```

`requirements.txt` should include at minimum:
```
openai>=1.0.0
pandas
yfinance
faiss-cpu
numpy
```

---

## Step-by-Step Workflow

### Step 1 – Collect raw stock data

```bash
python data_collection.py
```

Downloads OHLCV data for AAPL, MSFT, GOOGL, and AMZN from Yahoo Finance
(2019-01-01 to present) and stores each ticker as `data/<TICKER>.csv`.

### Step 2 – Preprocess the data

```bash
python prepocess_data.py
```

Reads every CSV in `data/`, drops rows with missing values, retains
`[Date, Open, High, Low, Close, Volume]`, and writes clean files to
`processed_data/<TICKER>_processed.csv`.

### Step 3 – Run the Baseline QA system (interactive)

```bash
python baseline_qa.py
```

Prompts for a ticker and a natural-language question. Constructs a prompt
containing the last 30 rows of processed data and sends it to GPT-4. Prints
the answer **and** evaluation metrics, then appends a JSON record to
`baseline_results.jsonl`.

### Step 4 – Run a batch evaluation of the baseline

```bash
python baseline_qa.py --batch AAPL
```

Runs five standard evaluation questions for `AAPL` (replace with any ticker)
and saves a summary table to `eval_baseline_AAPL.csv`.

### Step 5 – Evaluate and compare systems

```bash
# Baseline only
python evaluate.py --baseline baseline_results.jsonl

# Baseline vs RAG (once rag_results.jsonl exists)
python evaluate.py --baseline baseline_results.jsonl --rag rag_results.jsonl
```

Outputs a summary table and writes `evaluation_comparison.csv`.

---

## Key Design Decisions (for LLM context)

| Decision | Rationale |
|----------|-----------|
| GPT-4 with `temperature=0` | Deterministic outputs enable reproducible evaluation |
| Last 30 rows as context window | Balances prompt length vs recency |
| JSONL logging | Each run is one line; easy to parse, diff, and analyse |
| `price_error_pct` as primary metric | Directly measures factual obsolescence |
| `hallucination_flag` | Catches prices invented outside the context window |
| `staleness_days` | Measures data freshness relative to today |

---

## Evaluation Metrics Reference

| Metric | Description |
|--------|-------------|
| `price_error_pct` | Absolute % deviation of model's stated price from actual closing price |
| `factual_hit` | 1 if error ≤ 5%, else 0 |
| `hallucination` | 1 if stated price is not within 1% of any value in context window |
| `staleness_days` | Days since the last row of the processed dataset |
| `response_length` | Word count of model answer |

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: processed_data/…` | Run `prepocess_data.py` first |
| `AuthenticationError` from OpenAI | Check `OPENAI_API_KEY` environment variable |
| Empty `data/` folder | Run `data_collection.py` first |
| yfinance returns no data | Ticker may be delisted; check `config.py` ticker list |

---

## Next Steps (RAG System)

The upcoming RAG pipeline (`rag_qa.py`) will:
1. Embed the user's question with `text-embedding-3-large`
2. Query a FAISS vector index built from processed stock documents
3. Retrieve the top-k most relevant chunks
4. Pass retrieved chunks + question to GPT-4 for generation
5. Log results to `rag_results.jsonl` in the same schema as `baseline_results.jsonl`

Once `rag_results.jsonl` exists, run `evaluate.py --baseline … --rag …` to compare.
