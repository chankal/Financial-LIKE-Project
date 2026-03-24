import pandas as pd
from openai import OpenAI
import json
import os
import re
from datetime import datetime

client = OpenAI()

LOG_FILE = "baseline_results.jsonl"

# ── Evaluation helpers ──────────────────────────────────────────────────────

def contains_price(text: str) -> bool:
    """Check whether the response mentions any numeric price/value."""
    return bool(re.search(r"\$[\d,]+|\b\d+\.\d{2}\b", text))

def extract_mentioned_price(text: str) -> float | None:
    """Pull the first dollar-figure or decimal from a response string."""
    m = re.search(r"\$?([\d,]+\.\d+)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None

def compute_price_error(predicted: float | None, actual: float | None) -> float | None:
    """Absolute percentage error between predicted and actual price."""
    if predicted is None or actual is None or actual == 0:
        return None
    return abs(predicted - actual) / actual * 100

def score_response(response_text: str, context_df: pd.DataFrame) -> dict:
    """
    Evaluate a single baseline response against ground-truth processed data.

    Metrics
    -------
    - has_price      : model mentioned at least one numeric price
    - price_error_pct: % deviation from the most-recent closing price
    - staleness_days : gap between dataset's last date and today
    - length_tokens  : rough token count via word split
    """
    actual_close = float(context_df["Close"].iloc[-1])
    last_date    = pd.to_datetime(context_df["Date"].iloc[-1])
    staleness    = (datetime.today() - last_date).days

    predicted = extract_mentioned_price(response_text)
    error     = compute_price_error(predicted, actual_close)

    return {
        "has_price"       : contains_price(response_text),
        "predicted_price" : predicted,
        "actual_close"    : round(actual_close, 4),
        "price_error_pct" : round(error, 2) if error is not None else None,
        "staleness_days"  : staleness,
        "length_tokens"   : len(response_text.split()),
    }

# ── Data loading ────────────────────────────────────────────────────────────

def load_stock_context(ticker: str) -> tuple[str, pd.DataFrame]:
    """Return (context_string, dataframe) for the given ticker."""
    path = f"processed_data/{ticker}_processed.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No processed file found at {path}. "
            "Run prepocess_data.py first."
        )
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    context = df.tail(30).to_string(index=False)
    return context, df

# ── QA pipeline ─────────────────────────────────────────────────────────────

def build_prompt(ticker: str, context: str, question: str) -> str:
    return f"""You are a financial assistant.

Historical stock data for {ticker} (last 30 trading days):
{context}

Question:
{question}

Answer the question using ONLY the historical data provided above.
Do NOT use external knowledge about current prices.
If the data does not contain enough information to answer, say so explicitly.
"""

def ask_question(ticker: str | None = None, question: str | None = None) -> dict:
    """
    Run one QA turn. Accepts optional arguments for scripted / batch use.
    Returns a result dict that is also appended to baseline_results.jsonl.
    """
    if ticker is None:
        ticker = input("Enter stock ticker (AAPL, MSFT, GOOGL, AMZN): ").strip().upper()
    if question is None:
        question = input("Enter your question: ").strip()

    context, df = load_stock_context(ticker)
    prompt = build_prompt(ticker, context, question)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content

    metrics = score_response(answer, df)
    result = {
        "timestamp" : datetime.utcnow().isoformat(),
        "ticker"    : ticker,
        "question"  : question,
        "answer"    : answer,
        "metrics"   : metrics,
    }

    # Persist to JSONL log
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    print("\n── Answer ──────────────────────────────────────────────")
    print(answer)
    print("\n── Evaluation Metrics ──────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print()

    return result

# ── Batch evaluation ─────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    "What was the closing price on the most recent trading day in the dataset?",
    "What was the highest closing price over the last 30 days?",
    "What was the average closing price over the last 30 days?",
    "Was the stock trending up or down over the last two weeks?",
    "What was the trading volume on the most recent day?",
]

def run_batch_eval(ticker: str):
    """Run all EVAL_QUESTIONS for a ticker and print a summary table."""
    print(f"\n=== Batch Evaluation: {ticker} ===\n")
    rows = []
    for q in EVAL_QUESTIONS:
        print(f"Q: {q}")
        result = ask_question(ticker=ticker, question=q)
        rows.append({
            "question"        : q[:60],
            "has_price"       : result["metrics"]["has_price"],
            "price_error_pct" : result["metrics"]["price_error_pct"],
            "staleness_days"  : result["metrics"]["staleness_days"],
        })
    summary = pd.DataFrame(rows)
    print("\n── Batch Summary ────────────────────────────────────────")
    print(summary.to_string(index=False))
    summary.to_csv(f"eval_baseline_{ticker}.csv", index=False)
    print(f"\nSaved to eval_baseline_{ticker}.csv")

# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--batch":
        run_batch_eval(sys.argv[2].upper())
    else:
        ask_question()
