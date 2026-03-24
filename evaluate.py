import argparse
import json
import re
import os
from datetime import datetime
import pandas as pd
 
 
# ── Metric helpers ──────────────────────────────────────────────────────────
 
def extract_first_price(text: str) -> float | None:
    m = re.search(r"\$?([\d,]+\.\d{1,4})", text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None
 
 
def price_error_pct(predicted: float | None, actual: float) -> float | None:
    if predicted is None or actual == 0:
        return None
    return abs(predicted - actual) / actual * 100
 
 
def factual_hit(predicted: float | None, actual: float, threshold: float = 5.0) -> int:
    err = price_error_pct(predicted, actual)
    if err is None:
        return 0
    return 1 if err <= threshold else 0
 
 
def hallucination_flag(answer: str, context_prices: list[float],
                       tolerance: float = 1.0) -> int:
    """
    Returns 1 if the answer contains a price that is NOT within `tolerance`
    of any price appearing in the raw context window, else 0.
    """
    predicted = extract_first_price(answer)
    if predicted is None:
        return 0
    for p in context_prices:
        if abs(predicted - p) / max(p, 1e-6) * 100 <= tolerance:
            return 0
    return 1
 
 
def load_context_prices(ticker: str, n: int = 30) -> list[float]:
    path = f"processed_data/{ticker}_processed.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path).tail(n)
    return df["Close"].tolist()
 
 
def staleness_days(ticker: str) -> int:
    path = f"processed_data/{ticker}_processed.csv"
    if not os.path.exists(path):
        return -1
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    last_date = df["Date"].max()
    return (datetime.today() - last_date).days
 
 
# ── Evaluate a single JSONL log file ────────────────────────────────────────
 
def evaluate_log(log_path: str, label: str) -> pd.DataFrame:
    records = []
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            ticker  = rec.get("ticker", "AAPL")
            answer  = rec.get("answer", "")
            actual  = rec.get("metrics", {}).get("actual_close")
 
            ctx_prices = load_context_prices(ticker)
            pred       = extract_first_price(answer)
 
            records.append({
                "system"          : label,
                "ticker"          : ticker,
                "question"        : rec.get("question", "")[:60],
                "predicted_price" : pred,
                "actual_close"    : actual,
                "price_error_pct" : round(price_error_pct(pred, actual or 0) or -1, 2),
                "factual_hit"     : factual_hit(pred, actual or 0),
                "hallucination"   : hallucination_flag(answer, ctx_prices),
                "staleness_days"  : staleness_days(ticker),
                "response_length" : len(answer.split()),
            })
 
    return pd.DataFrame(records)
 
 
# ── Comparison report ────────────────────────────────────────────────────────
 
def comparison_report(baseline_df: pd.DataFrame,
                       rag_df: pd.DataFrame | None = None) -> None:
    all_df = pd.concat([baseline_df] + ([rag_df] if rag_df is not None else []))
 
    summary_cols = ["price_error_pct", "factual_hit", "hallucination",
                    "staleness_days", "response_length"]
 
    print("\n══════════════════════════════════════════════")
    print("         EVALUATION SUMMARY REPORT           ")
    print("══════════════════════════════════════════════")
    print(all_df.groupby("system")[summary_cols].mean().round(2).to_string())
 
    if rag_df is not None:
        print("\n── Improvement (Baseline → RAG) ─────────────")
        b_mean = baseline_df[summary_cols].mean()
        r_mean = rag_df[summary_cols].mean()
        delta  = (r_mean - b_mean).rename("delta")
        print(delta.round(2).to_string())
 
    out = "evaluation_comparison.csv"
    all_df.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")
 
 
# ── CLI entry point ──────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description="Evaluate QA system logs.")
    parser.add_argument("--baseline", required=True,
                        help="Path to baseline JSONL log file")
    parser.add_argument("--rag", default=None,
                        help="Path to RAG JSONL log file (optional)")
    args = parser.parse_args()
 
    baseline_df = evaluate_log(args.baseline, label="Baseline")
    rag_df = evaluate_log(args.rag, label="RAG") if args.rag else None
 
    comparison_report(baseline_df, rag_df)
 
 
if __name__ == "__main__":
    main()
