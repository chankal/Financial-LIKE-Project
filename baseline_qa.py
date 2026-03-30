import pandas as pd
import sys
import json
from datetime import datetime
import os

# Configuration - choose your backend
USE_OLLAMA = True  # Set to False to use OpenAI instead

if USE_OLLAMA:
    # Using Ollama (free, local)
    import requests
    
    def call_llm(prompt, model="llama3.2"):
        """Call Ollama API"""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            }
        )
        return response.json()["response"]
else:
    # Using OpenAI
    from openai import OpenAI
    client = OpenAI()
    
    def call_llm(prompt, model="gpt-4"):
        """Call OpenAI API"""
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content


def load_stock_context(ticker, rows=30):
    """Load recent stock data as context"""
    try:
        df = pd.read_csv(f"processed_data/{ticker}_processed.csv")
        context = df.tail(rows).to_string()
        
        # Get the most recent date
        df['Date'] = pd.to_datetime(df['Date'])
        most_recent = df['Date'].max().strftime('%Y-%m-%d')
        
        return context, most_recent
    except FileNotFoundError:
        print(f"Error: Could not find processed_data/{ticker}_processed.csv")
        return None, None


def ask_question(ticker=None, question=None):
    """
    Ask a question about a stock ticker.
    Can be called interactively or programmatically.
    """
    # Interactive mode
    if ticker is None:
        ticker = input("Enter stock ticker (AAPL, MSFT, etc): ").upper()
    if question is None:
        question = input("Enter your question: ")
    
    context, recent_date = load_stock_context(ticker)
    
    if context is None:
        return None
    
    prompt = f"""You are a financial assistant analyzing stock market data.

IMPORTANT: The data below is HISTORICAL data ending on {recent_date}. 
Do not claim to know anything beyond this date.

Historical stock data for {ticker}:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the historical data provided above
- Be specific and cite actual numbers from the data when possible
- If the question asks about data beyond {recent_date}, clearly state that your data only goes up to {recent_date}
- Do not make up or hallucinate information
- Keep your answer concise and factual

Answer:"""

    try:
        answer = call_llm(prompt)
        return answer
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


# Benchmark questions for evaluation
BENCHMARK_QUESTIONS = {
    "AAPL": [
        "What was the closing price on the most recent trading day in the dataset?",
        "What was the highest price reached in the last 30 days of data?",
        "What was the average trading volume over the last 10 trading days?",
        "Did the stock price increase or decrease over the last week of data?",
        "What was the price range (high minus low) on the most recent trading day?",
    ],
    "MSFT": [
        "What was the closing price on the most recent trading day in the dataset?",
        "What was the lowest price in the last 30 days of data?",
        "Compare the opening and closing prices on the most recent day - did it go up or down?",
        "What was the trading volume on the most recent trading day?",
        "What is the trend over the last 5 trading days - generally increasing or decreasing?",
    ],
    "GOOGL": [
        "What was the closing price on the most recent trading day in the dataset?",
        "What was the average closing price over the last 7 trading days?",
        "What was the largest single-day price increase in the last 30 days?",
        "What was the trading volume on the most recent trading day?",
        "Is the most recent closing price higher or lower than 7 days ago?",
    ]
}


def run_batch_eval(ticker):
    """Run batch evaluation for a specific ticker"""
    print(f"\n=== Batch Evaluation: {ticker} ===\n")
    
    if ticker not in BENCHMARK_QUESTIONS:
        print(f"No benchmark questions defined for {ticker}")
        return
    
    questions = BENCHMARK_QUESTIONS[ticker]
    results = []
    
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q}")
        result = ask_question(ticker=ticker, question=q)
        
        if result:
            print(f"A{i}: {result}\n")
            results.append({
                "question": q,
                "answer": result,
                "timestamp": datetime.now().isoformat()
            })
        else:
            print(f"A{i}: [Error generating answer]\n")
    
    # Save results
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{ticker}_baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "ticker": ticker,
            "model": "ollama-llama3.2" if USE_OLLAMA else "gpt-4",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    return results


def interactive_mode():
    """Run in interactive mode"""
    print("\n=== Stock Market QA System (Baseline) ===")
    print(f"Backend: {'Ollama (Local)' if USE_OLLAMA else 'OpenAI GPT-4'}")
    print("Type 'quit' to exit\n")
    
    while True:
        ticker = input("\nEnter stock ticker (or 'quit'): ").upper()
        
        if ticker == 'QUIT':
            break
            
        question = input("Enter your question: ")
        
        if question.lower() == 'quit':
            break
        
        print("\nThinking...\n")
        answer = ask_question(ticker=ticker, question=question)
        
        if answer:
            print(f"Answer: {answer}\n")
        else:
            print("Sorry, I couldn't generate an answer.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            # Batch evaluation mode
            if len(sys.argv) < 3:
                print("Usage: python baseline_qa.py --batch TICKER")
                sys.exit(1)
            run_batch_eval(sys.argv[2].upper())
        else:
            print("Unknown argument. Use --batch TICKER for batch evaluation")
    else:
        # Interactive mode
        interactive_mode()
