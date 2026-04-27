"""
RAG-based Question Answering System for Stock Market Data
Uses FAISS vector database and real-time data retrieval to reduce obsolescence
"""

import pandas as pd
import sys
import json
from datetime import datetime, timedelta
import os
import numpy as np
import requests

# Configuration - choose your backend
USE_OLLAMA = True  # Set to False to use OpenAI instead

if USE_OLLAMA:
    # Using Ollama (free, local)
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
    
    def get_embedding(text, model="nomic-embed-text"):
        """Get text embedding from Ollama"""
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": model,
                "prompt": text
            }
        )
        return np.array(response.json()["embedding"])
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
    
    def get_embedding(text, model="text-embedding-3-small"):
        """Get text embedding from OpenAI"""
        response = client.embeddings.create(
            model=model,
            input=text
        )
        # OpenAI returns embeddings in response.data[0].embedding
        return np.array(response.data[0].embedding)

def fetch_realtime_data(ticker, days=30):
    """
    Fetch recent stock data from Yahoo Finance API
    This simulates real-time data retrieval at query time
    """
    try:
        import yfinance as yf
        
        # Get data for the last N days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"Warning: No data retrieved for {ticker}")
            return None
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Keep only relevant columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
        
    except Exception as e:
        print(f"Error fetching real-time data for {ticker}: {e}")
        return None


def create_document_chunks(df, ticker):
    """
    Convert stock data into text chunks for embedding
    Each chunk represents a day or week of trading data
    """
    chunks = []
    
    # Create daily chunks
    for idx, row in df.iterrows():
        date = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
        
        chunk = f"""Stock: {ticker}
Date: {date}
Open: ${row['Open']:.2f}
High: ${row['High']:.2f}
Low: ${row['Low']:.2f}
Close: ${row['Close']:.2f}
Volume: {int(row['Volume']):,}
Price Range: ${(row['High'] - row['Low']):.2f}
"""
        chunks.append({
            'text': chunk,
            'date': date,
            'ticker': ticker,
            'close': row['Close'],
            'volume': row['Volume']
        })
    
    # Create weekly summary chunks
    if len(df) >= 7:
        for i in range(0, len(df) - 6, 7):
            week_data = df.iloc[i:i+7]
            start_date = week_data.iloc[0]['Date']
            end_date = week_data.iloc[-1]['Date']
            
            start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            
            weekly_chunk = f"""Stock: {ticker}
Week: {start_date_str} to {end_date_str}
Opening Price: ${week_data.iloc[0]['Open']:.2f}
Closing Price: ${week_data.iloc[-1]['Close']:.2f}
Week High: ${week_data['High'].max():.2f}
Week Low: ${week_data['Low'].min():.2f}
Average Volume: {int(week_data['Volume'].mean()):,}
Weekly Change: {((week_data.iloc[-1]['Close'] - week_data.iloc[0]['Open']) / week_data.iloc[0]['Open'] * 100):.2f}%
"""
            chunks.append({
                'text': weekly_chunk,
                'date': end_date_str,
                'ticker': ticker,
                'type': 'weekly_summary',
                'close': week_data.iloc[-1]['Close'],
                'volume': week_data['Volume'].mean()
            })
    
    return chunks


def build_vector_database(chunks):
    """
    Build FAISS vector database from document chunks
    Returns: (index, chunks) tuple
    """
    try:
        import faiss
    except ImportError:
        print("ERROR: FAISS not installed. Run: pip install faiss-cpu")
        sys.exit(1)
    
    print(f"Building vector database with {len(chunks)} chunks...")
    
    # Get embeddings for all chunks
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"  Embedding chunk {i+1}/{len(chunks)}...")
        embedding = get_embedding(chunk['text'])
        embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
    index.add(embeddings_array)
    
    print(f"✓ Vector database built with {index.ntotal} vectors")
    
    return index, chunks


def retrieve_relevant_chunks(query, index, chunks, top_k=5):
    """
    Retrieve the most relevant chunks for a query using vector similarity
    """
    # Get query embedding
    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding]).astype('float32')
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Get the relevant chunks
    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(chunks):  # Safety check
            relevant_chunks.append(chunks[idx])
    
    return relevant_chunks


def ask_question_rag(ticker=None, question=None, index=None, chunks=None):
    """
    Answer a question using RAG approach:
    1. Fetch fresh data from API
    2. Retrieve relevant chunks from vector database
    3. Generate answer with LLM
    """
    # Interactive mode
    if ticker is None:
        ticker = input("Enter stock ticker (AAPL, MSFT, etc): ").upper()
    if question is None:
        question = input("Enter your question: ")
    
    # Fetch real-time data
    print(f"\nFetching latest data for {ticker}...")
    df = fetch_realtime_data(ticker, days=60)
    
    if df is None or df.empty:
        print("Could not fetch data. Using vector database only.")
        recent_data_context = "No recent data available."
        most_recent_date = "Unknown"
    else:
        # Get the most recent trading day info
        most_recent = df.iloc[-1]
        most_recent_date = most_recent['Date'].strftime('%Y-%m-%d') if hasattr(most_recent['Date'], 'strftime') else str(most_recent['Date'])
        
        recent_data_context = f"""
MOST RECENT DATA (Live from Yahoo Finance - {most_recent_date}):
Date: {most_recent_date}
Open: ${most_recent['Open']:.2f}
High: ${most_recent['High']:.2f}
Low: ${most_recent['Low']:.2f}
Close: ${most_recent['Close']:.2f}
Volume: {int(most_recent['Volume']):,}

Last 7 days summary:
{df.tail(7)[['Date', 'Close', 'Volume']].to_string()}
"""
    
    # Build vector database if not provided
    if index is None or chunks is None:
        print("Building vector database from fetched data...")
        if df is not None and not df.empty:
            chunks = create_document_chunks(df, ticker)
            index, chunks = build_vector_database(chunks)
        else:
            print("Cannot build vector database without data.")
            return None
    
    # Retrieve relevant chunks
    print("Retrieving relevant information...")
    relevant_chunks = retrieve_relevant_chunks(question, index, chunks, top_k=5)
    
    # Build context from retrieved chunks
    retrieved_context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
    
    # Construct prompt
    prompt = f"""You are a financial assistant with access to REAL-TIME stock market data.

IMPORTANT: The data below is LIVE data retrieved from Yahoo Finance as of {datetime.now().strftime('%Y-%m-%d %H:%M')}.
This is NOT historical data - it includes the most recent trading information.

{recent_data_context}

RELEVANT HISTORICAL CONTEXT (from vector database):
{retrieved_context}

Question: {question}

Instructions:
- Answer based on the REAL-TIME data provided above
- Reference specific numbers and dates from the live data
- Compare with historical context if relevant
- Be specific and cite actual values
- If the question asks about very recent events, prioritize the live data
- State the date of the most recent data you're using

Answer:"""

    try:
        answer = call_llm(prompt)
        return answer
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


# Benchmark questions for RAG evaluation
BENCHMARK_QUESTIONS_RAG = {
    "AAPL": [
        "What is the most recent closing price?",
        "What was the highest price in the last 30 days?",
        "What is the current trend - is the stock going up or down?",
        "What was the trading volume on the most recent day?",
        "How does the current price compare to last week?",
    ],
    "MSFT": [
        "What is the most recent closing price?",
        "What was the price change over the last 5 trading days?",
        "What is the current trading volume compared to the average?",
        "Is the stock currently above or below its 7-day average?",
        "What was the highest price reached in the last month?",
    ],
    "GOOGL": [
        "What is the most recent closing price?",
        "What is the current price trend?",
        "What was the trading volume yesterday?",
        "How much has the stock changed in the last week?",
        "What was the lowest price in the last 30 days?",
    ]
}


def run_batch_eval_rag(ticker):
    """Run batch evaluation for RAG system"""
    print(f"\n=== RAG System Batch Evaluation: {ticker} ===\n")
    
    if ticker not in BENCHMARK_QUESTIONS_RAG:
        print(f"No benchmark questions defined for {ticker}")
        return
    
    # Fetch data and build vector database once
    print("Setting up RAG system...")
    df = fetch_realtime_data(ticker, days=60)
    
    if df is None or df.empty:
        print("Could not fetch data. Aborting.")
        return
    
    chunks = create_document_chunks(df, ticker)
    index, chunks = build_vector_database(chunks)
    
    questions = BENCHMARK_QUESTIONS_RAG[ticker]
    results = []
    
    print("\nRunning evaluation questions...\n")
    
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q}")
        result = ask_question_rag(ticker=ticker, question=q, index=index, chunks=chunks)
        
        if result:
            print(f"A{i}: {result}\n")
            results.append({
                "question": q,
                "answer": result,
                "timestamp": datetime.now().isoformat(),
                "system": "RAG"
            })
        else:
            print(f"A{i}: [Error generating answer]\n")
    
    # Save results
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{ticker}_rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "ticker": ticker,
            "model": "rag-" + ("ollama-llama3.2" if USE_OLLAMA else "gpt-4"),
            "timestamp": datetime.now().isoformat(),
            "system": "RAG",
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    return results


def interactive_mode():
    """Run in interactive mode"""
    print("\n=== Stock Market QA System (RAG) ===")
    print(f"Backend: {'Ollama (Local)' if USE_OLLAMA else 'OpenAI GPT-4'}")
    print("Features: Real-time data + Vector retrieval")
    print("Type 'quit' to exit\n")
    
    while True:
        ticker = input("\nEnter stock ticker (or 'quit'): ").upper()
        
        if ticker == 'QUIT':
            break
            
        question = input("Enter your question: ")
        
        if question.lower() == 'quit':
            break
        
        print("\nProcessing (fetching data, building vectors, generating answer)...\n")
        answer = ask_question_rag(ticker=ticker, question=question)
        
        if answer:
            print(f"\nAnswer: {answer}\n")
        else:
            print("Sorry, I couldn't generate an answer.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            # Batch evaluation mode
            if len(sys.argv) < 3:
                print("Usage: python rag_qa.py --batch TICKER")
                sys.exit(1)
            run_batch_eval_rag(sys.argv[2].upper())
        else:
            print("Unknown argument. Use --batch TICKER for batch evaluation")
    else:
        # Interactive mode
        interactive_mode()