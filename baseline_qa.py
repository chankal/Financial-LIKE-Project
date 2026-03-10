import pandas as pd
from openai import OpenAI

client = OpenAI()

def load_stock_context(ticker):

    df = pd.read_csv(f"processed_data/{ticker}_processed.csv")

    context = df.tail(30).to_string()

    return context


def ask_question():

    ticker = input("Enter stock ticker (AAPL, MSFT, etc): ")
    question = input("Enter your question: ")

    context = load_stock_context(ticker)

    prompt = f"""
You are a financial assistant.

Historical stock data:
{context}

Question:
{question}

Answer the question using the historical data above.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}]
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    ask_question()