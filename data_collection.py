import yfinance as yf
import pandas as pd
import os
from config import STOCK_TICKERS, START_DATE, END_DATE

DATA_FOLDER = "data"

os.makedirs(DATA_FOLDER, exist_ok=True)

def download_stock_data():

    for ticker in STOCK_TICKERS:
        print(f"Downloading {ticker} data...")

        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE
        )

        filepath = f"{DATA_FOLDER}/{ticker}.csv"

        data.to_csv(filepath)

        print(f"Saved to {filepath}")

if __name__ == "__main__":
    download_stock_data()