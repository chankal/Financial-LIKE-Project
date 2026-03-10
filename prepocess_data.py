import pandas as pd
import os

DATA_FOLDER = "data"
OUTPUT_FOLDER = "processed_data"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_stock_data():

    for file in os.listdir(DATA_FOLDER):

        if not file.endswith(".csv"):
            continue

        path = os.path.join(DATA_FOLDER, file)

        df = pd.read_csv(path)

        df = df.reset_index()

        df = df[["Date","Open","High","Low","Close","Volume"]]

        df = df.dropna()

        ticker = file.replace(".csv","")

        output_path = f"{OUTPUT_FOLDER}/{ticker}_processed.csv"

        df.to_csv(output_path, index=False)

        print(f"Processed {ticker}")

if __name__ == "__main__":
    preprocess_stock_data()