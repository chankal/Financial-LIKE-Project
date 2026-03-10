# Instructions for AI Workflow

## Setup

Clone the repository and install dependencies.

pip install -r requirements.txt

## Data Collection

Download stock data from Yahoo Finance.

python data_collection.py

## Data Preprocessing

Clean stock datasets.

python preprocess_data.py

## Baseline QA System

Run the baseline system.

python baseline_qa.py

Enter a stock ticker and question when prompted.

Example question:
"What was Apple's price trend last month?"

The baseline system answers using historical data and LLM knowledge only.