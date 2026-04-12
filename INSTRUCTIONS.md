# INSTRUCTIONS.md — AI Workflow Guide


---

## Project Overview

**Goal:** Investigate how quickly financial question-answering systems become outdated as stock market conditions change.

**Research Question:** How does LLM knowledge obsolescence manifest in financial QA, and can Retrieval-Augmented Generation (RAG) mitigate it?

**Approach:** Compare two QA systems:
1. **Baseline QA** *(implemented)* — uses static historical stock data + LLM knowledge only (no live data)
2. **RAG-based QA** *(implemented)* — retrieves fresh financial data at query time using vector search to reduce knowledge staleness

**Current Status (Checkpoint 4 - Week 10):**
- Baseline QA system fully implemented and tested
- Data collection and preprocessing pipeline operational
- Evaluation framework with benchmark questions
- RAG system with FAISS vector search implemented
- Real-time data retrieval integrated
- Comparison framework for measuring obsolescence
- (In progress) - Comprehensive testing and final analysis (Week 11-12)

---

## Repository Structure

```
Financial-LIKE-Project/
├── Baseline/                      # Baseline QA system (static data)
│   ├── data/                      # Raw CSV stock data
│   ├── sample_data/               # Sample CSVs for testing
│   ├── processed_data/            # Cleaned CSVs (auto-created)
│   ├── evaluation_results/        # Baseline evaluation outputs (auto-created)
│   │
│   ├── data_collection.py         # Downloads stock data from Yahoo Finance
│   ├── prepocess_data.py          # Cleans raw CSVs → processed_data/
│   ├── baseline_qa.py             # Baseline QA system (supports OpenAI & Ollama)
│   ├── evaluate.py                # Automated evaluation & accuracy scoring
│   ├── evaluation_benchmarks.json # 30+ structured test questions
│   ├── config.py                  # Configuration settings
│   └── requirements.txt           # Python dependencies
│
├── rag/                           # RAG QA system (real-time data + vector search)
│   ├── vector_db/                 # FAISS indices (auto-created)
│   ├── evaluation_results/        # RAG evaluation outputs (auto-created)
│   │
│   ├── rag_qa.py                  # RAG QA system with FAISS & real-time data
│   ├── compare_systems.py         # Baseline vs RAG comparison tool
│   ├── config.py                  # RAG configuration settings
│   ├── requirements.txt           # RAG-specific dependencies
│   └── README.md                  # RAG documentation
│
├── INSTRUCTIONS.md                # This file
└── README.md                      # Project summary
```

> **Important:** Note the capital 'B' in `Baseline/` and lowercase in `rag/`. Commands should be run from the appropriate directory.

---

## Prerequisites

### System Requirements
- **Python:** 3.8 or higher (3.9+ recommended)
- **RAM:** 4GB minimum (8GB recommended for RAG with Ollama)
- **Internet:** Required for data collection from Yahoo Finance
- **Disk Space:** ~500MB for dependencies + vector databases

### LLM Backend Options (Choose ONE)

| Backend | Cost | Setup | Best For |
|---------|------|-------|----------|
| **Ollama** (Recommended) | FREE | Medium | Development, unlimited testing |
| **OpenAI** | Paid | Easy | Production, highest quality |

Both systems support both backends seamlessly.

---

## Quick Start Guide

### Option 1: Test with Sample Data (Fastest)

```bash
# Clone repository
git clone https://github.com/chankal/Financial-LIKE-Project.git
cd Financial-LIKE-Project

# Test Baseline system
cd Baseline
pip install -r requirements.txt
cp sample_data/*.csv data/
python prepocess_data.py
python baseline_qa.py --batch AAPL

# Test RAG system
cd ../rag
pip install -r requirements.txt
python rag_qa.py --batch AAPL

# Compare results
python compare_systems.py --latest AAPL
```

### Option 2: Full Setup with Live Data

```bash
# Clone and setup
git clone https://github.com/chankal/Financial-LIKE-Project.git
cd Financial-LIKE-Project

# Setup Baseline
cd Baseline
pip install -r requirements.txt
python data_collection.py      # Fetch fresh data
python prepocess_data.py        # Clean data
python baseline_qa.py --batch AAPL  # Run evaluation

# Setup RAG
cd ../rag
pip install -r requirements.txt
python rag_qa.py --batch AAPL  # Run RAG evaluation (fetches live data automatically)

# Compare
python compare_systems.py --latest AAPL
```

---

## Detailed Setup Instructions

### Step 1: Clone Repository

```bash
git clone https://github.com/chankal/Financial-LIKE-Project.git
cd Financial-LIKE-Project
```

### Step 2: Choose and Configure LLM Backend

#### Option A: Ollama (FREE, Local, Recommended)

**Install Ollama:**
1. Download from https://ollama.com/download
2. Install for your operating system
3. Verify: `ollama --version`

**Pull Required Models:**
```bash
# For answer generation
ollama pull llama3.2

# For RAG embeddings
ollama pull nomic-embed-text

# Verify
ollama list
```

**Start Ollama Server:**
```bash
ollama serve
# Leave this running in a separate terminal
```

**Configure in Code:**
- Edit `Baseline/baseline_qa.py`, line ~13: `USE_OLLAMA = True`
- Edit `rag/rag_qa.py`, line ~13: `USE_OLLAMA = True`

#### Option B: OpenAI (Paid, Cloud)

**Get API Key:**
1. Sign up at https://platform.openai.com
2. Create API key at https://platform.openai.com/api-keys
3. Add billing at https://platform.openai.com/account/billing

**Set Environment Variable:**
```bash
# macOS/Linux
export OPENAI_API_KEY='your-api-key-here'

# Windows Command Prompt
set OPENAI_API_KEY=your-api-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Configure in Code:**
- Edit `Baseline/baseline_qa.py`, line ~13: `USE_OLLAMA = False`
- Edit `rag/rag_qa.py`, line ~13: `USE_OLLAMA = False`

---

## Running the Baseline System

### Setup
```bash
cd Baseline
pip install -r requirements.txt
```

### Step 1: Collect Stock Data
```bash
python data_collection.py
```
- Downloads last 5-6 years of data from Yahoo Finance
- Default tickers: AAPL, MSFT, GOOGL, TSLA, AMZN
- Saves to `data/{TICKER}.csv`
- **Skip this:** Use `cp sample_data/*.csv data/` to use sample data

### Step 2: Preprocess Data
```bash
python prepocess_data.py
```
> Note: Filename has typo (one 'r')

- Cleans raw CSVs
- Removes missing values
- Saves to `processed_data/{TICKER}_processed.csv`

### Step 3: Run Baseline QA

**Interactive Mode:**
```bash
python baseline_qa.py
```
Example:
```
Enter stock ticker: AAPL
Enter your question: What was the closing price on the most recent day?

Answer: Based on the historical data ending 2025-01-15, 
        Apple's closing price was $234.58.
```

**Batch Evaluation Mode:**
```bash
python baseline_qa.py --batch AAPL
```
- Runs all benchmark questions for AAPL
- Saves results to `evaluation_results/AAPL_baseline_results_*.json`

### Step 4: Evaluate Results
```bash
python evaluate.py evaluation_results/AAPL_baseline_results_*.json
```
- Compares answers to ground truth
- Calculates accuracy metrics
- Detects hallucinations
- Generates detailed report

---

## Running the RAG System

### Setup
```bash
cd rag
pip install -r requirements.txt
```

### Step 1: Run RAG QA (All-in-One)

The RAG system fetches fresh data automatically - no separate data collection step needed!

**Interactive Mode:**
```bash
python rag_qa.py
```
Example:
```
Enter stock ticker: AAPL
Enter your question: What is the current stock price?

Fetching latest data for AAPL...
Building vector database with 95 chunks...
Retrieving relevant information...

Answer: Based on live data from Yahoo Finance retrieved on 2026-03-30, 
        Apple's most recent closing price is $247.82 as of March 29, 2026. 
        This represents a 2.3% increase from yesterday's close.
```

**Batch Evaluation Mode:**
```bash
python rag_qa.py --batch AAPL
```
- Fetches latest 60 days of data from Yahoo Finance
- Builds FAISS vector database
- Runs all RAG benchmark questions
- Saves results to `evaluation_results/AAPL_rag_results_*.json`

### Step 2: Compare Systems

**Compare Latest Evaluations:**
```bash
python compare_systems.py --latest AAPL
```
- Finds most recent baseline and RAG results
- Shows side-by-side comparison
- Highlights differences in recency and accuracy

**Compare Specific Files:**
```bash
python compare_systems.py \
    ../Baseline/evaluation_results/AAPL_baseline_results.json \
    evaluation_results/AAPL_rag_results.json
```

**Analyze Obsolescence Over Time:**
```bash
python compare_systems.py --obsolescence AAPL
```
- Compares multiple evaluation runs
- Measures how answers change over time
- Demonstrates knowledge decay

---

## Complete Workflow Example

Here's a full end-to-end workflow an AI can execute:

```bash
# ========================================
# COMPLETE PROJECT WORKFLOW
# ========================================

# 1. Setup
git clone https://github.com/chankal/Financial-LIKE-Project.git
cd Financial-LIKE-Project

# 2. Setup Baseline System
cd Baseline
pip install -r requirements.txt

# Configure LLM backend (choose one):
# - Edit baseline_qa.py: USE_OLLAMA = True  (for Ollama)
# - Edit baseline_qa.py: USE_OLLAMA = False (for OpenAI)

# 3. Collect and process data
python data_collection.py
python prepocess_data.py

# 4. Run baseline evaluation
python baseline_qa.py --batch AAPL
python baseline_qa.py --batch MSFT
python baseline_qa.py --batch GOOGL

# 5. Evaluate baseline accuracy
python evaluate.py evaluation_results/AAPL_baseline_results_*.json

# 6. Setup RAG System
cd ../rag
pip install -r requirements.txt

# Configure LLM backend (same as baseline):
# - Edit rag_qa.py: USE_OLLAMA = True/False

# 7. Run RAG evaluation (fetches fresh data automatically)
python rag_qa.py --batch AAPL
python rag_qa.py --batch MSFT
python rag_qa.py --batch GOOGL

# 8. Compare systems
python compare_systems.py --latest AAPL
python compare_systems.py --latest MSFT
python compare_systems.py --latest GOOGL

# 9. Analyze obsolescence
python compare_systems.py --obsolescence AAPL
```

---

## Key Differences: Baseline vs RAG

| Feature | Baseline | RAG |
|---------|----------|-----|
| **Data Source** | Static CSV files | Live Yahoo Finance API |
| **Data Freshness** | Fixed at preprocessing time | Real-time at query time |
| **Retrieval Method** | Last 30 rows only | Vector similarity search (FAISS) |
| **Context** | Fixed window | Semantically relevant chunks |
| **Obsolescence** | High (ages quickly) | Low (always fresh) |
| **Accuracy on Recent Events** | Poor | Excellent |
| **Setup Complexity** | Low | Medium |
| **Query Speed** | Fast (~2s) | Slower (~5-8s) |
| **Cost** | Low | Higher (API calls + embeddings) |

---

## System Architectures

### Baseline Architecture
```
User Question
     ↓
[Load Last 30 Rows] ← processed_data/TICKER_processed.csv
     ↓
[Build Prompt] → "Here's historical data..."
     ↓
[LLM (GPT-4/Llama)] → Generate answer
     ↓
Final Answer (may be outdated)
```

### RAG Architecture
```
User Question
     ↓
[Fetch Real-Time Data] ← Yahoo Finance API (last 60 days)
     ↓
[Create Chunks] → Daily + Weekly summaries
     ↓
[Generate Embeddings] → OpenAI/Ollama embedding model
     ↓
[Build FAISS Index] → Vector database
     ↓
[Query Embedding] → Convert question to vector
     ↓
[Vector Search] → Find top-5 most relevant chunks
     ↓
[Build Prompt] → "Live data + Retrieved context..."
     ↓
[LLM (GPT-4/Llama)] → Generate answer
     ↓
Final Answer (always fresh)
```

---

## Configuration Files

### Baseline/config.py
```python
# Tickers to collect/process
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

# Date ranges
START_DATE = '2019-01-01'
END_DATE = '2025-01-31'

# File paths
DATA_DIR = 'data/'
PROCESSED_DIR = 'processed_data/'
```

### rag/config.py
```python
# LLM backend
USE_OLLAMA = True  # or False for OpenAI

# Models
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

# Retrieval settings
TOP_K_CHUNKS = 5
DATA_FETCH_DAYS = 60

# Vector database
VECTOR_DB_DIR = "vector_db"
```

---

## Evaluation Metrics

### Accuracy Metrics
- **Factual Accuracy:** Percentage of correct specific values
- **Temporal Accuracy:** Correct understanding of trends over time
- **Comparative Accuracy:** Correct period-to-period comparisons

### Obsolescence Metrics
- **Data Recency:** How recent is the most recent data point used?
- **Answer Decay:** How do answers change over time for same question?
- **Hallucination Rate:** Percentage of fabricated information

### Expected Results
| Metric | Baseline | RAG | Improvement |
|--------|----------|-----|-------------|
| Factual Accuracy | ~75% | ~90% | +20% |
| Temporal Accuracy | ~60% | ~90% | +50% |
| Hallucination Rate | ~15% | ~5% | -67% |
| Data Recency | Days/weeks old | Hours old | ✓ |

---

## Troubleshooting

### Common Issues - Both Systems

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError` | Dependencies not installed | `pip install -r requirements.txt` |
| `Connection refused: localhost:11434` | Ollama not running | `ollama serve` in separate terminal |
| `AuthenticationError` (OpenAI) | Missing/invalid API key | Set `OPENAI_API_KEY` environment variable |
| `model not found` (Ollama) | Model not pulled | `ollama pull llama3.2` |

### Common Issues - Baseline

| Problem | Cause | Solution |
|---------|-------|----------|
| `FileNotFoundError: processed_data/` | Preprocessing not run | `python prepocess_data.py` |
| `KeyError: 'Date'` | Bad CSV format | Re-run `python data_collection.py` |
| Empty `data/` directory | Data not collected | `python data_collection.py` or use sample_data |

### Common Issues - RAG

| Problem | Cause | Solution |
|---------|-------|----------|
| `No module named 'faiss'` | FAISS not installed | `pip install faiss-cpu --no-cache-dir` |
| Yahoo Finance timeout | API rate limit | Wait 60 seconds and retry |
| Out of memory | Too many chunks | Reduce `DATA_FETCH_DAYS` in config |
| Slow embedding | Many API calls | Use Ollama (local) instead of OpenAI |

### Verification Commands

```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip list | grep -E "faiss|yfinance|pandas|openai"

# Check Ollama
ollama list  # Should show llama3.2 and nomic-embed-text

# Check FAISS
python -c "import faiss; print('✓ FAISS OK')"

# Check file structure
ls Baseline/processed_data/  # Should have *_processed.csv files
ls rag/vector_db/  # May be empty initially
```

---

## Testing Checklist

### Baseline System
- [ ] Dependencies installed: `pip list | grep pandas`
- [ ] Data collected: `ls Baseline/data/*.csv`
- [ ] Data preprocessed: `ls Baseline/processed_data/*.csv`
- [ ] Interactive mode works: `python baseline_qa.py`
- [ ] Batch mode works: `python baseline_qa.py --batch AAPL`
- [ ] Evaluation works: `python evaluate.py evaluation_results/*.json`

### RAG System
- [ ] Dependencies installed: `pip list | grep faiss`
- [ ] FAISS imports: `python -c "import faiss"`
- [ ] Embedding model ready: `ollama list | grep nomic-embed-text`
- [ ] Interactive mode works: `python rag_qa.py`
- [ ] Batch mode works: `python rag_qa.py --batch AAPL`
- [ ] Comparison works: `python compare_systems.py --latest AAPL`

### Integration Testing
- [ ] Both systems run on same ticker: AAPL
- [ ] Results saved to evaluation_results/
- [ ] Comparison shows differences
- [ ] RAG shows more recent dates than baseline

---

## File Sizes & Performance

### Baseline
- `baseline_qa.py`: ~15 KB, ~200 lines
- `processed_data/*.csv`: ~50-100 KB per ticker
- Query time: ~2 seconds
- No significant disk usage

### RAG
- `rag_qa.py`: ~20 KB, ~450 lines
- `vector_db/*.index`: ~1-5 MB per ticker (auto-created)
- `evaluation_results/*.json`: ~5-20 KB per run
- Query time: ~5-8 seconds (includes data fetch + vector search)
- Disk usage: ~50 MB for 5 tickers

---

## Expected Outputs

### Baseline Output Example
```
Q: What is the most recent closing price?
A: Based on the historical data ending 2025-01-15, 
   Apple's closing price was $234.58.
```

### RAG Output Example
```
Q: What is the most recent closing price?
A: Based on live data from Yahoo Finance retrieved on 2026-03-30 14:25, 
   Apple's most recent closing price is $247.82 as of March 29, 2026. 
   This represents a 5.6% increase from the previous month.
```

**Key Differences:**
- RAG explicitly states data source and retrieval time
- RAG provides more recent date (March 2026 vs January 2025)
- RAG includes additional context (percentage change)
- RAG has awareness it's using fresh data

---

## Advanced Usage

### Custom Tickers
Edit `Baseline/config.py` or modify data collection:
```python
# In data_collection.py
tickers = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'INTC']
```

### Custom Questions
Edit `evaluation_benchmarks.json` to add your own:
```json
{
  "id": "AAPL_CUSTOM1",
  "question": "Your custom question here",
  "type": "factual",
  "difficulty": "easy"
}
```

### Adjust RAG Retrieval
Edit `rag/config.py`:
```python
TOP_K_CHUNKS = 10  # Retrieve more chunks for context
DATA_FETCH_DAYS = 90  # Get more historical data
```

### Change Embedding Model
Edit `rag/rag_qa.py`:
```python
# For OpenAI
EMBEDDING_MODEL = "text-embedding-3-large"  # Better quality, higher cost

# For Ollama
EMBEDDING_MODEL = "all-minilm"  # Faster, smaller
```

---

## Project Timeline & Deliverables

### Week 5-8: Baseline System ✅
- Data collection pipeline
- Baseline QA implementation
- Evaluation framework
- 30+ benchmark questions

### Week 9-10: RAG System ✅
- FAISS vector database integration
- Real-time data retrieval
- RAG QA implementation
- Comparison framework

### Week 11-12: Testing & Analysis ⏳
- Comprehensive benchmarking
- Obsolescence measurement
- Performance comparison
- Final report preparation

### Week 13-15: Completion ⏳
- Final testing
- Demo preparation
- Report writing
- Presentation

---

## Dependencies

### Baseline (from Baseline/requirements.txt)
```
pandas>=1.5.0
yfinance>=0.2.0
openai>=1.0.0
requests>=2.28.0
python-dotenv>=0.19.0
```

### RAG (from rag/requirements.txt)
```
pandas>=1.5.0
numpy>=1.24.0
yfinance>=0.2.0
faiss-cpu>=1.7.4
openai>=1.0.0
requests>=2.28.0
python-dotenv>=0.19.0
```

Install: `pip install -r requirements.txt` in each directory

---

## Contact & Support

- **Student:** Chandini Kalidindi
- **Repository:** https://github.com/chankal/Financial-LIKE-Project
- **Course:** CS 4365/6365: IEC - Spring 2026
- **For Issues:** Reach out to course TAs or create GitHub issue

---

## Summary for AI Agents

This project compares two stock market QA systems to measure knowledge obsolescence:

### Quick Execution Path:
```bash
# 1. Clone
git clone https://github.com/chankal/Financial-LIKE-Project.git

# 2. Baseline
cd Baseline && pip install -r requirements.txt
python data_collection.py && python prepocess_data.py
python baseline_qa.py --batch AAPL

# 3. RAG
cd ../rag && pip install -r requirements.txt
python rag_qa.py --batch AAPL

# 4. Compare
python compare_systems.py --latest AAPL
```

### Key Features:
- **Baseline:** Static data, fast, simple, ages quickly
- **RAG:** Real-time data, FAISS vectors, slow, always fresh
- **Goal:** Prove RAG reduces obsolescence in financial QA

### Success Metrics:
- RAG should show more recent dates in answers
- RAG should have higher accuracy on temporal questions
- RAG should maintain accuracy over time while baseline degrades
