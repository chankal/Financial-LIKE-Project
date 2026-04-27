# INSTRUCTIONS.md — Running the LIKE Project

**Project:** Knowledge Obsolescence in Financial QA Systems  
**Student:** Chandini Kalidindi  
**Course:** CS 4365/6365: IEC - Spring 2026  
**Group:** 21

---

## Project Overview

**Research Question:** How does knowledge obsolescence manifest in financial question-answering systems, and can temporal-aware RAG mitigate it?

**Approach:** Compare two QA systems:
1. **Baseline RAG** — Static dataset, no temporal awareness, no confidence scoring
2. **Enhanced RAG** — Real-time data retrieval, temporal validation, confidence scoring, auto-warnings

**Key Insight:** The enhanced system achieves 100% obsolescence detection vs 0% for baseline by tracking data age throughout the pipeline and explicitly warning users when information is stale.

---

## Repository Structure

```
Financial-LIKE-Project/
├── baseline/                      # Baseline RAG system (static data)
│   ├── data/                      # Stock data CSVs (auto-created)
│   ├── evaluation_results/        # Baseline outputs (auto-created)
│   │
│   ├── rag_baseline.py            # Main baseline system
│   ├── evaluation_metrics.py      # Metrics calculation
│   ├── config.py                  # Configuration
│   └── requirements.txt           # Dependencies
│
├── rag/                           # Enhanced RAG system (temporal-aware)
│   ├── vector_db/                 # FAISS indices (auto-created)
│   ├── evaluation_results/        # Enhanced outputs (auto-created)
│   │
│   ├── rag_enhanced.py            # Main enhanced system
│   ├── temporal_validator.py      # Temporal awareness logic
│   ├── evaluation_metrics.py      # Metrics calculation
│   ├── statistical_tests.py       # t-tests, Cohen's d
│   ├── ablation_study.py          # Component ablation analysis
│   ├── visualization.py           # Generate all 8 figures
│   ├── config.py                  # Configuration
│   └── requirements.txt           # Dependencies
│
├── results/                       # All evaluation outputs & figures
│   ├── figures/                   # 8 publication-quality figures (300 DPI)
│   └── evaluation_results/        # JSON outputs from both systems
│
├── INSTRUCTIONS.md                # This file
└── README.md                      # Project summary
```

---

## System Requirements

- **Python:** 3.10 or higher
- **RAM:** 8GB minimum (16GB recommended)
- **Internet:** Required for Yahoo Finance API
- **Disk Space:** ~1GB for dependencies + vector databases

### LLM Backend Options

| Backend | Cost | Setup | Speed |
|---------|------|-------|-------|
| **Ollama** (Recommended) | FREE | Medium | Fast |
| **OpenAI GPT-4** | Paid | Easy | Very Fast |

---

## Setup Instructions

### Step 1: Clone Repository

```bash
git clone https://github.com/chankal/Financial-LIKE-Project.git
cd Financial-LIKE-Project
```

### Step 2: Choose LLM Backend

#### Option A: Ollama (FREE, Recommended)

**Install Ollama:**
1. Download from https://ollama.com/download
2. Install for your OS (macOS, Linux, Windows)
3. Verify: `ollama --version`

**Pull Required Models:**
```bash
ollama pull llama3.2          # For answer generation
ollama pull nomic-embed-text  # For embeddings

# Verify
ollama list
```

**Start Ollama Server:**
```bash
ollama serve
# Keep this running in a separate terminal
```

**Configure Systems:**
```python
# Edit baseline/rag_baseline.py line ~20
USE_OLLAMA = True

# Edit rag/rag_enhanced.py line ~20
USE_OLLAMA = True
```

#### Option B: OpenAI GPT-4 (Paid)

**Set API Key:**
```bash
# macOS/Linux
export OPENAI_API_KEY='sk-your-key-here'

# Windows Command Prompt
set OPENAI_API_KEY=sk-your-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"
```

**Configure Systems:**
```python
# Edit baseline/rag_baseline.py line ~20
USE_OLLAMA = False

# Edit rag/rag_enhanced.py line ~20
USE_OLLAMA = False
```

---

## Running the Baseline System

### What the Baseline Does
- Fetches static AAPL stock data (one-time download)
- Creates document chunks without temporal metadata
- Builds FAISS vector index for retrieval
- Generates answers using LLM
- **Does NOT** track data age, provide confidence scores, or warn users

### Setup
```bash
cd baseline
pip install -r requirements.txt
```

### Interactive Mode

```bash
python rag_baseline.py
```

**Example:**
```
Enter stock ticker: AAPL
Enter your question: What is the current stock price?

Answer: Apple's current stock price is $271.06
```

**Problem:** No indication that data is 2.7 days old!

### Batch Evaluation Mode

```bash
python rag_baseline.py --batch AAPL
```

**What this runs:**
- 5 test questions on AAPL stock data
- Questions ask about "current" prices/trends
- Saves results to `evaluation_results/AAPL_baseline_results_TIMESTAMP.json`

**Output:**
```json
{
  "ticker": "AAPL",
  "data_date": "2026-04-24",
  "evaluation_date": "2026-04-27",
  "data_age_days": 2.7,
  "questions": [
    {
      "question": "What is the most recent closing price?",
      "answer": "Apple's current stock price is $271.06",
      "has_temporal_awareness": false,
      "has_confidence_score": false,
      "has_warning": false,
      "obsolescence_detected": false
    }
    // ... 4 more questions
  ],
  "summary": {
    "obsolescence_rate": 0.60,
    "warnings_provided": 0
  }
}
```

**Time:** ~2-3 minutes

---

## Running the Enhanced RAG System

### What the Enhanced System Does
- Fetches fresh AAPL data (latest 60 days) on every query
- Calculates data age in hours/days
- Categorizes freshness (real-time, fresh, recent, stale)
- Builds vector index with temporal metadata
- Generates confidence scores (0-1 scale)
- Performs self-verification on temporal claims
- Adds warnings when confidence < 0.6
- Categorizes obsolescence risk (LOW/MEDIUM/HIGH)

### Setup
```bash
cd rag
pip install -r requirements.txt
```

### Interactive Mode

```bash
python rag_enhanced.py
```

**Example:**
```
Enter stock ticker: AAPL
Enter your question: What is the current stock price?

[1/7] Fetching latest data...
[2/7] Validating data freshness... (2.7 days old)
[3/7] Creating chunks with temporal metadata...
[4/7] Building vector index...
[5/7] Retrieving relevant chunks...
[6/7] Generating answer with confidence scoring...
[7/7] Self-verification...

Answer:
Based on data from April 24, 2026, Apple's closing price was $271.06.

Important Notes:
• Data is 2.7 days old (66 hours)
• Confidence level: LOW (0.60)
• Obsolescence risk: MEDIUM
• May not reflect current market conditions
```

### Batch Evaluation Mode

```bash
python rag_enhanced.py --batch AAPL
```

**What this runs:**
- Same 5 test questions as baseline
- Full temporal awareness pipeline
- Calculates confidence scores and risk levels
- Saves results to `evaluation_results/AAPL_enhanced_results_TIMESTAMP.json`

**Output:**
```json
{
  "ticker": "AAPL",
  "data_date": "2026-04-24",
  "evaluation_date": "2026-04-27",
  "data_age_days": 2.7,
  "questions": [
    {
      "question": "What is the most recent closing price?",
      "answer": "Based on data from April 24, 2026...",
      "confidence": 0.60,
      "confidence_category": "LOW",
      "risk_level": "MEDIUM",
      "has_warning": true,
      "obsolescence_detected": true,
      "freshness_category": "recent",
      "temporal_mismatch": true
    }
    // ... 4 more questions
  ],
  "summary": {
    "mean_confidence": 0.61,
    "obsolescence_detection_rate": 1.00,
    "warnings_provided": 5
  }
}
```

**Time:** ~3-4 minutes

---

## Comparing Both Systems

### What the Comparison Script Does
- Loads baseline and enhanced results
- Compares obsolescence detection rates
- Compares confidence scores
- Runs statistical significance tests
- Generates summary tables

### Run Comparison

```bash
cd rag
python compare_systems.py --latest AAPL
```

**Output:**
```
=== SYSTEM COMPARISON ===

Obsolescence Detection:
  Baseline:  0/5 (0%)
  Enhanced:  5/5 (100%)

Confidence Scores:
  Baseline:  Mean = 0.85 (HIGH - unaware)
  Enhanced:  Mean = 0.61 (LOW - aware)
  Difference: -0.24

Statistical Significance:
  t(4) = 13.60
  p-value < 0.001
  Cohen's d = 3.59 (very large effect)
  
Interpretation: Enhanced system appropriately lowers
confidence when data is stale. Lower confidence = SUCCESS.
```

**Time:** ~30 seconds

---

## Running Statistical Tests

### What the Statistical Tests Script Does
- Paired t-test (baseline vs enhanced confidence)
- Effect size calculation (Cohen's d)
- 95% confidence intervals
- Permutation test (non-parametric validation)
- Saves all results to JSON

### Run Statistical Analysis

```bash
cd rag
python statistical_tests.py
```

**Output:**
```
=== STATISTICAL SIGNIFICANCE ANALYSIS ===

Paired t-test:
  t-statistic: 13.60
  p-value: < 0.001
  Degrees of freedom: 4
  
Effect Size:
  Cohen's d: 3.59
  Interpretation: Very large effect
  
95% Confidence Interval:
  [0.114, 0.151]
  Excludes zero: YES
  
Permutation Test (10,000 iterations):
  p-value: < 0.001
  Confirms t-test results
  
Conclusion: The difference is statistically significant
and NOT due to chance.

Results saved: evaluation_results/statistical_tests_results.json
```

**Time:** ~1-2 minutes

---

## Running Component Ablation Study

### What the Ablation Study Does
- Tests 6 system variants by removing one component at a time:
  1. Full system (all components)
  2. No self-verification
  3. No recency weighting
  4. No vector search
  5. No real-time data
  6. Baseline (no enhancements)
- Measures impact of each component on confidence scores
- Identifies which components are critical vs incremental

### Run Ablation Study

```bash
cd rag
python ablation_study.py
```

**Output:**
```
=== COMPONENT ABLATION STUDY ===

Testing 6 variants on AAPL (data age: 2.7 days)

Variant                    | Confidence | Contribution
---------------------------|------------|-------------
Full System                | 0.606      | Baseline
No Self-Verification       | 0.606      | +0.000
No Vector Search           | 0.700      | +0.094
No Recency Weighting       | 0.750      | +0.144
No Real-Time Data          | 0.000      | -0.606 ⚠️
Baseline                   | 0.000      | Reference

Component Ranking:
  1. Real-time Data:      +0.606  [CRITICAL]
  2. Self-Verification:   +0.000
  3. Vector Search:       -0.094
  4. Recency Weighting:   -0.144

Key Finding: Without real-time data, system completely
fails (confidence = 0). Real-time data is the foundation.

Results saved: evaluation_results/ablation_study_results.json
```

**Time:** ~5-8 minutes

---

## Generating Figures

### What the Visualization Script Does
- Generates 8 publication-quality figures (300 DPI):
  1. Obsolescence Detection Rate comparison
  2. Component Contribution (Ablation Study)
  3. Confidence Score Distribution
  4. Confidence Decay Over Time
  5. Retrieval & Generation Metrics (Radar Chart)
  6. Feature Comparison Heatmap
  7. Statistical Significance Analysis
  8. Example Output Comparison
- Saves as PNG and PDF to `results/figures/`

### Generate All Figures

```bash
cd rag
python visualization.py --all
```

**Output:**
```
=== GENERATING FIGURES ===

[1/8] Figure 1: Obsolescence Detection Rate... ✓
[2/8] Figure 2: Component Ablation... ✓
[3/8] Figure 3: Confidence Distribution... ✓
[4/8] Figure 4: Temporal Decay... ✓
[5/8] Figure 5: Retrieval & Generation Metrics... ✓
[6/8] Figure 6: Feature Comparison... ✓
[7/8] Figure 7: Statistical Significance... ✓
[8/8] Figure 8: Example Outputs... ✓

All figures saved to: results/figures/
  - PNG format (300 DPI)
  - PDF format (vector)
```

### Generate Individual Figures

```bash
python visualization.py --fig 3  # Just confidence distribution
python visualization.py --fig 2  # Just ablation study
```

**Time:** ~2-3 minutes for all figures

---

## Complete Workflow

### Full End-to-End Execution

```bash
# 1. Clone and Setup
git clone https://github.com/chankal/Financial-LIKE-Project.git
cd Financial-LIKE-Project

# 2. Setup Ollama (if using)
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve  # Keep running in separate terminal

# 3. Run Baseline
cd baseline
pip install -r requirements.txt
python rag_baseline.py --batch AAPL

# 4. Run Enhanced
cd ../rag
pip install -r requirements.txt
python rag_enhanced.py --batch AAPL

# 5. Compare Systems
python compare_systems.py --latest AAPL

# 6. Statistical Tests
python statistical_tests.py

# 7. Ablation Study
python ablation_study.py

# 8. Generate Figures
python visualization.py --all
```

**Total Time:** ~15-20 minutes  
**Outputs:** Complete replication of all project results

---

## What Each Script Accomplishes

### Baseline Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `rag_baseline.py` | Main baseline RAG system | Stock ticker, questions | Answers without temporal awareness |
| `rag_baseline.py --batch` | Batch evaluation | AAPL ticker | `evaluation_results/AAPL_baseline_results_*.json` |
| `evaluation_metrics.py` | Calculate retrieval/generation quality | Results JSON | Metrics: Recall, NDCG, Token F1, etc. |

### Enhanced Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `rag_enhanced.py` | Main enhanced RAG system | Stock ticker, questions | Answers with temporal awareness |
| `rag_enhanced.py --batch` | Batch evaluation | AAPL ticker | `evaluation_results/AAPL_enhanced_results_*.json` |
| `compare_systems.py` | Compare baseline vs enhanced | Both result files | Comparison table + statistics |
| `statistical_tests.py` | Statistical significance | Both result files | t-test, Cohen's d, p-values |
| `ablation_study.py` | Test component importance | AAPL ticker | Component contribution analysis |
| `visualization.py` | Generate all figures | All result files | 8 PNG/PDF figures |
| `temporal_validator.py` | Calculate data freshness | Timestamps | Age, freshness category, confidence |
| `evaluation_metrics.py` | Calculate all metrics | Results + ground truth | Complete metric suite |

---

## Expected Results

### Key Findings

| Metric | Baseline | Enhanced |
|--------|----------|----------|
| Obsolescence Detection | 0% (0/5) | 100% (5/5) |
| Mean Confidence | 0.85 (HIGH) | 0.61 (LOW) |
| Warnings Provided | 0/5 (0%) | 5/5 (100%) |
| Statistical Significance | — | t=13.60, p<0.001, d=3.59 |

**Interpretation:** Lower confidence in enhanced = SUCCESS (correctly reflects uncertainty with stale data)

### File Outputs

**Baseline:**
```
baseline/evaluation_results/
└── AAPL_baseline_results_20260427_140523.json
```

**Enhanced:**
```
rag/evaluation_results/
├── AAPL_enhanced_results_20260427_140623.json
├── statistical_tests_results.json
├── ablation_study_results.json
└── comparison_results.json
```

**Figures:**
```
results/figures/
├── figure1_obsolescence_detection.png
├── figure2_component_ablation.png
├── figure3_confidence_distribution.png
├── figure4_temporal_decay.png
├── figure5_retrieval_generation_metrics.png
├── figure6_feature_comparison.png
├── figure7_statistical_significance.png
└── figure8_example_outputs.png
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Connection refused: localhost:11434` | Start Ollama: `ollama serve` |
| `AuthenticationError` | Set `OPENAI_API_KEY` environment variable |
| `model not found` | Pull model: `ollama pull llama3.2` |
| `ImportError: faiss` | Install: `pip install faiss-cpu --no-cache-dir` |
| Yahoo Finance timeout | Wait 60 seconds and retry |

### Verification

```bash
# Check Python version (need 3.10+)
python --version

# Check dependencies
pip list | grep -E "faiss|yfinance|pandas|scipy|matplotlib"

# Check Ollama models
ollama list

# Test systems
cd baseline && python rag_baseline.py --test
cd ../rag && python rag_enhanced.py --test
```

---

## Quick Reference

```bash
# Baseline evaluation
cd baseline
python rag_baseline.py --batch AAPL

# Enhanced evaluation
cd rag
python rag_enhanced.py --batch AAPL

# Comparison
python compare_systems.py --latest AAPL

# Statistics
python statistical_tests.py

# Ablation
python ablation_study.py

# Figures
python visualization.py --all
```

---

## Contact

- **Student:** Chandini Kalidindi
- **GitHub:** https://github.com/chankal/Financial-LIKE-Project
- **Course:** CS 4365/6365: IEC - Spring 2026

---
