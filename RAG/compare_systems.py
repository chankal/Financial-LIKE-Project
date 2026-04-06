
import json
import sys
import os
from datetime import datetime
import glob


def load_evaluation_results(filepath):
    """Load evaluation results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_systems(baseline_file, rag_file):
    """
    Compare baseline and RAG system results
    Generates side-by-side comparison and improvement metrics
    """
    baseline_data = load_evaluation_results(baseline_file)
    rag_data = load_evaluation_results(rag_file)
    
    ticker = baseline_data.get('ticker', 'Unknown')
    
    print("\n" + "="*80)
    print(f"BASELINE vs RAG COMPARISON: {ticker}")
    print("="*80)
    
    # System info
    print(f"\nBaseline Model: {baseline_data.get('model', 'Unknown')}")
    print(f"RAG Model: {rag_data.get('model', 'Unknown')}")
    print(f"Baseline Timestamp: {baseline_data.get('timestamp', 'Unknown')}")
    print(f"RAG Timestamp: {rag_data.get('timestamp', 'Unknown')}")
    
    # Get results
    baseline_results = baseline_data.get('results', [])
    rag_results = rag_data.get('results', [])
    
    print(f"\nQuestions Answered:")
    print(f"  Baseline: {len(baseline_results)}")
    print(f"  RAG: {len(rag_results)}")
    
    # Side-by-side comparison
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    
    max_questions = max(len(baseline_results), len(rag_results))
    
    for i in range(max_questions):
        print(f"\n--- Question {i+1} ---")
        
        if i < len(baseline_results):
            baseline_q = baseline_results[i]
            print(f"Q: {baseline_q['question']}")
            print(f"\n[BASELINE] {baseline_q['answer']}")
        else:
            print("[BASELINE] No answer")
        
        if i < len(rag_results):
            rag_q = rag_results[i]
            if i >= len(baseline_results):
                print(f"Q: {rag_q['question']}")
            print(f"\n[RAG] {rag_q['answer']}")
        else:
            print("[RAG] No answer")
        
        print("\n" + "-"*80)
    
    # Analysis
    print("\n" + "="*80)
    print("QUALITATIVE ANALYSIS")
    print("="*80)
    
    print("""
Key Differences to Look For:
1. Recency: Does RAG reference more recent dates?
2. Specificity: Does RAG provide more specific numbers?
3. Context: Does RAG show awareness of live data vs historical?
4. Accuracy: Are the answers factually correct based on current data?

Expected RAG Advantages:
- More recent data points (closer to current date)
- Explicit mention of "real-time" or "latest" data
- Better performance on temporal questions ("current trend", "recent changes")
- Lower obsolescence (answers stay relevant longer)

Expected Baseline Limitations:
- Data cutoff at preprocessing time
- No awareness of very recent market movements
- May reference outdated information
- Higher obsolescence rate over time
""")
    
    return {
        'ticker': ticker,
        'baseline_model': baseline_data.get('model'),
        'rag_model': rag_data.get('model'),
        'baseline_questions': len(baseline_results),
        'rag_questions': len(rag_results),
        'comparison_timestamp': datetime.now().isoformat()
    }


def analyze_obsolescence(results_dir, ticker):
    """
    Analyze how answers change over time (obsolescence analysis)
    Compares multiple evaluation runs to measure decay
    """
    print(f"\n=== Obsolescence Analysis: {ticker} ===\n")
    
    # Find all result files for this ticker
    baseline_files = sorted(glob.glob(f"{results_dir}/{ticker}_baseline_results_*.json"))
    rag_files = sorted(glob.glob(f"{results_dir}/{ticker}_rag_results_*.json"))
    
    print(f"Found {len(baseline_files)} baseline evaluations")
    print(f"Found {len(rag_files)} RAG evaluations")
    
    if len(baseline_files) < 2 and len(rag_files) < 2:
        print("\nNeed at least 2 evaluation runs to analyze obsolescence.")
        print("Run evaluations at different times and compare how answers change.")
        return
    
    # Analyze baseline obsolescence
    if len(baseline_files) >= 2:
        print("\n--- Baseline System Obsolescence ---")
        first = load_evaluation_results(baseline_files[0])
        last = load_evaluation_results(baseline_files[-1])
        
        print(f"First evaluation: {first['timestamp']}")
        print(f"Last evaluation: {last['timestamp']}")
        print(f"Time span: {len(baseline_files)} evaluations")
        
        # Compare same questions across time
        if len(first['results']) > 0 and len(last['results']) > 0:
            sample_q = first['results'][0]['question']
            first_answer = first['results'][0]['answer']
            last_answer = last['results'][0]['answer']
            
            print(f"\nExample: '{sample_q}'")
            print(f"First answer: {first_answer[:100]}...")
            print(f"Last answer: {last_answer[:100]}...")
            
            if first_answer == last_answer:
                print("→ Answers identical (expected for baseline with static data)")
            else:
                print("→ Answers differ (unexpected for baseline)")
    
    # Analyze RAG obsolescence
    if len(rag_files) >= 2:
        print("\n--- RAG System Obsolescence ---")
        first = load_evaluation_results(rag_files[0])
        last = load_evaluation_results(rag_files[-1])
        
        print(f"First evaluation: {first['timestamp']}")
        print(f"Last evaluation: {last['timestamp']}")
        print(f"Time span: {len(rag_files)} evaluations")
        
        if len(first['results']) > 0 and len(last['results']) > 0:
            sample_q = first['results'][0]['question']
            first_answer = first['results'][0]['answer']
            last_answer = last['results'][0]['answer']
            
            print(f"\nExample: '{sample_q}'")
            print(f"First answer: {first_answer[:100]}...")
            print(f"Last answer: {last_answer[:100]}...")
            
            if first_answer != last_answer:
                print("→ Answers differ (expected - RAG updates with fresh data)")
            else:
                print("→ Answers identical (may indicate same market conditions)")


def generate_comparison_report(baseline_file, rag_file, output_file=None):
    """Generate a detailed comparison report and save to file"""
    comparison = compare_systems(baseline_file, rag_file)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✓ Comparison report saved to: {output_file}")
    
    return comparison


def main():
    if len(sys.argv) < 2:
        print("""
Usage:
  python compare_systems.py <baseline_file> <rag_file>
      Compare two specific evaluation files
  
  python compare_systems.py --obsolescence TICKER
      Analyze obsolescence over multiple evaluation runs
  
  python compare_systems.py --latest TICKER
      Compare most recent baseline vs RAG evaluation for a ticker

Examples:
  python compare_systems.py baseline_results.json rag_results.json
  python compare_systems.py --latest AAPL
  python compare_systems.py --obsolescence AAPL
""")
        sys.exit(1)
    
    if sys.argv[1] == "--obsolescence":
        if len(sys.argv) < 3:
            print("Please specify ticker: python compare_systems.py --obsolescence AAPL")
            sys.exit(1)
        analyze_obsolescence("evaluation_results", sys.argv[2].upper())
    
    elif sys.argv[1] == "--latest":
        if len(sys.argv) < 3:
            print("Please specify ticker: python compare_systems.py --latest AAPL")
            sys.exit(1)
        
        ticker = sys.argv[2].upper()
        
        # Find most recent files
        baseline_files = sorted(glob.glob(f"evaluation_results/{ticker}_baseline_results_*.json"))
        rag_files = sorted(glob.glob(f"evaluation_results/{ticker}_rag_results_*.json"))
        
        if not baseline_files:
            print(f"No baseline results found for {ticker}")
            sys.exit(1)
        if not rag_files:
            print(f"No RAG results found for {ticker}")
            sys.exit(1)
        
        baseline_file = baseline_files[-1]
        rag_file = rag_files[-1]
        
        print(f"Comparing:")
        print(f"  Baseline: {baseline_file}")
        print(f"  RAG: {rag_file}")
        
        output_file = f"evaluation_results/{ticker}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generate_comparison_report(baseline_file, rag_file, output_file)
    
    else:
        # Direct file comparison
        if len(sys.argv) < 3:
            print("Usage: python compare_systems.py <baseline_file> <rag_file>")
            sys.exit(1)
        
        baseline_file = sys.argv[1]
        rag_file = sys.argv[2]
        
        if not os.path.exists(baseline_file):
            print(f"Baseline file not found: {baseline_file}")
            sys.exit(1)
        if not os.path.exists(rag_file):
            print(f"RAG file not found: {rag_file}")
            sys.exit(1)
        
        generate_comparison_report(baseline_file, rag_file)


if __name__ == "__main__":
    main()