"""
Test baseline system with stale data to demonstrate obsolescence problem
Run this in Baseline/ folder
"""

import json
from datetime import datetime
import sys
import os

# You'll need to adjust this import based on your baseline file structure
# If your baseline is in baseline_qa.py with function ask_question():
try:
    from baseline_qa import ask_question
except:
    print(" Could not import baseline system")
    print("Please adjust the import statement to match your baseline file")
    sys.exit(1)

def test_baseline_obsolescence():
    """
    Run baseline on questions to show it gives confident answers 
    despite data being stale
    """
    
    ticker = "AAPL"
    questions = [
        "What is the current stock price?",
        "What was the highest price in the last 30 days?",
        "What is the current trend?",
        "What was the trading volume on the most recent day?",
        "How does the current price compare to last week?"
    ]
    
    results = []
    
    print("="*70)
    print("BASELINE OBSOLESCENCE TEST")
    print("Testing with data that is 2.7 days old (weekend)")
    print("="*70)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/5] {question}")
        
        # Run baseline
        try:
            answer = ask_question(ticker=ticker, question=question)
        except Exception as e:
            print(f"Error running baseline: {e}")
            answer = "ERROR"
        
        # Analyze for obsolescence indicators
        has_temporal_words = any(word in question.lower() for word in 
                                ['current', 'now', 'today', 'latest', 'recent'])
        
        answer_claims_current = any(word in answer.lower() for word in 
                                    ['current', 'now', 'today', 'latest'])
        
        has_data_age_warning = 'days old' in answer.lower() or 'stale' in answer.lower()
        has_confidence_score = 'confidence' in answer.lower()
        
        result = {
            'question': question,
            'answer': answer,
            'asks_about_current': has_temporal_words,
            'claims_current': answer_claims_current,
            'warns_about_age': has_data_age_warning,
            'shows_confidence': has_confidence_score,
            'obsolescence_problem': has_temporal_words and answer_claims_current and not has_data_age_warning
        }
        
        results.append(result)
        
        print(f"Answer: {answer[:150]}...")
        print(f" Obsolescence problem: {result['obsolescence_problem']}")
    
    # Summary
    obsolescence_count = sum(1 for r in results if r['obsolescence_problem'])
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Questions asking about 'current' data: {sum(1 for r in results if r['asks_about_current'])}/5")
    print(f"Answers claiming 'current' data: {sum(1 for r in results if r['claims_current'])}/5")
    print(f"Answers with data age warnings: {sum(1 for r in results if r['warns_about_age'])}/5")
    print(f"Answers with confidence scores: {sum(1 for r in results if r['shows_confidence'])}/5")
    print(f"\n❌ OBSOLESCENCE PROBLEMS: {obsolescence_count}/5")
    print("   (Claims current data without warning it's 2.7 days old)")
    
    # Save results
    os.makedirs('evaluation_results', exist_ok=True)
    output_file = f'evaluation_results/baseline_obsolescence_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'ticker': ticker,
            'system': 'BASELINE',
            'timestamp': datetime.now().isoformat(),
            'data_age_days': 2.7,
            'results': results,
            'summary': {
                'total_questions': len(results),
                'obsolescence_problems': obsolescence_count,
                'obsolescence_rate': obsolescence_count / len(results),
                'asks_about_current': sum(1 for r in results if r['asks_about_current']),
                'claims_current_without_warning': sum(1 for r in results if r['claims_current'] and not r['warns_about_age'])
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    return results

if __name__ == "__main__":
    test_baseline_obsolescence()