import json
import pandas as pd
import re
from datetime import datetime
import os

def extract_numbers(text):
    """Extract all numbers from text"""
    # Find all numbers including decimals and with commas
    numbers = re.findall(r'\$?(\d+(?:,\d+)?(?:\.\d+)?)', text)
    # Remove commas and convert to float
    return [float(n.replace(',', '')) for n in numbers]


def load_ground_truth(ticker):
    """Calculate ground truth answers from the actual data"""
    try:
        df = pd.read_csv(f"processed_data/{ticker}_processed.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        ground_truth = {}
        
        # Most recent day
        recent = df.iloc[-1]
        ground_truth['most_recent_close'] = recent['Close']
        ground_truth['most_recent_volume'] = recent['Volume']
        ground_truth['most_recent_high'] = recent['High']
        ground_truth['most_recent_low'] = recent['Low']
        ground_truth['most_recent_open'] = recent['Open']
        
        # Last 30 days
        last_30 = df.tail(30)
        ground_truth['highest_30d'] = last_30['High'].max()
        ground_truth['lowest_30d'] = last_30['Low'].min()
        
        # Last 10 days
        last_10 = df.tail(10)
        ground_truth['avg_volume_10d'] = last_10['Volume'].mean()
        
        # Price range on most recent day
        ground_truth['recent_range'] = recent['High'] - recent['Low']
        
        # Last 7 days
        last_7 = df.tail(7)
        first_of_7 = last_7.iloc[0]['Close']
        last_of_7 = last_7.iloc[-1]['Close']
        ground_truth['week_direction'] = 'increased' if last_of_7 > first_of_7 else 'decreased'
        ground_truth['week_change_pct'] = ((last_of_7 - first_of_7) / first_of_7) * 100
        
        # Comparison to 30 days ago
        if len(df) >= 30:
            thirty_days_ago = df.iloc[-30]['Close']
            ground_truth['vs_30d_ago'] = 'higher' if recent['Close'] > thirty_days_ago else 'lower'
        
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth for {ticker}: {e}")
        return None


def check_accuracy(question_id, answer, ground_truth):
    """
    Check if answer is accurate against ground truth.
    Returns (is_correct, confidence_level)
    """
    answer_lower = answer.lower()
    
    # Extract numbers from the answer
    answer_numbers = extract_numbers(answer)
    
    # Check based on question type
    if 'most recent' in question_id.lower() and 'close' in question_id.lower():
        expected = ground_truth.get('most_recent_close')
        if expected and answer_numbers:
            # Allow 1% tolerance for rounding
            for num in answer_numbers:
                if abs(num - expected) / expected < 0.01:
                    return True, 'high'
        return False, 'low'
    
    elif 'highest' in question_id.lower() and '30' in question_id.lower():
        expected = ground_truth.get('highest_30d')
        if expected and answer_numbers:
            for num in answer_numbers:
                if abs(num - expected) / expected < 0.01:
                    return True, 'high'
        return False, 'low'
    
    elif 'lowest' in question_id.lower() and '30' in question_id.lower():
        expected = ground_truth.get('lowest_30d')
        if expected and answer_numbers:
            for num in answer_numbers:
                if abs(num - expected) / expected < 0.01:
                    return True, 'high'
        return False, 'low'
    
    elif 'average' in question_id.lower() and 'volume' in question_id.lower():
        expected = ground_truth.get('avg_volume_10d')
        if expected and answer_numbers:
            for num in answer_numbers:
                # Volume can have wider tolerance
                if abs(num - expected) / expected < 0.05:
                    return True, 'high'
        return False, 'low'
    
    elif 'range' in question_id.lower():
        expected = ground_truth.get('recent_range')
        if expected and answer_numbers:
            for num in answer_numbers:
                if abs(num - expected) / expected < 0.02:
                    return True, 'high'
        return False, 'low'
    
    elif 'increase' in question_id.lower() or 'decrease' in question_id.lower():
        expected = ground_truth.get('week_direction')
        if expected:
            if expected in answer_lower:
                return True, 'high'
        return False, 'medium'
    
    elif 'higher or lower' in question_id.lower():
        if '30' in question_id.lower():
            expected = ground_truth.get('vs_30d_ago')
            if expected and expected in answer_lower:
                return True, 'high'
        return False, 'medium'
    
    # Default: can't automatically verify
    return None, 'unknown'


def detect_hallucinations(answer, df):
    """
    Detect potential hallucinations by checking if numbers in answer
    exist anywhere in the dataset
    """
    answer_numbers = extract_numbers(answer)
    if not answer_numbers:
        return False, []
    
    # Get all numbers from dataset
    dataset_values = set()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            dataset_values.update(df[col].round(2).values)
    
    hallucinated = []
    for num in answer_numbers:
        # Check if this number (or close to it) exists in dataset
        found = False
        for val in dataset_values:
            if abs(num - val) / max(val, 1) < 0.01:  # 1% tolerance
                found = True
                break
        
        if not found and num > 1:  # Ignore very small numbers (might be percentages)
            hallucinated.append(num)
    
    return len(hallucinated) > 0, hallucinated


def evaluate_results(results_file):
    """Evaluate a results file and generate a report"""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    ticker = data['ticker']
    results = data['results']
    
    # Load ground truth
    ground_truth = load_ground_truth(ticker)
    if ground_truth is None:
        print("Could not load ground truth data")
        return
    
    # Load actual dataset for hallucination detection
    df = pd.read_csv(f"processed_data/{ticker}_processed.csv")
    
    # Evaluate each result
    evaluation = {
        'ticker': ticker,
        'model': data['model'],
        'timestamp': data['timestamp'],
        'total_questions': len(results),
        'questions': []
    }
    
    correct_count = 0
    verifiable_count = 0
    hallucination_count = 0
    
    for result in results:
        question = result['question']
        answer = result['answer']
        
        # Check accuracy
        is_correct, confidence = check_accuracy(question, answer, ground_truth)
        
        # Check for hallucinations
        has_hallucination, hallucinated_nums = detect_hallucinations(answer, df)
        
        question_eval = {
            'question': question,
            'answer': answer,
            'is_correct': is_correct,
            'confidence': confidence,
            'has_hallucination': has_hallucination,
            'hallucinated_values': hallucinated_nums
        }
        
        evaluation['questions'].append(question_eval)
        
        if is_correct is not None:
            verifiable_count += 1
            if is_correct:
                correct_count += 1
        
        if has_hallucination:
            hallucination_count += 1
    
    # Calculate metrics
    evaluation['metrics'] = {
        'accuracy': correct_count / verifiable_count if verifiable_count > 0 else 0,
        'verifiable_questions': verifiable_count,
        'correct_answers': correct_count,
        'hallucination_rate': hallucination_count / len(results),
        'hallucinated_responses': hallucination_count
    }
    
    # Save evaluation
    eval_file = results_file.replace('results', 'evaluation')
    with open(eval_file, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print(f"EVALUATION REPORT: {ticker}")
    print("="*60)
    print(f"Model: {data['model']}")
    print(f"Total Questions: {len(results)}")
    print(f"Verifiable Questions: {verifiable_count}")
    print(f"\nAccuracy: {evaluation['metrics']['accuracy']:.1%}")
    print(f"Correct Answers: {correct_count}/{verifiable_count}")
    print(f"Hallucination Rate: {evaluation['metrics']['hallucination_rate']:.1%}")
    print(f"Hallucinated Responses: {hallucination_count}/{len(results)}")
    print("\n" + "="*60)
    
    # Print details
    print("\nDETAILED RESULTS:\n")
    for i, q in enumerate(evaluation['questions'], 1):
        status = "✓" if q['is_correct'] else "✗" if q['is_correct'] is False else "?"
        halluc = " [⚠ HALLUCINATION]" if q['has_hallucination'] else ""
        
        print(f"{i}. {status} {q['question']}{halluc}")
        if q['has_hallucination']:
            print(f"   Suspicious values: {q['hallucinated_values']}")
        print()
    
    print(f"✓ Evaluation saved to: {eval_file}\n")
    
    return evaluation


def compare_models(eval_files):
    """Compare evaluation results from multiple models"""
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60 + "\n")
    
    results = []
    for file in eval_files:
        with open(file, 'r') as f:
            data = json.load(f)
            results.append({
                'model': data['model'],
                'ticker': data['ticker'],
                'accuracy': data['metrics']['accuracy'],
                'hallucination_rate': data['metrics']['hallucination_rate']
            })
    
    # Print comparison table
    print(f"{'Model':<20} {'Ticker':<10} {'Accuracy':<12} {'Hallucination Rate':<20}")
    print("-" * 62)
    for r in results:
        print(f"{r['model']:<20} {r['ticker']:<10} {r['accuracy']:<12.1%} {r['hallucination_rate']:<20.1%}")
    
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <results_file.json>")
        print("   or: python evaluate.py --compare <file1> <file2> ...")
        sys.exit(1)
    
    if sys.argv[1] == "--compare":
        compare_models(sys.argv[2:])
    else:
        evaluate_results(sys.argv[1])