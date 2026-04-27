"""
Error Analysis and Categorization
Systematically categorize and analyze errors to understand failure modes

Used in: FinQA, FreshLLMs, Self-RAG - all research papers do this
"""

import re
from typing import List, Dict, Tuple
from datetime import datetime
import json


class ErrorCategorizer:
    """
    Categorize errors into meaningful types
    Helps understand WHERE the system fails
    """
    
    @staticmethod
    def categorize_error(question: str, 
                        predicted_answer: str,
                        reference_answer: str,
                        retrieved_context: str) -> Dict:
        """
        Categorize what type of error occurred
        
        Error types:
        1. Temporal error: Wrong dates/times
        2. Factual error: Wrong numbers/facts
        3. Retrieval failure: Relevant info not retrieved
        4. Hallucination: Info not in context
        5. Calculation error: Wrong math
        6. Incomplete answer: Missing information
        """
        errors = []
        
        # Extract numbers and dates
        pred_numbers = set(re.findall(r'\$?\d+\.?\d*', predicted_answer))
        ref_numbers = set(re.findall(r'\$?\d+\.?\d*', reference_answer))
        context_numbers = set(re.findall(r'\$?\d+\.?\d*', retrieved_context))
        
        pred_dates = set(re.findall(r'\d{4}-\d{2}-\d{2}', predicted_answer))
        ref_dates = set(re.findall(r'\d{4}-\d{2}-\d{2}', reference_answer))
        context_dates = set(re.findall(r'\d{4}-\d{2}-\d{2}', retrieved_context))
        
        # Check for temporal errors
        if pred_dates != ref_dates:
            wrong_dates = pred_dates - ref_dates
            if wrong_dates:
                errors.append({
                    'type': 'TEMPORAL_ERROR',
                    'severity': 'HIGH',
                    'description': f'Wrong date(s): {wrong_dates}',
                    'expected': list(ref_dates),
                    'actual': list(pred_dates)
                })
        
        # Check for factual errors (wrong numbers)
        if pred_numbers != ref_numbers:
            wrong_numbers = pred_numbers - ref_numbers
            if wrong_numbers:
                errors.append({
                    'type': 'FACTUAL_ERROR',
                    'severity': 'HIGH',
                    'description': f'Wrong number(s): {wrong_numbers}',
                    'expected': list(ref_numbers),
                    'actual': list(pred_numbers)
                })
        
        # Check for hallucination (numbers not in context)
        hallucinated_numbers = pred_numbers - context_numbers
        if hallucinated_numbers and len(hallucinated_numbers) > 0:
            # Filter out small differences (rounding)
            significant_hallucinations = [
                n for n in hallucinated_numbers 
                if not any(abs(float(n) - float(c)) < 0.01 for c in context_numbers if c.replace('.', '').isdigit())
            ]
            
            if significant_hallucinations:
                errors.append({
                    'type': 'HALLUCINATION',
                    'severity': 'CRITICAL',
                    'description': f'Fabricated number(s) not in context: {significant_hallucinations}',
                    'context_numbers': list(context_numbers)
                })
        
        # Check for retrieval failure
        if ref_numbers and not (ref_numbers & context_numbers):
            errors.append({
                'type': 'RETRIEVAL_FAILURE',
                'severity': 'HIGH',
                'description': 'Ground truth information not in retrieved context',
                'missing_info': list(ref_numbers)
            })
        
        # Check for incomplete answer
        pred_length = len(predicted_answer.split())
        ref_length = len(reference_answer.split())
        
        if pred_length < ref_length * 0.5:  # Less than half the expected length
            errors.append({
                'type': 'INCOMPLETE_ANSWER',
                'severity': 'MEDIUM',
                'description': 'Answer significantly shorter than expected',
                'pred_length': pred_length,
                'ref_length': ref_length
            })
        
        # Determine primary error type
        if errors:
            primary_error = max(errors, key=lambda x: 
                {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}[x['severity']])
        else:
            primary_error = None
        
        return {
            'has_error': len(errors) > 0,
            'error_count': len(errors),
            'errors': errors,
            'primary_error_type': primary_error['type'] if primary_error else None,
            'primary_error_severity': primary_error['severity'] if primary_error else None
        }
    
    @staticmethod
    def analyze_error_patterns(error_analyses: List[Dict]) -> Dict:
        """
        Analyze patterns across all errors
        
        Returns distribution and examples for each error type
        """
        error_counts = {
            'TEMPORAL_ERROR': 0,
            'FACTUAL_ERROR': 0,
            'RETRIEVAL_FAILURE': 0,
            'HALLUCINATION': 0,
            'CALCULATION_ERROR': 0,
            'INCOMPLETE_ANSWER': 0
        }
        
        error_examples = {k: [] for k in error_counts.keys()}
        
        for analysis in error_analyses:
            if analysis['has_error']:
                for error in analysis['errors']:
                    error_type = error['type']
                    error_counts[error_type] += 1
                    
                    # Store first 3 examples of each type
                    if len(error_examples[error_type]) < 3:
                        error_examples[error_type].append({
                            'description': error['description'],
                            'severity': error['severity']
                        })
        
        total_errors = sum(error_counts.values())
        
        # Calculate percentages
        error_distribution = {
            error_type: {
                'count': count,
                'percentage': (count / total_errors * 100) if total_errors > 0 else 0,
                'examples': error_examples[error_type]
            }
            for error_type, count in error_counts.items()
        }
        
        return {
            'total_errors': total_errors,
            'total_questions': len(error_analyses),
            'error_rate': total_errors / len(error_analyses) if error_analyses else 0,
            'error_distribution': error_distribution
        }


class MultipleBaselines:
    """
    Compare against multiple baseline systems
    Research papers compare 3-5 different approaches
    """
    
    @staticmethod
    def zero_shot_llm(question: str, ticker: str, model: str = 'gpt-4') -> str:
        """
        Baseline 1: Zero-shot LLM (no retrieval, no examples)
        Just ask the LLM directly
        """
        from rag_qa_enhanced import call_llm
        
        prompt = f"""Answer this question about {ticker} stock:

Question: {question}

Answer based on your general knowledge:"""
        
        answer = call_llm(prompt)
        return answer
    
    @staticmethod
    def few_shot_llm(question: str, ticker: str, examples: List[Dict], model: str = 'gpt-4') -> str:
        """
        Baseline 2: Few-shot LLM (no retrieval, but with examples)
        """
        from rag_qa_enhanced import call_llm
        
        # Build prompt with examples
        prompt = f"""Answer questions about stock market data.

Here are some examples:

"""
        
        for ex in examples:
            prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
        
        prompt += f"Now answer this question about {ticker}:\n\nQ: {question}\nA:"
        
        answer = call_llm(prompt)
        return answer
    
    @staticmethod
    def bm25_retrieval(question: str, chunks: List[Dict], k: int = 5) -> List[Dict]:
        """
        Baseline 3: BM25 retrieval (traditional IR, no neural embeddings)
        
        BM25 is a bag-of-words retrieval algorithm
        Used as baseline in most RAG papers
        """
        from collections import Counter
        import math
        
        # Tokenize question
        question_tokens = question.lower().split()
        
        # Calculate BM25 scores
        k1 = 1.5  # Term frequency saturation parameter
        b = 0.75  # Length normalization parameter
        
        # Calculate average document length
        avg_doc_len = np.mean([len(chunk['text'].split()) for chunk in chunks])
        
        # Calculate document frequency for each term
        doc_freq = Counter()
        for chunk in chunks:
            unique_tokens = set(chunk['text'].lower().split())
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Score each document
        scores = []
        N = len(chunks)
        
        for chunk in chunks:
            doc_tokens = chunk['text'].lower().split()
            doc_len = len(doc_tokens)
            term_freq = Counter(doc_tokens)
            
            score = 0
            for term in question_tokens:
                if term in term_freq:
                    tf = term_freq[term]
                    df = doc_freq[term]
                    
                    # IDF component
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    
                    # TF component with saturation
                    tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                    
                    score += idf * tf_component
            
            scores.append((chunk, score))
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, score in scores[:k]]
        
        return top_chunks
    
    @staticmethod
    def run_all_baselines(question: str, ticker: str, chunks: List[Dict]) -> Dict:
        """
        Run all baseline comparisons
        """
        print(f"\nRunning multiple baselines for: {question}")
        
        results = {}
        
        # 1. Zero-shot
        print("  [1/3] Zero-shot LLM...")
        results['zero_shot'] = {
            'answer': MultipleBaselines.zero_shot_llm(question, ticker),
            'method': 'No retrieval, no examples'
        }
        
        # 2. Few-shot
        print("  [2/3] Few-shot LLM...")
        examples = [
            {'question': 'What is the stock price?', 'answer': 'Example: $250.00'},
            {'question': 'What was the volume?', 'answer': 'Example: 50M shares'}
        ]
        results['few_shot'] = {
            'answer': MultipleBaselines.few_shot_llm(question, ticker, examples),
            'method': 'No retrieval, with examples'
        }
        
        # 3. BM25 + LLM
        print("  [3/3] BM25 retrieval...")
        bm25_chunks = MultipleBaselines.bm25_retrieval(question, chunks, k=5)
        bm25_context = "\n\n".join([c['text'] for c in bm25_chunks])
        
        from rag_qa_enhanced import call_llm
        prompt = f"Context:\n{bm25_context}\n\nQuestion: {question}\n\nAnswer:"
        
        results['bm25_rag'] = {
            'answer': call_llm(prompt),
            'method': 'BM25 retrieval + LLM',
            'retrieved_chunks': len(bm25_chunks)
        }
        
        return results


def generate_error_analysis_report(predictions: List[Dict], 
                                   ground_truth: List[Dict],
                                   retrieved_contexts: List[str]) -> str:
    """
    Generate comprehensive error analysis report for paper
    
    Args:
        predictions: List of {question, answer}
        ground_truth: List of {question, answer}
        retrieved_contexts: List of retrieved context strings
    
    Returns:
        Markdown formatted error analysis report
    """
    # Categorize all errors
    error_analyses = []
    
    for pred, truth, context in zip(predictions, ground_truth, retrieved_contexts):
        analysis = ErrorCategorizer.categorize_error(
            question=pred['question'],
            predicted_answer=pred['answer'],
            reference_answer=truth['answer'],
            retrieved_context=context
        )
        analysis['question'] = pred['question']
        analysis['predicted'] = pred['answer']
        analysis['reference'] = truth['answer']
        error_analyses.append(analysis)
    
    # Analyze patterns
    patterns = ErrorCategorizer.analyze_error_patterns(error_analyses)
    
    # Generate report
    report = []
    report.append("# Error Analysis Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nTotal questions analyzed: {patterns['total_questions']}")
    report.append(f"Total errors found: {patterns['total_errors']}")
    report.append(f"Overall error rate: {patterns['error_rate']:.1%}")
    report.append("\n---\n")
    
    # Error distribution
    report.append("## Error Type Distribution\n")
    report.append("| Error Type | Count | Percentage |")
    report.append("|------------|-------|------------|")
    
    for error_type, data in sorted(patterns['error_distribution'].items(), 
                                   key=lambda x: x[1]['count'], reverse=True):
        if data['count'] > 0:
            report.append(f"| {error_type} | {data['count']} | {data['percentage']:.1f}% |")
    
    # Examples for each error type
    report.append("\n---\n")
    report.append("## Error Examples by Type\n")
    
    for error_type, data in patterns['error_distribution'].items():
        if data['examples']:
            report.append(f"\n### {error_type}\n")
            for i, example in enumerate(data['examples'], 1):
                report.append(f"{i}. **{example['severity']}**: {example['description']}")
    
    # Specific error instances
    report.append("\n---\n")
    report.append("## Detailed Error Instances\n")
    
    error_instances = [ea for ea in error_analyses if ea['has_error']][:5]  # First 5
    
    for i, instance in enumerate(error_instances, 1):
        report.append(f"\n### Error {i}\n")
        report.append(f"**Question:** {instance['question']}\n")
        report.append(f"**Predicted:** {instance['predicted'][:200]}...\n")
        report.append(f"**Reference:** {instance['reference'][:200]}...\n")
        report.append(f"**Error type:** {instance['primary_error_type']}\n")
        report.append(f"**Severity:** {instance['primary_error_severity']}\n")
        
        for error in instance['errors']:
            report.append(f"- {error['description']}")
    
    return "\n".join(report)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Error Analysis and Categorization")
    print("="*70)
    
    # Example error categorization
    question = "What was Apple's closing price on April 24?"
    predicted = "Apple closed at $250.00 on April 25, 2026."
    reference = "Apple closed at $247.82 on April 24, 2026."
    context = "Stock: AAPL, Date: 2026-04-24, Close: $247.82"
    
    analysis = ErrorCategorizer.categorize_error(question, predicted, reference, context)
    
    print("\nExample Error Analysis:")
    print(f"Has error: {analysis['has_error']}")
    print(f"Error count: {analysis['error_count']}")
    print(f"Primary error: {analysis['primary_error_type']}")
    
    for error in analysis['errors']:
        print(f"\n  Type: {error['type']}")
        print(f"  Severity: {error['severity']}")
        print(f"  Description: {error['description']}")
    
    # Example: Multiple baselines comparison
    print("\n" + "="*70)
    print("Multiple Baselines Comparison")
    print("="*70)
    
    # This would use actual chunks
    example_chunks = [
        {'text': 'AAPL closed at $247.82 on 2026-04-24', 'id': 'chunk1'},
        {'text': 'AAPL volume was 52M shares', 'id': 'chunk2'},
    ]
    
    baselines = MultipleBaselines.run_all_baselines(
        question="What was the closing price?",
        ticker="AAPL",
        chunks=example_chunks
    )
    
    print("\nBaseline Results:")
    for name, result in baselines.items():
        print(f"\n{name}:")
        print(f"  Method: {result['method']}")
        print(f"  Answer: {result['answer'][:100]}...")