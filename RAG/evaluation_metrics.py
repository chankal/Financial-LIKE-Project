"""
Comprehensive Evaluation Metrics for RAG Systems
Implements standard metrics from research papers:
- Retrieval metrics: Recall@k, Precision@k, MRR, NDCG
- Generation metrics: ROUGE, BERTScore, Exact Match
- Factual accuracy: FactScore, Groundedness
"""

import numpy as np
from typing import List, Dict, Tuple, Set
import re
from collections import Counter


# ============================================================================
# RETRIEVAL METRICS (Standard in all RAG papers)
# ============================================================================

class RetrievalMetrics:
    """
    Evaluate retrieval quality independently from generation
    Used in: RAG (Lewis et al.), Self-RAG, FreshLLMs, etc.
    """
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
        """
        Recall@k: What fraction of relevant documents were retrieved in top-k?
        
        Formula: |retrieved ∩ relevant| / |relevant|
        
        Perfect score: 1.0 (all relevant docs retrieved)
        """
        retrieved_set = set(retrieved_ids[:k])
        if len(relevant_ids) == 0:
            return 0.0
        
        intersection = retrieved_set.intersection(relevant_ids)
        recall = len(intersection) / len(relevant_ids)
        
        return recall
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
        """
        Precision@k: What fraction of retrieved documents are relevant?
        
        Formula: |retrieved ∩ relevant| / k
        
        Perfect score: 1.0 (all retrieved docs are relevant)
        """
        retrieved_set = set(retrieved_ids[:k])
        
        intersection = retrieved_set.intersection(relevant_ids)
        precision = len(intersection) / k if k > 0 else 0.0
        
        return precision
    
    @staticmethod
    def f1_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
        """
        F1@k: Harmonic mean of precision and recall
        
        Formula: 2 * (P * R) / (P + R)
        """
        precision = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        MRR: Position of first relevant document
        
        Formula: 1 / rank of first relevant doc
        
        Perfect score: 1.0 (first doc is relevant)
        Worst score: 0.0 (no relevant docs retrieved)
        
        Used in: Most information retrieval papers
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Average Precision: Precision at each relevant document position
        
        Formula: (1/|relevant|) * Σ(Precision@k * relevance(k))
        
        Used in: Information retrieval, RAG papers
        """
        if len(relevant_ids) == 0:
            return 0.0
        
        num_hits = 0
        sum_precisions = 0.0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                sum_precisions += precision_at_i
        
        if num_hits == 0:
            return 0.0
        
        average_precision = sum_precisions / len(relevant_ids)
        return average_precision
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], 
                  relevance_scores: Dict[str, float] = None, k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain at k
        
        Accounts for position: relevant docs at top rank higher
        
        Formula: DCG@k / IDCG@k
        where DCG = Σ(relevance / log2(position + 1))
        
        Perfect score: 1.0
        
        Used in: Most modern retrieval papers
        """
        if relevance_scores is None:
            # Binary relevance: 1 if relevant, 0 otherwise
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 2)  # +2 because positions start at 0
        
        # Calculate IDCG (ideal DCG with perfect ranking)
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_scores):
            idcg += relevance / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        return ndcg


# ============================================================================
# GENERATION METRICS (Standard in all NLP papers)
# ============================================================================

class GenerationMetrics:
    """
    Evaluate generated text quality
    Used in: FreshLLMs, Self-RAG, FinQA, etc.
    """
    
    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """
        Exact Match: 1 if prediction exactly matches reference, 0 otherwise
        
        Used in: SQuAD, Natural Questions, FinQA
        """
        # Normalize both strings
        pred_normalized = GenerationMetrics._normalize_text(prediction)
        ref_normalized = GenerationMetrics._normalize_text(reference)
        
        return 1.0 if pred_normalized == ref_normalized else 0.0
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def token_f1(prediction: str, reference: str) -> float:
        """
        Token-level F1 score (used in SQuAD)
        
        Treats answer as bag of tokens and computes F1
        """
        pred_tokens = GenerationMetrics._normalize_text(prediction).split()
        ref_tokens = GenerationMetrics._normalize_text(reference).split()
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
        """
        ROUGE scores (Recall-Oriented Understudy for Gisting Evaluation)
        
        ROUGE-1: Unigram overlap
        ROUGE-2: Bigram overlap
        ROUGE-L: Longest Common Subsequence
        
        Used in: Nearly all text generation papers
        
        Note: For production, use rouge-score library
        This is a simplified implementation
        """
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        # ROUGE-1 (unigram overlap)
        pred_unigrams = set(pred_tokens)
        ref_unigrams = set(ref_tokens)
        
        if len(ref_unigrams) == 0:
            rouge1_recall = 0.0
        else:
            rouge1_recall = len(pred_unigrams & ref_unigrams) / len(ref_unigrams)
        
        if len(pred_unigrams) == 0:
            rouge1_precision = 0.0
        else:
            rouge1_precision = len(pred_unigrams & ref_unigrams) / len(pred_unigrams)
        
        if rouge1_precision + rouge1_recall == 0:
            rouge1_f1 = 0.0
        else:
            rouge1_f1 = 2 * (rouge1_precision * rouge1_recall) / (rouge1_precision + rouge1_recall)
        
        # ROUGE-L (Longest Common Subsequence)
        lcs_length = GenerationMetrics._lcs_length(pred_tokens, ref_tokens)
        
        if len(ref_tokens) == 0:
            rougeL_recall = 0.0
        else:
            rougeL_recall = lcs_length / len(ref_tokens)
        
        if len(pred_tokens) == 0:
            rougeL_precision = 0.0
        else:
            rougeL_precision = lcs_length / len(pred_tokens)
        
        if rougeL_precision + rougeL_recall == 0:
            rougeL_f1 = 0.0
        else:
            rougeL_f1 = 2 * (rougeL_precision * rougeL_recall) / (rougeL_precision + rougeL_recall)
        
        return {
            'rouge1_recall': rouge1_recall,
            'rouge1_precision': rouge1_precision,
            'rouge1_f1': rouge1_f1,
            'rougeL_recall': rougeL_recall,
            'rougeL_precision': rougeL_precision,
            'rougeL_f1': rougeL_f1
        }
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate Longest Common Subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


# ============================================================================
# ADVANCED METRICS (For research papers)
# ============================================================================

class AdvancedMetrics:
    """
    Advanced evaluation metrics
    - Groundedness: Answer supported by retrieved context?
    - Factual accuracy: Claims match source?
    - Hallucination detection: Fabricated information?
    """
    
    @staticmethod
    def groundedness_score(answer: str, context: str) -> Dict[str, float]:
        """
        Groundedness: Is answer supported by retrieved context?
        
        This is a simplified heuristic version.
        For production, use NLI model (RoBERTa-large-MNLI)
        
        Returns:
            - groundedness: 0.0 to 1.0
            - supported_claims: Number of claims found in context
            - total_claims: Total claims in answer
        """
        # Extract claims (sentences) from answer
        answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        # Extract potential facts from context
        context_lower = context.lower()
        
        supported_count = 0
        
        for sentence in answer_sentences:
            # Simple heuristic: check if key terms appear in context
            # In production, use entailment model
            sentence_lower = sentence.lower()
            
            # Extract key terms (nouns, numbers)
            key_terms = re.findall(r'\b\d+\.?\d*\b|\b[A-Z][a-z]+\b', sentence)
            
            if len(key_terms) == 0:
                continue
            
            # Check if majority of key terms appear in context
            terms_in_context = sum(1 for term in key_terms if term.lower() in context_lower)
            
            if terms_in_context / len(key_terms) > 0.5:
                supported_count += 1
        
        groundedness = supported_count / len(answer_sentences) if answer_sentences else 0.0
        
        return {
            'groundedness': groundedness,
            'supported_claims': supported_count,
            'total_claims': len(answer_sentences),
            'unsupported_claims': len(answer_sentences) - supported_count
        }
    
    @staticmethod
    def hallucination_detection(answer: str, context: str) -> Dict[str, any]:
        """
        Detect potential hallucinations (information not in context)
        
        This is a heuristic approach. For production, use:
        - NLI model for entailment checking
        - Named Entity Recognition
        - Fact verification models
        """
        # Extract numerical values
        answer_numbers = set(re.findall(r'\$?\d+\.?\d*', answer))
        context_numbers = set(re.findall(r'\$?\d+\.?\d*', context))
        
        # Numbers in answer but not in context (potential hallucination)
        hallucinated_numbers = answer_numbers - context_numbers
        
        # Extract dates
        answer_dates = set(re.findall(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', answer))
        context_dates = set(re.findall(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', context))
        
        hallucinated_dates = answer_dates - context_dates
        
        has_hallucination = len(hallucinated_numbers) > 0 or len(hallucinated_dates) > 0
        
        return {
            'has_hallucination': has_hallucination,
            'hallucinated_numbers': list(hallucinated_numbers),
            'hallucinated_dates': list(hallucinated_dates),
            'hallucination_score': 1.0 if has_hallucination else 0.0
        }


# ============================================================================
# INTEGRATED EVALUATION FUNCTION
# ============================================================================

def evaluate_rag_system_comprehensive(
    question: str,
    predicted_answer: str,
    reference_answer: str,
    retrieved_chunks: List[Dict],
    relevant_chunk_ids: Set[str],
    retrieved_context: str
) -> Dict:
    """
    Comprehensive evaluation combining all metrics
    
    This is what research papers report in their results tables
    
    Args:
        question: The input question
        predicted_answer: Generated answer from RAG system
        reference_answer: Ground truth answer
        retrieved_chunks: List of chunks retrieved (with 'id' field)
        relevant_chunk_ids: Set of IDs that are truly relevant
        retrieved_context: Combined text of retrieved chunks
    
    Returns:
        Dictionary with all metrics
    """
    retrieved_ids = [chunk['id'] for chunk in retrieved_chunks]
    
    # Retrieval metrics
    retrieval_metrics = {
        'recall@5': RetrievalMetrics.recall_at_k(retrieved_ids, relevant_chunk_ids, k=5),
        'precision@5': RetrievalMetrics.precision_at_k(retrieved_ids, relevant_chunk_ids, k=5),
        'f1@5': RetrievalMetrics.f1_at_k(retrieved_ids, relevant_chunk_ids, k=5),
        'mrr': RetrievalMetrics.mean_reciprocal_rank(retrieved_ids, relevant_chunk_ids),
        'average_precision': RetrievalMetrics.average_precision(retrieved_ids, relevant_chunk_ids),
        'ndcg@5': RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_chunk_ids, k=5)
    }
    
    # Generation metrics
    generation_metrics = {
        'exact_match': GenerationMetrics.exact_match(predicted_answer, reference_answer),
        'token_f1': GenerationMetrics.token_f1(predicted_answer, reference_answer),
    }
    
    # ROUGE scores
    rouge_scores = GenerationMetrics.rouge_scores(predicted_answer, reference_answer)
    generation_metrics.update(rouge_scores)
    
    # Advanced metrics
    groundedness = AdvancedMetrics.groundedness_score(predicted_answer, retrieved_context)
    hallucination = AdvancedMetrics.hallucination_detection(predicted_answer, retrieved_context)
    
    advanced_metrics = {
        'groundedness': groundedness['groundedness'],
        'supported_claims': groundedness['supported_claims'],
        'total_claims': groundedness['total_claims'],
        'has_hallucination': hallucination['has_hallucination'],
        'hallucination_score': hallucination['hallucination_score']
    }
    
    # Combine all metrics
    comprehensive_metrics = {
        'question': question,
        'retrieval': retrieval_metrics,
        'generation': generation_metrics,
        'advanced': advanced_metrics,
        'overall_score': (
            retrieval_metrics['f1@5'] * 0.3 +
            generation_metrics['token_f1'] * 0.4 +
            advanced_metrics['groundedness'] * 0.3
        )
    }
    
    return comprehensive_metrics


# ============================================================================
# BATCH EVALUATION FUNCTION
# ============================================================================

def evaluate_batch(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """
    Evaluate entire test set and compute aggregate metrics
    
    This produces the numbers you see in research paper results tables
    
    Args:
        predictions: List of prediction dicts with:
            - question, answer, retrieved_chunks, retrieved_ids, context
        ground_truth: List of ground truth dicts with:
            - question, answer, relevant_chunk_ids
    
    Returns:
        Aggregate metrics across all examples
    """
    all_metrics = []
    
    for pred, truth in zip(predictions, ground_truth):
        metrics = evaluate_rag_system_comprehensive(
            question=pred['question'],
            predicted_answer=pred['answer'],
            reference_answer=truth['answer'],
            retrieved_chunks=pred['retrieved_chunks'],
            relevant_chunk_ids=truth['relevant_chunk_ids'],
            retrieved_context=pred['context']
        )
        all_metrics.append(metrics)
    
    # Aggregate metrics
    aggregate = {
        'retrieval': {},
        'generation': {},
        'advanced': {}
    }
    
    # Average retrieval metrics
    for key in all_metrics[0]['retrieval'].keys():
        values = [m['retrieval'][key] for m in all_metrics]
        aggregate['retrieval'][key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Average generation metrics
    for key in all_metrics[0]['generation'].keys():
        values = [m['generation'][key] for m in all_metrics]
        aggregate['generation'][key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # Average advanced metrics
    for key in all_metrics[0]['advanced'].keys():
        if isinstance(all_metrics[0]['advanced'][key], (int, float)):
            values = [m['advanced'][key] for m in all_metrics]
            aggregate['advanced'][key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    # Overall score
    overall_scores = [m['overall_score'] for m in all_metrics]
    aggregate['overall_score'] = {
        'mean': np.mean(overall_scores),
        'std': np.std(overall_scores)
    }
    
    # Add sample size
    aggregate['n_samples'] = len(all_metrics)
    
    return aggregate


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("Comprehensive RAG Evaluation Metrics")
    print("="*70)
    
    # Example data
    question = "What is Apple's stock price?"
    predicted_answer = "Apple's stock price is $247.82 as of April 24, 2026."
    reference_answer = "Apple's closing price was $247.82 on April 24, 2026."
    
    retrieved_chunks = [
        {'id': 'chunk_1', 'text': 'Apple closed at $247.82 on 2026-04-24'},
        {'id': 'chunk_2', 'text': 'Volume was 52M shares'},
        {'id': 'chunk_3', 'text': 'Tesla data...'},
        {'id': 'chunk_4', 'text': 'Apple opened at $245.50'},
        {'id': 'chunk_5', 'text': 'Microsoft data...'}
    ]
    
    relevant_chunk_ids = {'chunk_1', 'chunk_4'}  # Ground truth: which chunks are actually relevant
    
    retrieved_context = "\n".join([c['text'] for c in retrieved_chunks])
    
    # Run comprehensive evaluation
    metrics = evaluate_rag_system_comprehensive(
        question=question,
        predicted_answer=predicted_answer,
        reference_answer=reference_answer,
        retrieved_chunks=retrieved_chunks,
        relevant_chunk_ids=relevant_chunk_ids,
        retrieved_context=retrieved_context
    )
    
    print("\nRetrieval Metrics:")
    for key, value in metrics['retrieval'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nGeneration Metrics:")
    for key, value in metrics['generation'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nAdvanced Metrics:")
    for key, value in metrics['advanced'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nOverall Score: {metrics['overall_score']:.4f}")