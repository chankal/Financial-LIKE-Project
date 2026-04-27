"""
Statistical Significance Testing for RAG System Evaluation
REQUIRED in all research papers to prove improvements are not due to chance

Implements:
- Paired t-test
- Cohen's d (effect size)
- Bootstrap confidence intervals
- Permutation tests
- Multiple comparison corrections
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import warnings


class StatisticalSignificance:
    """
    Test if improvements are statistically significant
    Papers must report p-values to claim "better performance"
    """
    
    @staticmethod
    def paired_t_test(system_a_scores: List[float], 
                     system_b_scores: List[float],
                     alpha: float = 0.05) -> Dict:
        """
        Paired t-test: Are two systems significantly different?
        
        Use when: Same test questions answered by both systems
        
        Null hypothesis: Mean difference = 0
        
        Args:
            system_a_scores: Scores from system A (e.g., baseline)
            system_b_scores: Scores from system B (e.g., enhanced RAG)
            alpha: Significance level (typically 0.05)
        
        Returns:
            Dict with t-statistic, p-value, and interpretation
        """
        if len(system_a_scores) != len(system_b_scores):
            raise ValueError("Paired test requires same number of scores")
        
        # Perform paired t-test
        t_statistic, p_value = stats.ttest_rel(system_b_scores, system_a_scores)
        
        # Determine significance
        is_significant = p_value < alpha
        
        # Calculate mean difference
        mean_diff = np.mean(system_b_scores) - np.mean(system_a_scores)
        
        # Interpretation
        if is_significant:
            if mean_diff > 0:
                interpretation = f"System B is significantly better (p={p_value:.4f} < {alpha})"
            else:
                interpretation = f"System B is significantly worse (p={p_value:.4f} < {alpha})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f} >= {alpha})"
        
        return {
            't_statistic': t_statistic,
            'p_value': p_value,
            'degrees_of_freedom': len(system_a_scores) - 1,
            'mean_difference': mean_diff,
            'is_significant': is_significant,
            'significance_level': alpha,
            'interpretation': interpretation,
            'system_a_mean': np.mean(system_a_scores),
            'system_b_mean': np.mean(system_b_scores)
        }
    
    @staticmethod
    def cohens_d(system_a_scores: List[float], 
                 system_b_scores: List[float]) -> Dict:
        """
        Cohen's d: Effect size (how big is the difference?)
        
        Effect size interpretation:
        - Small: 0.2
        - Medium: 0.5
        - Large: 0.8+
        
        This tells you if the difference is PRACTICALLY significant
        (p-value tells you if it's STATISTICALLY significant)
        
        Args:
            system_a_scores: Baseline scores
            system_b_scores: Enhanced system scores
        
        Returns:
            Dict with Cohen's d and interpretation
        """
        mean_a = np.mean(system_a_scores)
        mean_b = np.mean(system_b_scores)
        
        std_a = np.std(system_a_scores, ddof=1)
        std_b = np.std(system_b_scores, ddof=1)
        
        # Pooled standard deviation
        n_a = len(system_a_scores)
        n_b = len(system_b_scores)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        
        # Cohen's d
        d = (mean_b - mean_a) / pooled_std
        
        # Interpret effect size
        abs_d = abs(d)
        if abs_d < 0.2:
            effect_size = "negligible"
        elif abs_d < 0.5:
            effect_size = "small"
        elif abs_d < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        return {
            'cohens_d': d,
            'effect_size': effect_size,
            'mean_difference': mean_b - mean_a,
            'pooled_std': pooled_std,
            'interpretation': f"Cohen's d = {d:.3f} ({effect_size} effect)"
        }
    
    @staticmethod
    def bootstrap_confidence_interval(system_a_scores: List[float],
                                     system_b_scores: List[float],
                                     n_iterations: int = 10000,
                                     confidence_level: float = 0.95) -> Dict:
        """
        Bootstrap confidence interval for mean difference
        
        Non-parametric method: doesn't assume normal distribution
        
        Args:
            system_a_scores: Baseline scores
            system_b_scores: Enhanced scores
            n_iterations: Number of bootstrap samples
            confidence_level: CI level (0.95 = 95% CI)
        
        Returns:
            Dict with confidence interval and interpretation
        """
        differences = []
        
        for _ in range(n_iterations):
            # Resample with replacement
            sample_a = np.random.choice(system_a_scores, size=len(system_a_scores), replace=True)
            sample_b = np.random.choice(system_b_scores, size=len(system_b_scores), replace=True)
            
            # Calculate difference of means
            diff = np.mean(sample_b) - np.mean(sample_a)
            differences.append(diff)
        
        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(differences, lower_percentile)
        ci_upper = np.percentile(differences, upper_percentile)
        
        # Observed difference
        observed_diff = np.mean(system_b_scores) - np.mean(system_a_scores)
        
        # Check if CI excludes zero (significant)
        excludes_zero = not (ci_lower <= 0 <= ci_upper)
        
        return {
            'observed_difference': observed_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
            'excludes_zero': excludes_zero,
            'interpretation': f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]" + 
                            (" (significant)" if excludes_zero else " (not significant)")
        }
    
    @staticmethod
    def permutation_test(system_a_scores: List[float],
                        system_b_scores: List[float],
                        n_permutations: int = 10000) -> Dict:
        """
        Permutation test: Non-parametric significance test
        
        Null hypothesis: Labels don't matter (systems perform the same)
        
        More robust than t-test when assumptions are violated
        
        Args:
            system_a_scores: Baseline scores
            system_b_scores: Enhanced scores
            n_permutations: Number of random permutations
        
        Returns:
            Dict with p-value from permutation test
        """
        # Observed difference
        observed_diff = np.mean(system_b_scores) - np.mean(system_a_scores)
        
        # Combine all scores
        all_scores = np.concatenate([system_a_scores, system_b_scores])
        n_a = len(system_a_scores)
        
        # Count how many random permutations give difference >= observed
        extreme_count = 0
        
        for _ in range(n_permutations):
            # Randomly shuffle labels
            np.random.shuffle(all_scores)
            
            # Split into two groups
            perm_a = all_scores[:n_a]
            perm_b = all_scores[n_a:]
            
            # Calculate difference
            perm_diff = np.mean(perm_b) - np.mean(perm_a)
            
            # Check if as extreme as observed
            if abs(perm_diff) >= abs(observed_diff):
                extreme_count += 1
        
        # Calculate p-value
        p_value = extreme_count / n_permutations
        
        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'n_permutations': n_permutations,
            'is_significant': p_value < 0.05,
            'interpretation': f"Permutation test p-value: {p_value:.4f}"
        }
    
    @staticmethod
    def multiple_comparisons_correction(p_values: List[float], 
                                       method: str = 'bonferroni') -> Dict:
        """
        Correct p-values when testing multiple hypotheses
        
        Problem: Testing 5 systems gives ~5 chances to find p<0.05 by chance
        Solution: Adjust p-values to control family-wise error rate
        
        Methods:
        - bonferroni: Most conservative (p_adjusted = p * n_tests)
        - holm: Less conservative but still controls FWER
        - benjamini-hochberg: Controls false discovery rate (FDR)
        
        Args:
            p_values: List of p-values from multiple tests
            method: Correction method
        
        Returns:
            Corrected p-values and which are still significant
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            # Bonferroni correction
            corrected = p_values * n_tests
            corrected = np.minimum(corrected, 1.0)  # Cap at 1.0
            
        elif method == 'holm':
            # Holm-Bonferroni (less conservative)
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected = np.zeros_like(p_values)
            for i, p in enumerate(sorted_p):
                corrected[sorted_indices[i]] = min(p * (n_tests - i), 1.0)
        
        elif method == 'benjamini_hochberg':
            # Benjamini-Hochberg (FDR control)
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected = np.zeros_like(p_values)
            for i, p in enumerate(sorted_p):
                corrected[sorted_indices[i]] = min(p * n_tests / (i + 1), 1.0)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Check which are still significant after correction
        significant = corrected < 0.05
        
        return {
            'method': method,
            'original_p_values': p_values.tolist(),
            'corrected_p_values': corrected.tolist(),
            'significant_after_correction': significant.tolist(),
            'n_significant_before': int(np.sum(p_values < 0.05)),
            'n_significant_after': int(np.sum(significant))
        }


# ============================================================================
# COMPREHENSIVE STATISTICAL ANALYSIS
# ============================================================================

def comprehensive_statistical_analysis(baseline_scores: List[float],
                                      rag_scores: List[float],
                                      system_names: Tuple[str, str] = ('Baseline', 'Enhanced RAG')) -> Dict:
    """
    Run all statistical tests and return comprehensive report
    
    This is what you'd report in a research paper
    
    Args:
        baseline_scores: Scores from baseline system
        rag_scores: Scores from enhanced RAG system
        system_names: Names of the two systems
    
    Returns:
        Comprehensive statistical report
    """
    print("\n" + "="*70)
    print(f"STATISTICAL ANALYSIS: {system_names[0]} vs {system_names[1]}")
    print("="*70)
    
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(f"{system_names[0]:20s}: Mean={np.mean(baseline_scores):.4f}, SD={np.std(baseline_scores, ddof=1):.4f}")
    print(f"{system_names[1]:20s}: Mean={np.mean(rag_scores):.4f}, SD={np.std(rag_scores, ddof=1):.4f}")
    
    # 1. Paired t-test
    print("\n" + "-"*70)
    print("1. PAIRED T-TEST")
    print("-"*70)
    t_test = StatisticalSignificance.paired_t_test(baseline_scores, rag_scores)
    print(f"t-statistic: {t_test['t_statistic']:.4f}")
    print(f"p-value: {t_test['p_value']:.6f}")
    print(f"Interpretation: {t_test['interpretation']}")
    
    # 2. Effect size
    print("\n" + "-"*70)
    print("2. EFFECT SIZE (Cohen's d)")
    print("-"*70)
    effect_size = StatisticalSignificance.cohens_d(baseline_scores, rag_scores)
    print(f"Cohen's d: {effect_size['cohens_d']:.4f}")
    print(f"Effect size: {effect_size['effect_size']}")
    print(f"Interpretation: {effect_size['interpretation']}")
    
    # 3. Bootstrap CI
    print("\n" + "-"*70)
    print("3. BOOTSTRAP CONFIDENCE INTERVAL")
    print("-"*70)
    bootstrap = StatisticalSignificance.bootstrap_confidence_interval(baseline_scores, rag_scores)
    print(f"Observed difference: {bootstrap['observed_difference']:.4f}")
    print(f"95% CI: [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]")
    print(f"Excludes zero: {bootstrap['excludes_zero']} (significant)" if bootstrap['excludes_zero'] else "Excludes zero: False (not significant)")
    
    # 4. Permutation test
    print("\n" + "-"*70)
    print("4. PERMUTATION TEST (Non-parametric)")
    print("-"*70)
    permutation = StatisticalSignificance.permutation_test(baseline_scores, rag_scores, n_permutations=10000)
    print(f"p-value: {permutation['p_value']:.6f}")
    print(f"Significant: {permutation['is_significant']}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    improvement = np.mean(rag_scores) - np.mean(baseline_scores)
    improvement_pct = (improvement / np.mean(baseline_scores)) * 100
    
    print(f"\nImprovement: +{improvement:.4f} ({improvement_pct:.1f}%)")
    print(f"Statistical significance: p={t_test['p_value']:.6f} {'(SIGNIFICANT)' if t_test['is_significant'] else '(NOT SIGNIFICANT)'}")
    print(f"Effect size: {effect_size['effect_size']} (d={effect_size['cohens_d']:.3f})")
    print(f"95% Confidence interval: [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]")
    
    # Compile results
    return {
        'descriptive': {
            'baseline_mean': np.mean(baseline_scores),
            'baseline_std': np.std(baseline_scores, ddof=1),
            'rag_mean': np.mean(rag_scores),
            'rag_std': np.std(rag_scores, ddof=1),
            'improvement': improvement,
            'improvement_percentage': improvement_pct
        },
        't_test': t_test,
        'effect_size': effect_size,
        'bootstrap_ci': bootstrap,
        'permutation_test': permutation
    }


# ============================================================================
# REPORTING UTILITIES
# ============================================================================

def generate_paper_statistics_text(baseline_scores: List[float],
                                   rag_scores: List[float]) -> str:
    """
    Generate text for paper's results section
    
    Example output:
    "Enhanced RAG achieved significantly higher accuracy (M=0.915, SD=0.082) 
    compared to baseline (M=0.710, SD=0.095), t(29)=12.34, p<.001, d=2.31 (large effect)."
    """
    # Run analysis
    results = comprehensive_statistical_analysis(baseline_scores, rag_scores)
    
    baseline_mean = results['descriptive']['baseline_mean']
    baseline_std = results['descriptive']['baseline_std']
    rag_mean = results['descriptive']['rag_mean']
    rag_std = results['descriptive']['rag_std']
    
    t_stat = results['t_test']['t_statistic']
    df = results['t_test']['degrees_of_freedom']
    p_value = results['t_test']['p_value']
    
    cohens_d = results['effect_size']['cohens_d']
    effect_size = results['effect_size']['effect_size']
    
    # Format p-value
    if p_value < 0.001:
        p_str = "p<.001"
    else:
        p_str = f"p={p_value:.3f}"
    
    text = (f"Enhanced RAG achieved significantly higher accuracy "
            f"(M={rag_mean:.3f}, SD={rag_std:.3f}) compared to baseline "
            f"(M={baseline_mean:.3f}, SD={baseline_std:.3f}), "
            f"t({df})={t_stat:.2f}, {p_str}, d={cohens_d:.2f} ({effect_size} effect).")
    
    return text


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Comparing baseline vs enhanced RAG
    # These would be actual accuracy scores on 30 test questions
    
    np.random.seed(42)
    
    # Baseline: Mean ~0.71, some variance
    baseline_scores = np.random.beta(7, 3, 30) * 0.5 + 0.5  # Mean ~0.71
    
    # Enhanced RAG: Mean ~0.915, less variance
    rag_scores = np.random.beta(9, 1, 30) * 0.2 + 0.8  # Mean ~0.915
    
    # Run comprehensive analysis
    results = comprehensive_statistical_analysis(baseline_scores, rag_scores)
    
    # Generate paper text
    print("\n" + "="*70)
    print("TEXT FOR PAPER (Results Section):")
    print("="*70)
    paper_text = generate_paper_statistics_text(baseline_scores, rag_scores)
    print(f"\n{paper_text}\n")
    
    # Multiple comparisons example
    print("\n" + "="*70)
    print("MULTIPLE COMPARISONS CORRECTION")
    print("="*70)
    
    # Simulate comparing 5 different systems
    p_values = [0.001, 0.023, 0.045, 0.067, 0.089]
    
    correction = StatisticalSignificance.multiple_comparisons_correction(p_values, method='bonferroni')
    
    print(f"\nOriginal p-values: {correction['original_p_values']}")
    print(f"Corrected (Bonferroni): {correction['corrected_p_values']}")
    print(f"Significant before: {correction['n_significant_before']}/5")
    print(f"Significant after: {correction['n_significant_after']}/5")