"""
Phase 0 Validation Script
Proves the core concept works before building the browser extension.
Tests:
1. Basic bandit learning on synthetic data
2. Relevance guardrails working correctly
3. Carbon-CTR tradeoff analysis
4. Comparison against baselines
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.carbon.resolver import CarbonResolver
from src.bandit.linucb import LinUCB
from src.bandit.context import ContextBuilder
from src.reranker.engine import GreenReRanker
from src.data.simulator import SERPSimulator
from src.evaluation.metrics import evaluate_session, compute_carbon_ctr_tradeoff


# =============================================================================
# Configuration
# =============================================================================

N_TRAINING_QUERIES = 500
N_EVAL_QUERIES = 200
N_RESULTS_PER_PAGE = 10
BANDIT_ALPHA = 1.0
TAU = 0.05  # Relevance guardrail threshold
BOOST_LEVELS = [0, 1, 2, 3]
RANDOM_SEED = 42

# =============================================================================
# Reward Function
# =============================================================================

def reward_function(clicked: bool, carbon_score: float, position: int, boost_applied: int) -> float:
    """
    Composite reward: clicks matter most, but green clicks get bonus.
    
    Args:
        clicked: Whether the result was clicked
        carbon_score: Carbon score of the result (0=dirty, 1=green)
        position: Final position of the result
        boost_applied: How many positions this result was boosted
    
    Returns:
        Reward value
    """
    if not clicked:
        return 0.0
    
    # Base reward for click
    reward = 1.0
    
    # Green bonus: click on green site is worth more
    reward += 0.2 * carbon_score
    
    # Position penalty: clicks on lower results are worth slightly less
    reward -= 0.02 * position
    
    # Small penalty for high boosts (to prevent over-boosting)
    reward -= 0.01 * boost_applied
    
    return max(reward, 0.0)


# =============================================================================
# Baselines
# =============================================================================

class BaselineRankers:
    """Collection of baseline re-ranking strategies for comparison."""
    
    @staticmethod
    def no_rerank(results, relevance_scores, carbon_scores):
        """Keep original relevance ranking."""
        return list(range(len(results)))
    
    @staticmethod
    def green_only(results, relevance_scores, carbon_scores):
        """Rank purely by carbon score."""
        domains = [r.domain for r in results]
        carbon_list = [carbon_scores.get(d, 0.5) for d in domains]
        return np.argsort(carbon_list)[::-1]  # Highest carbon first
    
    @staticmethod
    def static_weighted(results, relevance_scores, carbon_scores, carbon_weight=0.3):
        """Weighted combination of relevance and carbon."""
        domains = [r.domain for r in results]
        carbon_list = [carbon_scores.get(d, 0.5) for d in domains]
        scores = [(1 - carbon_weight) * relevance_scores[i] + carbon_weight * carbon_list[i] 
                  for i in range(len(results))]
        return np.argsort(scores)[::-1]


# =============================================================================
# Main Experiment
# =============================================================================

def run_phase0_validation():
    """Run the complete Phase 0 validation experiment."""
    
    print("=" * 70)
    print("PHASE 0 VALIDATION: Green Search Brain")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training queries: {N_TRAINING_QUERIES}")
    print(f"Evaluation queries: {N_EVAL_QUERIES}")
    print(f"Bandit alpha: {BANDIT_ALPHA}")
    print(f"Relevance guardrail tau: {TAU}")
    print("-" * 70)
    
    # Initialize components
    print("\n[1/6] Initializing components...")
    np.random.seed(RANDOM_SEED)
    
    carbon_resolver = CarbonResolver()
    context_builder = ContextBuilder(config={'context_dim': 8})
    bandit = LinUCB(n_arms=len(BOOST_LEVELS), context_dim=8, alpha=BANDIT_ALPHA, random_seed=RANDOM_SEED)
    reranker = GreenReRanker(bandit=bandit, context_builder=context_builder, 
                             tau=TAU, boost_levels=BOOST_LEVELS)
    simulator = SERPSimulator(n_results_per_page=N_RESULTS_PER_PAGE, 
                             green_preference=0.3, click_noise=0.1, random_seed=RANDOM_SEED)
    
    print("   ✓ CarbonResolver initialized")
    print("   ✓ ContextBuilder initialized")
    print("   ✓ LinUCB bandit initialized")
    print("   ✓ GreenReRanker initialized")
    print("   ✓ SERPSimulator initialized")
    
    # Training phase
    print(f"\n[2/6] Training bandit on {N_TRAINING_QUERIES} queries...")
    
    training_metrics = []
    arm_counts_history = []
    mask_rate_history = []
    
    for query_idx in tqdm(range(N_TRAINING_QUERIES), desc="Training"):
        # Generate a SERP
        serp = simulator.generate_batch(1)[0]
        results = serp['results']
        relevance_scores = serp['relevance_scores']
        
        # Get carbon scores
        domains = [r.domain for r in results]
        carbon_scores = carbon_resolver.get_scores_batch(domains)
        
        # Build query-like string
        query = f"training_query_{query_idx}"
        
        # Re-rank
        reranked, metadata = reranker.rerank(query, results, relevance_scores, carbon_scores)
        
        # Simulate clicks
        positions = [r.new_position for r in reranked]
        clicked_domains = simulator.simulate_clicks(results, positions, query)
        
        # Build contexts for update
        contexts = context_builder.build_batch_contexts(query, results, carbon_scores)
        
        # Update bandit
        reranker.observe_clicks(reranked, clicked_domains, contexts, reward_function)
        
        # Track metrics
        metrics = evaluate_session(results, reranked, clicked_domains, carbon_scores)
        training_metrics.append(metrics)
        arm_counts_history.append(bandit.arm_counts.copy())
        mask_rate_history.append(reranker.get_mask_rate())
    
    print("   ✓ Training complete")
    
    # Print training summary
    train_summary = compute_carbon_ctr_tradeoff(training_metrics)
    print(f"\n   Training Summary (last 100 queries):")
    recent_metrics = training_metrics[-100:]
    recent_summary = compute_carbon_ctr_tradeoff(recent_metrics)
    print(f"   NDCG@5: {recent_summary['mean_ndcg@5']:.4f}")
    print(f"   CTR: {recent_summary['mean_ctr']:.4f}")
    print(f"   Avg Carbon Clicked: {recent_summary['mean_avg_carbon_clicked']:.4f}")
    print(f"   Carbon Improvement: {recent_summary['mean_carbon_improvement']:.4f}")
    print(f"   Mean boosts per query: {recent_summary['mean_boost_per_query']:.2f}")
    
    # Arm statistics
    print(f"\n[3/6] Learned arm statistics...")
    arm_stats = bandit.get_arm_stats()
    for arm_name, stats in arm_stats.items():
        print(f"   {arm_name}: count={stats['count']}, avg_reward={stats['avg_reward']:.4f}")
    
    # Evaluation phase - compare against baselines
    print(f"\n[4/6] Evaluating against baselines on {N_EVAL_QUERIES} queries...")
    
    eval_results = {
        'bandit': [],
        'baseline_no_rerank': [],
        'baseline_green_only': [],
        'baseline_static_weighted': []
    }
    
    for query_idx in tqdm(range(N_EVAL_QUERIES), desc="Evaluating"):
        # Generate SERP
        serp = simulator.generate_batch(1)[0]
        results = serp['results']
        relevance_scores = serp['relevance_scores']
        domains = [r.domain for r in results]
        carbon_scores = carbon_resolver.get_scores_batch(domains)
        query = f"eval_query_{query_idx}"
        
        # 1. Bandit re-ranking
        reranked, metadata = reranker.rerank(query, results, relevance_scores, carbon_scores)
        positions = [r.new_position for r in reranked]
        clicked_bandit = simulator.simulate_clicks(results, positions, query)
        metrics_bandit = evaluate_session(results, reranked, clicked_bandit, carbon_scores)
        eval_results['bandit'].append(metrics_bandit)
        
        # 2. No re-rank baseline
        clicked_no_rerank = simulator.simulate_clicks(results, list(range(len(results))), query)
        fake_reranked = [type('obj', (object,), {
            'domain': r.domain, 'relevance_score': r.relevance_score,
            'carbon_score': r.carbon_score, 'new_position': i,
            'boost_applied': 0, 'original_position': i
        })() for i, r in enumerate(results)]
        metrics_no_rerank = evaluate_session(results, fake_reranked, clicked_no_rerank, carbon_scores)
        eval_results['baseline_no_rerank'].append(metrics_no_rerank)
        
        # 3. Green-only baseline
        green_order = BaselineRankers.green_only(results, relevance_scores, carbon_scores)
        reordered_results = [results[i] for i in green_order]
        clicked_green = simulator.simulate_clicks(reordered_results, list(range(len(results))), query)
        fake_green_reranked = [type('obj', (object,), {
            'domain': r.domain, 'relevance_score': r.relevance_score,
            'carbon_score': r.carbon_score, 'new_position': i,
            'boost_applied': 0, 'original_position': green_order[i]
        })() for i, r in enumerate(reordered_results)]
        metrics_green = evaluate_session(results, fake_green_reranked, clicked_green, carbon_scores)
        eval_results['baseline_green_only'].append(metrics_green)
        
        # 4. Static weighted baseline
        weighted_order = BaselineRankers.static_weighted(results, relevance_scores, carbon_scores)
        reordered_weighted = [results[i] for i in weighted_order]
        clicked_weighted = simulator.simulate_clicks(reordered_weighted, list(range(len(results))), query)
        fake_weighted_reranked = [type('obj', (object,), {
            'domain': r.domain, 'relevance_score': r.relevance_score,
            'carbon_score': r.carbon_score, 'new_position': i,
            'boost_applied': 0, 'original_position': weighted_order[i]
        })() for i, r in enumerate(reordered_weighted)]
        metrics_weighted = evaluate_session(results, fake_weighted_reranked, clicked_weighted, carbon_scores)
        eval_results['baseline_static_weighted'].append(metrics_weighted)
    
    # Summarize evaluation
    print(f"\n[5/6] Evaluation Results:")
    print("-" * 70)
    print(f"{'Method':<30} {'NDCG@5':>8} {'NDCG@10':>8} {'CTR':>8} {'Avg Carbon':>10}")
    print("-" * 70)
    
    for method_name, metrics_list in eval_results.items():
        summary = compute_carbon_ctr_tradeoff(metrics_list)
        print(f"{method_name:<30} {summary['mean_ndcg@5']:>8.4f} {summary['mean_ndcg@10']:>8.4f} "
              f"{summary['mean_ctr']:>8.4f} {summary['mean_avg_carbon_clicked']:>10.4f}")
    
    print("-" * 70)
    
    # Visualization
    print(f"\n[6/6] Generating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Learning curve: cumulative reward
    ax = axes[0, 0]
    cumulative_reward = np.cumsum([m['avg_carbon_clicked'] for m in training_metrics]) / np.arange(1, len(training_metrics)+1)
    ax.plot(cumulative_reward)
    ax.set_xlabel('Query')
    ax.set_ylabel('Running Avg Carbon Clicked')
    ax.set_title('Learning Curve: Carbon Score of Clicks')
    ax.grid(True, alpha=0.3)
    
    # 2. Arm selection over time
    ax = axes[0, 1]
    arm_history = np.array(arm_counts_history)
    for a in range(arm_history.shape[1]):
        ax.plot(arm_history[:, a], label=f'Boost +{a}')
    ax.set_xlabel('Query')
    ax.set_ylabel('Times Selected')
    ax.set_title('Arm Selection Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Mask rate over time
    ax = axes[0, 2]
    window = 20
    mask_smooth = np.convolve(mask_rate_history, np.ones(window)/window, mode='valid')
    ax.plot(mask_smooth)
    ax.set_xlabel('Query')
    ax.set_ylabel('Action Mask Rate')
    ax.set_title('Guardrail Activation Over Time')
    ax.grid(True, alpha=0.3)
    
    # 4. NDCG comparison
    ax = axes[1, 0]
    methods = list(eval_results.keys())
    ndcg5_vals = [compute_carbon_ctr_tradeoff(eval_results[m])['mean_ndcg@5'] for m in methods]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(range(len(methods)), ndcg5_vals, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('baseline_', '') for m in methods], rotation=45, ha='right')
    ax.set_ylabel('NDCG@5')
    ax.set_title('Relevance Preservation')
    for bar, val in zip(bars, ndcg5_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center')
    
    # 5. Carbon score of clicked results
    ax = axes[1, 1]
    carbon_vals = [compute_carbon_ctr_tradeoff(eval_results[m])['mean_avg_carbon_clicked'] for m in methods]
    bars = ax.bar(range(len(methods)), carbon_vals, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('baseline_', '') for m in methods], rotation=45, ha='right')
    ax.set_ylabel('Avg Carbon of Clicked Results')
    ax.set_title('Carbon Impact')
    for bar, val in zip(bars, carbon_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center')
    
    # 6. Trade-off scatter
    ax = axes[1, 2]
    for i, method in enumerate(methods):
        summary = compute_carbon_ctr_tradeoff(eval_results[method])
        ax.scatter(summary['mean_ndcg@5'], summary['mean_avg_carbon_clicked'], 
                  color=colors[i], s=200, label=method.replace('baseline_', ''), edgecolors='black')
    ax.set_xlabel('NDCG@5 (Relevance)')
    ax.set_ylabel('Avg Carbon Clicked (Sustainability)')
    ax.set_title('Relevance vs Carbon Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = 'experiments/output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/phase0_results.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved figure to {output_dir}/phase0_results.png")
    
    plt.show()
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 0 VALIDATION COMPLETE")
    print("=" * 70)
    
    bandit_summary = compute_carbon_ctr_tradeoff(eval_results['bandit'])
    baseline_summary = compute_carbon_ctr_tradeoff(eval_results['baseline_no_rerank'])
    
    print(f"\nKey Findings:")
    print(f"  • Carbon improvement over baseline: {(bandit_summary['mean_avg_carbon_clicked'] - baseline_summary['mean_avg_carbon_clicked']):.4f}")
    print(f"  • NDCG@5 change: {(bandit_summary['mean_ndcg@5'] - baseline_summary['mean_ndcg@5']):.4f}")
    print(f"  • CTR change: {(bandit_summary['mean_ctr'] - baseline_summary['mean_ctr']):.4f}")
    print(f"  • Average boosts per query: {bandit_summary['mean_boost_per_query']:.2f}")
    print(f"  • Final mask rate: {mask_rate_history[-1]:.3f}")
    
    if bandit_summary['mean_avg_carbon_clicked'] > baseline_summary['mean_avg_carbon_clicked']:
        print(f"\n  ✓ Bandit successfully increased carbon score of clicked results")
    
    if bandit_summary['mean_ndcg@5'] >= baseline_summary['mean_ndcg@5'] * 0.95:
        print(f"  ✓ Relevance maintained within 5% of baseline")
    
    return eval_results, training_metrics


if __name__ == "__main__":
    eval_results, training_metrics = run_phase0_validation()