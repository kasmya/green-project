"""
Offline Replay Evaluation on Real Data
======================================
Loads collected SERP dataset and replays bandit decisions.
Compares against baselines using real domains and carbon scores.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.bandit.linucb import LinUCB
from src.bandit.context import ContextBuilder
from src.reranker.engine import GreenReRanker
from src.evaluation.metrics import evaluate_session, compute_carbon_ctr_tradeoff


# ============================================================================
# Configuration
# ============================================================================

class ReplayConfig:
    """Configuration for offline replay evaluation."""
    def __init__(self):
        self.dataset_path = "data/collected/serp_dataset.json"
        self.n_folds = 5  # Cross-validation folds
        self.bandit_alpha = 1.0
        self.tau = 0.05  # Relevance guardrail threshold
        self.boost_levels = [0, 1, 2, 3]
        self.random_seed = 42
        self.output_dir = "experiments/output"


# ============================================================================
# Click Simulator for Real Data
# ============================================================================

class RealDataClickSimulator:
    """
    Simulates clicks on real SERP data using position-bias model.
    Uses actual relevance scores from the dataset.
    """
    
    def __init__(self, click_probability: float = 0.40,  # HIGHER base probability
                 position_decay: float = 0.5,
                 green_bonus: float = 0.2,
                 seed: int = 42):
        self.click_prob = click_probability
        self.position_decay = position_decay
        self.green_bonus = green_bonus
        self.rng = np.random.RandomState(seed)
    
    def simulate_clicks(self, 
                       results: List[Dict],
                       positions: List[int]) -> List[str]:
        """
        Simulate clicks on results at given positions.
        User is MORE LIKELY to click overall, and prefers green.
        """
        clicked = []
        
        for i, (result, pos) in enumerate(zip(results, positions)):
            # Position bias (stronger decay)
            pos_weight = np.exp(-self.position_decay * pos)
            
            # Relevance weight (use the real relevance score)
            rel_weight = result.get('relevance_score', 0.5)
            
            # Green bonus (stronger preference for green sites)
            carbon = result.get('carbon_score', 0.5)
            green_weight = 1.0 + self.green_bonus * carbon
            
            # Higher click probability
            p_click = self.click_prob * pos_weight * rel_weight * green_weight
            p_click = np.clip(p_click, 0.0, 0.90)  # Allow up to 90%
            
            if self.rng.rand() < p_click:
                clicked.append(result.get('domain', ''))
        
        return clicked


# ============================================================================
# Baseline Re-rankers
# ============================================================================

class BaselineReRankers:
    """Collection of baseline strategies for comparison."""
    
    @staticmethod
    def no_rerank(results: List[Dict]) -> List[int]:
        """Keep original Google ordering."""
        return list(range(len(results)))
    
    @staticmethod
    def carbon_only(results: List[Dict]) -> List[int]:
        """Rank purely by carbon score (greenest first)."""
        carbons = [r.get('carbon_score', 0.5) for r in results]
        sorted_indices = np.argsort(carbons)[::-1]
        # Return the new positions
        positions = [0] * len(results)
        for new_pos, old_idx in enumerate(sorted_indices):
            positions[old_idx] = new_pos
        return positions
    
    @staticmethod
    def static_weighted(results: List[Dict], carbon_weight: float = 0.3) -> List[int]:
        """Static weighted combination of relevance and carbon."""
        scores = []
        for r in results:
            rel = r.get('relevance_score', 0.5)
            carbon = r.get('carbon_score', 0.5)
            combined = (1 - carbon_weight) * rel + carbon_weight * carbon
            scores.append(combined)
        
        sorted_indices = np.argsort(scores)[::-1]
        positions = [0] * len(results)
        for new_pos, old_idx in enumerate(sorted_indices):
            positions[old_idx] = new_pos
        return positions


# ============================================================================
# Offline Replay Evaluator
# ============================================================================

class OfflineReplayEvaluator:
    """
    Evaluates bandit on real collected data.
    """
    
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.click_simulator = RealDataClickSimulator(seed=config.random_seed)
        
        # Load dataset
        with open(config.dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        print(f"Loaded dataset: {len(self.dataset['queries'])} queries")
        print(f"Unique domains: {len(self.dataset['all_domains'])}")
        
        # Compute dataset statistics
        self._compute_dataset_stats()
    
    def _compute_dataset_stats(self):
        """Compute summary statistics of the dataset."""
        all_carbons = []
        green_count = 0
        
        for query_data in self.dataset['queries']:
            for result in query_data['results']:
                carbon = result.get('carbon_score', 0.5)
                all_carbons.append(carbon)
                if carbon >= 0.5:
                    green_count += 1
        
        self.dataset_stats = {
            'n_queries': len(self.dataset['queries']),
            'n_results': sum(len(q['results']) for q in self.dataset['queries']),
            'mean_carbon': np.mean(all_carbons),
            'std_carbon': np.std(all_carbons),
            'green_fraction': green_count / len(all_carbons) if all_carbons else 0,
        }
        
        print("\nDataset Statistics:")
        for key, val in self.dataset_stats.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")
    
    def _get_results_list(self, query_data: Dict) -> List[Dict]:
        """Convert query data to results list format expected by re-ranker."""
        return query_data['results']
    
    def _get_carbon_scores(self, results: List[Dict]) -> Dict[str, float]:
        """Extract carbon scores from results."""
        return {r['domain']: r.get('carbon_score', 0.5) for r in results}
    
    def _get_relevance_scores(self, results: List[Dict]) -> np.ndarray:
        """Extract relevance scores from results."""
        return np.array([r.get('relevance_score', 0.5) for r in results])
    
    def evaluate_bandit(self, 
                       bandit: LinUCB,
                       context_builder: ContextBuilder,
                       reranker: GreenReRanker,
                       train_queries: List[Dict],
                       test_queries: List[Dict],
                       reward_fn: Callable) -> Dict:
        """
        Train bandit on train_queries, evaluate on test_queries.
        """
        # Training phase
        print(f"\n  Training on {len(train_queries)} queries...")
        
        train_metrics = []
        for query_data in tqdm(train_queries, desc="  Training"):
            results = self._get_results_list(query_data)
            relevance = self._get_relevance_scores(results)
            carbon_scores = self._get_carbon_scores(results)
            query = query_data['query']
            
            # Re-rank
            reranked, metadata = reranker.rerank(query, results, relevance, carbon_scores)
            
            # Simulate clicks
            positions = [r.new_position for r in reranked]
            clicked = self.click_simulator.simulate_clicks(results, positions)
            
            # Build contexts for update
            contexts = context_builder.build_batch_contexts(query, results, carbon_scores)
            
            # Update bandit
            reranker.observe_clicks(reranked, clicked, contexts, reward_fn)
            
            # Record metrics
            metrics = evaluate_session(results, reranked, clicked, carbon_scores)
            train_metrics.append(metrics)
        
        # Testing phase
        print(f"  Testing on {len(test_queries)} queries...")
        
        test_metrics = []
        for query_data in tqdm(test_queries, desc="  Testing"):
            results = self._get_results_list(query_data)
            relevance = self._get_relevance_scores(results)
            carbon_scores = self._get_carbon_scores(results)
            query = query_data['query']
            
            # Re-rank (no update)
            reranked, metadata = reranker.rerank(query, results, relevance, carbon_scores)
            
            # Simulate clicks
            positions = [r.new_position for r in reranked]
            clicked = self.click_simulator.simulate_clicks(results, positions)
            
            # Record metrics
            metrics = evaluate_session(results, reranked, clicked, carbon_scores)
            test_metrics.append(metrics)
        
        return {
            'train': compute_carbon_ctr_tradeoff(train_metrics),
            'test': compute_carbon_ctr_tradeoff(test_metrics),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
        }
    
    def evaluate_baseline(self,
                         baseline_fn: Callable,
                         queries: List[Dict],
                         name: str) -> Dict:
        """Evaluate a baseline re-ranking strategy."""
        print(f"  Evaluating {name} on {len(queries)} queries...")
        
        all_metrics = []
        for query_data in tqdm(queries, desc=f"  {name}"):
            results = self._get_results_list(query_data)
            carbon_scores = self._get_carbon_scores(results)
            
            # Get new positions from baseline
            positions = baseline_fn(results)
            
            # Create fake reranked results
            RerankedResult = type('RerankedResult', (object,), {})
            reranked = []
            for i, pos in enumerate(positions):
                r = results[i]
                obj = RerankedResult()
                obj.domain = r['domain']
                obj.relevance_score = r.get('relevance_score', 0.5)
                obj.carbon_score = r.get('carbon_score', 0.5)
                obj.new_position = pos
                obj.original_position = i
                obj.boost_applied = max(0, i - pos)
                reranked.append(obj)
            
            # Sort by new position
            reranked.sort(key=lambda x: x.new_position)
            
            # Simulate clicks
            clicked = self.click_simulator.simulate_clicks(results, positions)
            
            # Record metrics
            metrics = evaluate_session(results, reranked, clicked, carbon_scores)
            all_metrics.append(metrics)
        
        return {
            'summary': compute_carbon_ctr_tradeoff(all_metrics),
            'all_metrics': all_metrics,
        }
    
    def run_full_evaluation(self, reward_fn: Callable, n_folds: int = 5) -> Dict:
        """Run full evaluation with cross-validation."""
        
        print(f"\n{'='*60}")
        print(f"FULL EVALUATION ({n_folds}-fold cross-validation)")
        print(f"{'='*60}")
        
        # Shuffle queries
        queries = self.dataset['queries'].copy()
        np.random.seed(self.config.random_seed)
        np.random.shuffle(queries)
        
        # Split into folds
        fold_size = len(queries) // n_folds
        
        all_results = {
            'bandit': [],
            'no_rerank': [],
            'carbon_only': [],
            'static_weighted': [],
        }
        
        for fold in range(n_folds):
            print(f"\n{'─'*60}")
            print(f"FOLD {fold + 1}/{n_folds}")
            print(f"{'─'*60}")
            
            # Split data
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size
            
            test_queries = queries[test_start:test_end]
            train_queries = queries[:test_start] + queries[test_end:]
            
            # Initialize fresh bandit for this fold
            bandit = LinUCB(
                n_arms=len(self.config.boost_levels),
                context_dim=8,
                alpha=self.config.bandit_alpha,
                random_seed=self.config.random_seed + fold
            )
            context_builder = ContextBuilder(config={'context_dim': 8})
            reranker = GreenReRanker(
                bandit=bandit,
                context_builder=context_builder,
                tau=self.config.tau,
                boost_levels=self.config.boost_levels
            )
            
            # Evaluate bandit
            bandit_result = self.evaluate_bandit(
                bandit, context_builder, reranker,
                train_queries, test_queries, reward_fn
            )
            all_results['bandit'].append(bandit_result)
            
            # Evaluate baselines
            no_rerank_result = self.evaluate_baseline(
                BaselineReRankers.no_rerank, test_queries, "No Re-rank"
            )
            all_results['no_rerank'].append(no_rerank_result)
            
            carbon_only_result = self.evaluate_baseline(
                BaselineReRankers.carbon_only, test_queries, "Carbon Only"
            )
            all_results['carbon_only'].append(carbon_only_result)
            
            static_result = self.evaluate_baseline(
                lambda r: BaselineReRankers.static_weighted(r, 0.3),
                test_queries, "Static Weighted"
            )
            all_results['static_weighted'].append(static_result)
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        return aggregated
    
    def _aggregate_results(self, all_results: Dict) -> Dict:
        """Aggregate results across folds."""
        aggregated = {}
        
        for method, fold_results in all_results.items():
            # Extract test summaries
            if method == 'bandit':
                test_summaries = [r['test'] for r in fold_results]
            else:
                test_summaries = [r['summary'] for r in fold_results]
            
            # Compute mean and std for each metric
            if test_summaries:
                metrics = list(test_summaries[0].keys())
                agg = {}
                for metric in metrics:
                    values = [s[metric] for s in test_summaries]
                    agg[f'{metric}_mean'] = np.mean(values)
                    agg[f'{metric}_std'] = np.std(values)
                
                aggregated[method] = agg
        
        return aggregated
    
    def print_results(self, aggregated: Dict):
        """Print evaluation results in a formatted table."""
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*80}")
        
        metrics_to_show = [
            'mean_ndcg@5', 'mean_ndcg@10', 'mean_ctr',
            'mean_avg_carbon_clicked', 'mean_carbon_improvement',
        ]
        
        # Header
        header = f"{'Method':<25}"
        for m in metrics_to_show:
            header += f" {m:>12}"
        print(header)
        print("-" * 80)
        
        # Results
        for method in ['no_rerank', 'carbon_only', 'static_weighted', 'bandit']:
            if method not in aggregated:
                continue
            results = aggregated[method]
            row = f"{method:<25}"
            for m in metrics_to_show:
                mean_key = f'{m}_mean'
                if mean_key in results:
                    row += f" {results[mean_key]:>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)
        
        print("-" * 80)
        
        # Statistical comparison
        print(f"\n{'='*80}")
        print(f"STATISTICAL COMPARISON (Bandit vs No Re-rank)")
        print(f"{'='*80}")
        
        if 'bandit' in aggregated and 'no_rerank' in aggregated:
            bandit_res = aggregated['bandit']
            baseline_res = aggregated['no_rerank']
            
            comparisons = [
                ('NDCG@5', 'mean_ndcg@5_mean'),
                ('NDCG@10', 'mean_ndcg@10_mean'),
                ('CTR', 'mean_ctr_mean'),
                ('Avg Carbon Clicked', 'mean_avg_carbon_clicked_mean'),
                ('Carbon Improvement', 'mean_carbon_improvement_mean'),
            ]
            
            for name, key in comparisons:
                if key in bandit_res and key in baseline_res:
                    diff = bandit_res[key] - baseline_res[key]
                    direction = "↑" if diff > 0 else "↓"
                    print(f"  {name:<25}: {diff:+.4f} {direction}")
        
        # Save results
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_file = os.path.join(self.config.output_dir, 'offline_replay_results.json')
        
        # Convert numpy values to Python native types for JSON
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(convert(aggregated), f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")


# ============================================================================
# Reward Function
# ============================================================================

def real_data_reward(clicked: bool, carbon_score: float, position: int, boost_applied: int) -> float:
    """Reward function for real data evaluation."""
    if not clicked:
        return 0.0
    
    reward = 1.0
    reward += 0.2 * carbon_score  # Green bonus
    reward -= 0.02 * position      # Position penalty
    reward -= 0.01 * boost_applied # Boost penalty
    
    return max(reward, 0.0)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run offline replay evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Offline replay evaluation on real SERP data')
    parser.add_argument('--dataset', type=str, default='data/collected/serp_dataset.json',
                       help='Path to collected dataset')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Bandit exploration parameter')
    parser.add_argument('--tau', type=float, default=0.05,
                       help='Relevance guardrail threshold')
    parser.add_argument('--output', type=str, default='experiments/output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    config = ReplayConfig()
    config.dataset_path = args.dataset
    config.n_folds = args.folds
    config.bandit_alpha = args.alpha
    config.tau = args.tau
    config.output_dir = args.output
    
    # Check if dataset exists
    if not os.path.exists(config.dataset_path):
        print(f"\n{'!'*60}")
        print(f"DATASET NOT FOUND: {config.dataset_path}")
        print(f"{'!'*60}")
        print("\nYou need to collect data first. Run:")
        print(f"  python src/data/real_serp_collector.py --num-queries 50")
        print("\nOr use the synthetic simulator:")
        print(f"  python experiments/phase0_validation.py")
        return
    
    evaluator = OfflineReplayEvaluator(config)
    
    aggregated = evaluator.run_full_evaluation(
        reward_fn=real_data_reward,
        n_folds=config.n_folds
    )
    
    evaluator.print_results(aggregated)
    
    return aggregated


if __name__ == "__main__":
    aggregated = main()