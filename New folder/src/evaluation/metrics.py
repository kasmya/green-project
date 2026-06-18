import numpy as np
from typing import List, Dict, Union
from ..reranker.engine import ReRankedResult


def compute_dcg(relevance_scores: np.ndarray, k: int = None) -> float:
    """Compute Discounted Cumulative Gain."""
    if k is None:
        k = len(relevance_scores)
    
    relevance_scores = np.array(relevance_scores[:k], dtype=float)
    discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
    return np.sum(relevance_scores / discounts)


def compute_ndcg(original_relevance: np.ndarray, 
                 reranked_relevance: np.ndarray, 
                 k: int = None) -> float:
    """Compute Normalized DCG."""
    dcg = compute_dcg(reranked_relevance, k)
    idcg = compute_dcg(sorted(original_relevance, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_ctr(clicked_domains: List[str], n_results: int) -> float:
    """Compute Click-Through Rate."""
    return len(clicked_domains) / n_results if n_results > 0 else 0.0


def compute_avg_carbon_clicked(clicked_domains: List[str],
                               carbon_scores: Dict[str, float]) -> float:
    """Compute average carbon score of clicked results."""
    if not clicked_domains:
        return 0.0
    
    scores = [carbon_scores.get(d, 0.5) for d in clicked_domains]
    return np.mean(scores)


def compute_position_weighted_carbon(results: List[Union[Dict, ReRankedResult]]) -> float:
    """
    Compute position-weighted carbon score of the SERP.
    Lower positions (higher on page) get higher weight.
    """
    if not results:
        return 0.0
    
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for i, result in enumerate(results):
        # Position weight: 1/log2(position+2) like DCG
        weight = 1.0 / np.log2(i + 2)
        
        # Handle both dict and object types
        if isinstance(result, dict):
            carbon_score = result.get('carbon_score', 0.5)
        else:
            carbon_score = result.carbon_score
            
        weighted_sum += weight * carbon_score
        weight_sum += weight
    
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0


def evaluate_session(original_results: List[Union[Dict, ReRankedResult]],
                    reranked_results: List[ReRankedResult],
                    clicked_domains: List[str],
                    carbon_scores: Dict[str, float]) -> dict:
    """
    Evaluate a single re-ranking session.
    Returns metrics dictionary.
    """
    # Helper function to extract relevance score from dict or object
    def get_relevance_score(item):
        if isinstance(item, dict):
            return item.get('relevance_score', 0.5)
        else:
            return item.relevance_score
    
    # Helper function to extract carbon score from dict or object
    def get_carbon_score(item):
        if isinstance(item, dict):
            return item.get('carbon_score', 0.5)
        else:
            return item.carbon_score
    
    # Helper function to extract boost applied from dict or object
    def get_boost_applied(item):
        if isinstance(item, dict):
            return item.get('boost_applied', 0)
        else:
            return item.boost_applied
    
    # Original relevance ordering
    original_relevance = np.array([get_relevance_score(r) for r in original_results])
    
    # Re-ranked relevance ordering
    reranked_relevance = np.array([get_relevance_score(r) for r in reranked_results])
    
    # Carbon scores in original order
    original_carbon = np.array([get_carbon_score(r) for r in original_results])
    
    # Carbon scores in reranked order
    reranked_carbon = np.array([get_carbon_score(r) for r in reranked_results])
    
    metrics = {
        'ndcg@5': compute_ndcg(original_relevance, reranked_relevance, k=5),
        'ndcg@10': compute_ndcg(original_relevance, reranked_relevance, k=10),
        'ctr': compute_ctr(clicked_domains, len(reranked_results)),
        'avg_carbon_clicked': compute_avg_carbon_clicked(clicked_domains, carbon_scores),
        'original_avg_carbon': np.mean(original_carbon),
        'reranked_avg_carbon': np.mean(reranked_carbon),
        'carbon_improvement': np.mean(reranked_carbon) - np.mean(original_carbon),
        'position_weighted_carbon_original': compute_position_weighted_carbon(original_results),
        'position_weighted_carbon_reranked': compute_position_weighted_carbon(reranked_results),
        'n_boosts': sum(1 for r in reranked_results if get_boost_applied(r) > 0),
        'avg_boost': np.mean([get_boost_applied(r) for r in reranked_results])
    }
    
    return metrics


def compute_carbon_ctr_tradeoff(sessions_metrics: List[dict]) -> dict:
    """
    Aggregate metrics across multiple sessions.
    """
    aggregated = {
        'mean_ndcg@5': np.mean([m['ndcg@5'] for m in sessions_metrics]),
        'mean_ndcg@10': np.mean([m['ndcg@10'] for m in sessions_metrics]),
        'mean_ctr': np.mean([m['ctr'] for m in sessions_metrics]),
        'mean_avg_carbon_clicked': np.mean([m['avg_carbon_clicked'] for m in sessions_metrics]),
        'mean_original_avg_carbon': np.mean([m['original_avg_carbon'] for m in sessions_metrics]),
        'mean_reranked_avg_carbon': np.mean([m['reranked_avg_carbon'] for m in sessions_metrics]),
        'mean_carbon_improvement': np.mean([m['carbon_improvement'] for m in sessions_metrics]),
        'mean_position_weighted_carbon_original': np.mean([m['position_weighted_carbon_original'] for m in sessions_metrics]),
        'mean_position_weighted_carbon_reranked': np.mean([m['position_weighted_carbon_reranked'] for m in sessions_metrics]),
        'mean_n_boosts': np.mean([m['n_boosts'] for m in sessions_metrics]),
        'mean_avg_boost': np.mean([m['avg_boost'] for m in sessions_metrics]),
        'std_ndcg@5': np.std([m['ndcg@5'] for m in sessions_metrics]),
        'std_ndcg@10': np.std([m['ndcg@10'] for m in sessions_metrics]),
        'std_ctr': np.std([m['ctr'] for m in sessions_metrics]),
        'std_avg_carbon_clicked': np.std([m['avg_carbon_clicked'] for m in sessions_metrics]),
        'std_original_avg_carbon': np.std([m['original_avg_carbon'] for m in sessions_metrics]),
        'std_reranked_avg_carbon': np.std([m['reranked_avg_carbon'] for m in sessions_metrics]),
        'std_carbon_improvement': np.std([m['carbon_improvement'] for m in sessions_metrics]),
        'std_position_weighted_carbon_original': np.std([m['position_weighted_carbon_original'] for m in sessions_metrics]),
        'std_position_weighted_carbon_reranked': np.std([m['position_weighted_carbon_reranked'] for m in sessions_metrics]),
        'std_n_boosts': np.std([m['n_boosts'] for m in sessions_metrics]),
        'std_avg_boost': np.std([m['avg_boost'] for m in sessions_metrics]),
        'n_sessions': len(sessions_metrics)
    }
    
    return aggregated