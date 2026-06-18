import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ..bandit.linucb import LinUCB
from ..bandit.context import ContextBuilder


@dataclass
class ReRankedResult:
    """Represents a result after re-ranking."""
    domain: str
    title: str
    original_position: int
    new_position: int
    boost_applied: int
    carbon_score: float
    relevance_score: float
    was_masked: bool


class GreenReRanker:
    """
    Unified re-ranking engine that applies carbon-aware boosting
    within relevance guardrails (Novelty 3: single loop).
    """
    
    def __init__(self, 
                 bandit: LinUCB,
                 context_builder: ContextBuilder,
                 tau: float = 0.05,
                 boost_levels: List[int] = None):
        """
        Args:
            bandit: Trained or fresh LinUCB instance
            context_builder: Context feature builder
            tau: Maximum relevance gap allowed for boosting (guardrail)
            boost_levels: Available boost amounts (positions to move up)
        """
        self.bandit = bandit
        self.context_builder = context_builder
        self.tau = tau
        self.boost_levels = boost_levels or [0, 1, 2, 3]
        
        # Tracking
        self.total_reranks = 0
        self.actions_masked = 0
        self.boosts_applied = 0
    
    def _compute_action_mask(self,
                            result_index: int,
                            relevance_scores: np.ndarray,
                            boost: int) -> bool:
        """
        Determines if a boost action is allowed for a result.
        Action is masked if boosting would cause the result to overtake
        a significantly more relevant result.
        
        Args:
            result_index: Current position of the result
            relevance_scores: Array of relevance scores for all results
            boost: Amount to boost (positions)
        
        Returns:
            True if action is ALLOWED, False if masked
        """
        if boost == 0:
            return True
        
        new_position = result_index - boost
        if new_position < 0:
            return False  # Can't boost past position 0
        
        # Check all results we would overtake
        current_relevance = relevance_scores[result_index]
        
        for overtaken_idx in range(new_position, result_index):
            overtaken_relevance = relevance_scores[overtaken_idx]
            relevance_gap = current_relevance - overtaken_relevance
            
            # If we're significantly less relevant than a result we'd jump over
            if relevance_gap < -self.tau:
                return False
        
        return True
    
    def rerank(self,
              query: str,
              results: List[dict],
              relevance_scores: np.ndarray,
              carbon_scores: Dict[str, float]) -> Tuple[List[ReRankedResult], dict]:
        """
        Re-rank a single SERP using the bandit with relevance guardrails.
        
        Args:
            query: Search query
            results: List of result dicts with 'domain', 'title', etc.
            relevance_scores: Array of relevance scores (same order as results)
            carbon_scores: Dict mapping domain -> carbon score
        
        Returns:
            Tuple of (re-ranked results list, metadata dict)
        """
        self.total_reranks += 1
        
        # Build contexts for all results
        contexts = self.context_builder.build_batch_contexts(query, results, carbon_scores)
        
        boosted_results = []
        metadata = {
            'query': query,
            'actions_taken': [],
            'masked_actions': 0,
            'total_boosts': 0
        }
        
        for i, (result, context) in enumerate(zip(results, contexts)):
            domain = result.get('domain', '')
            carbon = carbon_scores.get(domain, 0.5)
            
            # Compute action mask for each possible boost level
            action_mask = []
            for boost in self.boost_levels:
                allowed = self._compute_action_mask(i, relevance_scores, boost)
                action_mask.append(allowed)
                if not allowed:
                    self.actions_masked += 1
                    metadata['masked_actions'] += 1
            
            # Select action using bandit with mask
            arm_index = self.bandit.select_arm(context, action_mask)
            boost = self.boost_levels[arm_index]
            
            # Calculate new position
            new_position = max(0, i - boost)
            was_masked = not action_mask[arm_index] if arm_index < len(action_mask) else False
            
            boosted_results.append(ReRankedResult(
                domain=domain,
                title=result.get('title', ''),
                original_position=i,
                new_position=new_position,
                boost_applied=boost,
                carbon_score=carbon,
                relevance_score=relevance_scores[i],
                was_masked=was_masked
            ))
            
            if boost > 0:
                self.boosts_applied += 1
                metadata['total_boosts'] += 1
            
            metadata['actions_taken'].append({
                'domain': domain,
                'arm': arm_index,
                'boost': boost,
                'masked': was_masked
            })
        
        # Sort by new position, break ties with original position
        reranked = sorted(boosted_results, 
                         key=lambda r: (r.new_position, r.original_position))
        
        # Re-assign final positions after sorting
        for i, result in enumerate(reranked):
            result.new_position = i
        
        return reranked, metadata
    
    def observe_clicks(self,
                      reranked_results: List[ReRankedResult],
                      clicked_domains: List[str],
                      contexts: List[np.ndarray],
                      reward_fn):
        """
        Update bandit based on observed clicks.
        
        Args:
            reranked_results: Results from rerank()
            clicked_domains: List of domains that were clicked
            contexts: Context vectors used for each result
            reward_fn: Function(clicked, carbon_score, position) -> reward
        """
        for i, (result, context) in enumerate(zip(reranked_results, contexts)):
            clicked = result.domain in clicked_domains
            reward = reward_fn(
                clicked=clicked,
                carbon_score=result.carbon_score,
                position=result.new_position,
                boost_applied=result.boost_applied
            )
            
            # Find which arm was chosen
            arm = self.boost_levels.index(result.boost_applied) if result.boost_applied in self.boost_levels else 0
            
            self.bandit.update(arm, context, reward)
    
    def get_mask_rate(self) -> float:
        """Returns fraction of actions that were masked."""
        total_actions = self.total_reranks * len(self.boost_levels)
        return self.actions_masked / total_actions if total_actions > 0 else 0.0