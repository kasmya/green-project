import numpy as np
from typing import List, Optional
import json


class LinUCB:
    """
    Linear Upper Confidence Bound for contextual bandits.
    Supports action masking for relevance guardrails.
    """
    
    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0, random_seed: int = 42):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.rng = np.random.RandomState(random_seed)
        
        # Initialize per-arm parameters
        # A[a] = identity matrix (context_dim x context_dim)
        # b[a] = zero vector (context_dim,)
        self.A = [np.identity(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        
        # Tracking
        self.arm_counts = np.zeros(n_arms)
        self.total_rewards = np.zeros(n_arms)
    
    def select_arm(self, context: np.ndarray, action_mask: Optional[List[bool]] = None) -> int:
        """
        Select arm with highest UCB score.
        
        Args:
            context: Feature vector (context_dim,)
            action_mask: List of booleans, True means action is ALLOWED.
                        If None, all actions are allowed.
        
        Returns:
            Index of selected arm
        """
        if action_mask is None:
            action_mask = [True] * self.n_arms
        
        ucb_scores = np.full(self.n_arms, -np.inf)
        
        for a in range(self.n_arms):
            if not action_mask[a]:
                continue
            
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            
            # UCB = estimated reward + uncertainty bonus
            expected_reward = theta_a @ context
            uncertainty = self.alpha * np.sqrt(context @ A_inv @ context.T)
            ucb_scores[a] = expected_reward + uncertainty
        
        # Small random tie-breaking
        max_score = np.max(ucb_scores)
        candidates = np.where(np.abs(ucb_scores - max_score) < 1e-10)[0]
        
        if len(candidates) > 1:
            return self.rng.choice(candidates)
        return np.argmax(ucb_scores)
    
    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Update bandit parameters after observing reward.
        
        Args:
            arm: Index of chosen arm
            context: Feature vector (context_dim,)
            reward: Observed reward (float)
        """
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        self.arm_counts[arm] += 1
        self.total_rewards[arm] += reward
    
    def get_arm_stats(self) -> dict:
        """Returns statistics about each arm."""
        stats = {}
        for a in range(self.n_arms):
            count = self.arm_counts[a]
            avg_reward = self.total_rewards[a] / count if count > 0 else 0
            theta = np.linalg.inv(self.A[a]) @ self.b[a]
            stats[f"arm_{a}"] = {
                "count": int(count),
                "avg_reward": float(avg_reward),
                "theta": theta.tolist()
            }
        return stats
    
    def save_state(self, filepath: str):
        """Save bandit state to file."""
        state = {
            'A': [a.tolist() for a in self.A],
            'b': [b.tolist() for b in self.b],
            'arm_counts': self.arm_counts.tolist(),
            'total_rewards': self.total_rewards.tolist(),
            'n_arms': self.n_arms,
            'context_dim': self.context_dim,
            'alpha': self.alpha
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)
    
    @classmethod
    def load_state(cls, filepath: str) -> 'LinUCB':
        """Load bandit state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        bandit = cls(
            n_arms=state['n_arms'],
            context_dim=state['context_dim'],
            alpha=state['alpha']
        )
        bandit.A = [np.array(a) for a in state['A']]
        bandit.b = [np.array(b) for b in state['b']]
        bandit.arm_counts = np.array(state['arm_counts'])
        bandit.total_rewards = np.array(state['total_rewards'])
        
        return bandit