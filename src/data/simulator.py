import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class SimulatedResult:
    """A simulated search result for testing."""
    domain: str
    title: str
    relevance_score: float
    carbon_score: float


class SERPSimulator:
    """
    Simulates search engine result pages with known properties
    for offline training and evaluation of the bandit.
    """
    
    # Sample domains for simulation
    SAMPLE_DOMAINS = [
        'example.com', 'wikipedia.org', 'github.com', 'stackoverflow.com',
        'medium.com', 'nytimes.com', 'reddit.com', 'amazon.com',
        'microsoft.com', 'apple.com', 'google.com', 'facebook.com',
        'twitter.com', 'linkedin.com', 'youtube.com', 'instagram.com',
        'netflix.com', 'spotify.com', 'wordpress.org', 'mozilla.org'
    ]
    
    def __init__(self, 
                 n_results_per_page: int = 10,
                 green_preference: float = 0.3,
                 click_noise: float = 0.1,
                 random_seed: int = 42):
        """
        Args:
            n_results_per_page: Number of results per SERP
            green_preference: How much users prefer green sites (0-1)
            click_noise: Random noise in click probability
            random_seed: Random seed
        """
        self.n_results = n_results_per_page
        self.green_preference = green_preference
        self.click_noise = click_noise
        self.rng = np.random.RandomState(random_seed)
        
        # Pre-generate domain properties
        self.domain_properties = self._generate_domain_properties()
    
    def _generate_domain_properties(self) -> Dict[str, dict]:
        """Generate realistic properties for sample domains."""
        props = {}
        for i, domain in enumerate(self.SAMPLE_DOMAINS):
            # Vary relevance across domains
            base_relevance = 0.3 + 0.5 * (1 - i / len(self.SAMPLE_DOMAINS))
            
            # Vary carbon scores
            if '.org' in domain or '.gov' in domain or '.edu' in domain:
                base_carbon = self.rng.uniform(0.6, 1.0)
            elif domain in ['google.com', 'microsoft.com', 'apple.com', 'mozilla.org']:
                base_carbon = self.rng.uniform(0.7, 1.0)
            elif domain in ['amazon.com', 'netflix.com', 'youtube.com']:
                base_carbon = self.rng.uniform(0.1, 0.5)
            else:
                base_carbon = self.rng.uniform(0.2, 0.8)
            
            props[domain] = {
                'base_relevance': base_relevance,
                'base_carbon': base_carbon
            }
        return props
    
    def generate_serp(self, query: str = None) -> Tuple[List[SimulatedResult], np.ndarray]:
        """
        Generate a simulated SERP.
        
        Returns:
            List of SimulatedResult and array of relevance scores
        """
        # Select random domains for this SERP
        selected = self.rng.choice(self.SAMPLE_DOMAINS, 
                                   size=min(self.n_results, len(self.SAMPLE_DOMAINS)),
                                   replace=False)
        
        results = []
        relevance_scores = np.zeros(len(selected))
        
        for i, domain in enumerate(selected):
            props = self.domain_properties[domain]
            
            # Add some noise to relevance for this specific query
            relevance = np.clip(
                props['base_relevance'] + self.rng.normal(0, 0.1),
                0.0, 1.0
            )
            
            # Carbon score with slight variation
            carbon = np.clip(
                props['base_carbon'] + self.rng.normal(0, 0.05),
                0.0, 1.0
            )
            
            results.append(SimulatedResult(
                domain=domain,
                title=f"Result for {domain}",
                relevance_score=relevance,
                carbon_score=carbon
            ))
            
            relevance_scores[i] = relevance
        
        # Sort by relevance (highest first) - simulating base ranker
        sorted_indices = np.argsort(relevance_scores)[::-1]
        results = [results[i] for i in sorted_indices]
        relevance_scores = relevance_scores[sorted_indices]
        
        return results, relevance_scores
    
    def simulate_clicks(self,
                       results: List[SimulatedResult],
                       positions: List[int] = None,
                       query: str = None) -> List[str]:
        """
        Simulate user clicks using position-based model with green preference.
        
        Click probability = f(position) * relevance * (1 + green_preference * carbon_score) + noise
        
        Args:
            results: List of results
            positions: Current positions (if re-ranked). If None, uses original order.
        
        Returns:
            List of domain names that were clicked
        """
        if positions is None:
            positions = list(range(len(results)))
        
        clicked = []
        
        # Position bias: higher positions get more clicks
        position_bias = np.exp(-0.3 * np.array(positions))
        
        for i, (result, pos) in enumerate(zip(results, positions)):
            # Base click probability
            p_click = position_bias[i] * result.relevance_score
            
            # Green preference increases click probability
            p_click *= (1 + self.green_preference * result.carbon_score)
            
            # Add noise
            p_click = np.clip(p_click + self.rng.normal(0, self.click_noise), 0.0, 1.0)
            
            # Sample click
            if self.rng.random() < p_click:
                clicked.append(result.domain)
        
        return clicked
    
    def generate_batch(self, n_queries: int) -> List[dict]:
        """Generate a batch of SERPs for training/testing."""
        batch = []
        for _ in range(n_queries):
            results, relevance_scores = self.generate_serp()
            batch.append({
                'results': results,
                'relevance_scores': relevance_scores,
                'query': f"query_{_}"  # Placeholder
            })
        return batch