import numpy as np
from typing import Dict, List


class ContextBuilder:
    """
    Builds feature vectors for the contextual bandit from SERP and result data.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def classify_query_type(self, query: str) -> float:
        """
        Heuristic query type classifier.
        Returns 0.0 (transactional) to 1.0 (navigational), with informational in between.
        """
        query_lower = query.lower().strip()
        
        # Navigational indicators
        navigational_words = ['login', 'sign in', 'official', 'homepage', 'website']
        if any(word in query_lower for word in navigational_words):
            return 1.0
        
        # Transactional indicators
        transactional_words = ['buy', 'price', 'cheap', 'discount', 'deal', 'order', 
                              'shipping', 'coupon', 'sale', 'shop', 'purchase']
        if any(word in query_lower for word in transactional_words):
            return 0.0
        
        # Informational indicators
        informational_words = ['how', 'what', 'why', 'when', 'where', 'who',
                              'guide', 'tutorial', 'learn', 'definition', 'meaning',
                              'vs', 'versus', 'compare', 'review', 'best']
        if any(word in query_lower for word in informational_words):
            return 0.5
        
        # Default: slightly informational based on length
        if len(query.split()) > 3:
            return 0.6
        
        return 0.5
    
    def extract_domain_features(self, domain: str) -> np.ndarray:
        """
        Extract simple features from domain name.
        Returns array of [is_com, is_org, is_gov, is_edu, domain_length_normalized]
        """
        domain_lower = domain.lower().replace('www.', '')
        
        tld = domain_lower.split('.')[-1] if '.' in domain_lower else ''
        domain_name = domain_lower.split('.')[0] if '.' in domain_lower else domain_lower
        
        features = np.array([
            1.0 if tld == 'com' else 0.0,
            1.0 if tld == 'org' else 0.0,
            1.0 if tld in ['gov', 'edu'] else 0.0,
            min(len(domain_name) / 20.0, 1.0),  # normalized length
        ])
        
        return features
    
    def build_context(self, 
                     query: str,
                     result: dict,
                     carbon_score: float,
                     original_position: int,
                     avg_carbon_all_results: float,
                     query_features: dict = None) -> np.ndarray:
        """
        Build full context vector for a single result in a SERP.
        
        Args:
            query: Search query string
            result: Dict with result data (domain, title, etc.)
            carbon_score: Carbon score for this result's domain
            original_position: Original rank position (0-indexed)
            avg_carbon_all_results: Average carbon score across all results
            query_features: Optional additional query features
        
        Returns:
            Context vector (context_dim,)
        """
        context_parts = []
        
        # 1. Query type (0=transactional, 0.5=informational, 1=navigational)
        context_parts.append(self.classify_query_type(query))
        
        # 2. Carbon score for this result
        context_parts.append(carbon_score)
        
        # 3. How this result's carbon compares to average
        context_parts.append(carbon_score - avg_carbon_all_results)
        
        # 4. Original position (normalized 0-1, 0=top)
        context_parts.append(original_position / 10.0)
        
        # 5-8. Domain features
        domain = result.get('domain', '')
        domain_features = self.extract_domain_features(domain)
        context_parts.extend(domain_features)
        
        context = np.array(context_parts)
        
        # Ensure correct dimension (pad if needed, truncate if too long)
        target_dim = self.config.get('context_dim', 8)
        if len(context) < target_dim:
            context = np.pad(context, (0, target_dim - len(context)), 'constant')
        
        return context[:target_dim]
    
    def build_batch_contexts(self,
                            query: str,
                            results: list,
                            carbon_scores: Dict[str, float],
                            query_features: dict = None) -> List[np.ndarray]:
        """
        Build context vectors for all results on a SERP.
        """
        avg_carbon = np.mean([carbon_scores.get(r.get('domain', ''), 0.5) for r in results])
        
        contexts = []
        for i, result in enumerate(results):
            domain = result.get('domain', '')
            carbon = carbon_scores.get(domain, 0.5)
            
            ctx = self.build_context(
                query=query,
                result=result,
                carbon_score=carbon,
                original_position=i,
                avg_carbon_all_results=avg_carbon,
                query_features=query_features
            )
            contexts.append(ctx)
        
        return contexts