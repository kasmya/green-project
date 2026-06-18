import hashlib
import json
import os
import time
from typing import Dict, Optional
import requests
import yaml


class CarbonResolver:
    """
    Resolves carbon scores for domains using Green Web Foundation API
    with local caching and fallback heuristics.
    """
    
    def __init__(self, config_path: str = "config.yaml", cache_dir: str = "data/cache"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.api_url = self.config['carbon_api']['green_web_url']
        self.cache_ttl = self.config['carbon_api']['cache_ttl_hours'] * 3600
        self.fallback_score = self.config['carbon_api']['fallback_score']
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = self._load_cache()
    
    def _cache_key(self, domain: str) -> str:
        return hashlib.md5(domain.encode()).hexdigest()
    
    def _cache_path(self, domain: str) -> str:
        return os.path.join(self.cache_dir, f"{self._cache_key(domain)}.json")
    
    def _load_cache(self) -> Dict:
        cache = {}
        if not os.path.exists(self.cache_dir):
            return cache
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.cache_dir, filename), 'r') as f:
                    data = json.load(f)
                    cache[data['domain']] = data
        return cache
    
    def _save_to_cache(self, domain: str, score: float, source: str):
        data = {
            'domain': domain,
            'score': score,
            'source': source,
            'timestamp': time.time()
        }
        with open(self._cache_path(domain), 'w') as f:
            json.dump(data, f)
        self.cache[domain] = data
    
    def _is_cache_valid(self, domain: str) -> bool:
        if domain not in self.cache:
            return False
        age = time.time() - self.cache[domain]['timestamp']
        return age < self.cache_ttl
    
    def _query_green_web_api(self, domain: str) -> Optional[float]:
        """Queries Green Web Foundation API. Returns 1.0 if green, 0.0 if not, None if error."""
        try:
            response = requests.get(f"{self.api_url}{domain}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return 1.0 if data.get('green', False) else 0.0
            return None
        except Exception:
            return None
    
    def _local_heuristic(self, domain: str) -> float:
        """
        Fallback heuristic when API is unavailable.
        Uses known patterns and common green hosts.
        """
        domain_lower = domain.lower()
        
        # Known green hosting providers
        green_hosts = [
            'netlify.com', 'vercel.com', 'github.io', 'gitlab.io',
            'googleapis.com', 'cloudflare.com', 'fastly.com',
            'wordpress.com', 'wix.com', 'squarespace.com'
        ]
        
        # Domains that are often on renewable energy
        green_tlds = ['.org', '.gov', '.edu', '.eu', '.nz', '.fi', '.se']
        
        # Check if it's a known green subdomain
        for host in green_hosts:
            if host in domain_lower:
                return 0.9
        
        # TLD-based heuristic
        for tld in green_tlds:
            if domain_lower.endswith(tld):
                return 0.6
        
        # Large tech companies known for green initiatives
        green_companies = ['google', 'microsoft', 'apple', 'meta', 'amazon']
        for company in green_companies:
            if company in domain_lower:
                return 0.8
        
        return self.fallback_score
    
    def get_score(self, domain: str) -> float:
        """
        Returns carbon score for a domain (0 = dirty, 1 = green).
        Checks cache first, then API, falls back to heuristic.
        """
        # Clean domain
        domain = domain.lower().strip()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check cache
        if self._is_cache_valid(domain):
            return self.cache[domain]['score']
        
        # Try API
        score = self._query_green_web_api(domain)
        
        if score is not None:
            self._save_to_cache(domain, score, 'api')
            return score
        
        # Fallback
        score = self._local_heuristic(domain)
        self._save_to_cache(domain, score, 'heuristic')
        return score
    
    def get_scores_batch(self, domains: list) -> Dict[str, float]:
        """Resolves carbon scores for multiple domains."""
        return {domain: self.get_score(domain) for domain in domains}