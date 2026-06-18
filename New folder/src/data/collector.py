"""
SERP Collector for Research Dataset
====================================
Collects Google SERPs via SerpAPI and labels with Green Web API.
Designed for 1000 queries with rate limiting and caching.
"""

import json
import os
import time
import hashlib
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

class CollectorConfig:
    """Configuration for SERP collection."""
    
    def __init__(
        self,
        serpapi_key: str,
        queries_file: str = "data/queries.txt",
        output_dir: str = "data/raw",
        cache_dir: str = "data/cache",
        green_web_api_url: str = "https://api.thegreenwebfoundation.org/api/v3/greencheck/",
        results_per_query: int = 10,
        delay_serpapi: float = 2.0,     # Seconds between SerpAPI calls
        delay_green_api: float = 1.0,   # Seconds between Green Web API calls
        batch_size: int = 100,          # Save progress every N queries
    ):
        self.serpapi_key = serpapi_key
        self.queries_file = queries_file
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.green_web_api_url = green_web_api_url
        self.results_per_query = results_per_query
        self.delay_serpapi = delay_serpapi
        self.delay_green_api = delay_green_api
        self.batch_size = batch_size


# ============================================================================
# Carbon Labeler
# ============================================================================

class CarbonLabeler:
    """Labels domains with Green Web Foundation API scores."""
    
    def __init__(self, api_url: str, cache_dir: str, delay: float = 1.0):
        self.api_url = api_url
        self.cache_dir = cache_dir
        self.delay = delay
        
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "carbon_scores.json")
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _is_cache_valid(self, domain: str) -> bool:
        if domain not in self.cache:
            return False
        age = time.time() - self.cache[domain].get('timestamp', 0)
        return age < 86400 * 7  # 7 days
    
    def _query_api(self, domain: str) -> Optional[float]:
        """Query Green Web Foundation API."""
        try:
            response = requests.get(f"{self.api_url}{domain}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return 1.0 if data.get('green', False) else 0.0
            return None
        except Exception:
            return None
    
    def _fallback_heuristic(self, domain: str) -> float:
        """Fallback when API unavailable."""
        domain_lower = domain.lower()
        
        # Known green domains
        known_green = [
            'ecosia.org', 'allbirds.com', 'patagonia.com', 'treehugger.com',
            'ethicalconsumer.org', 'goodonyou.eco', 'earth911.com', 'grist.org',
            'greenpeace.org', 'sierraclub.org', 'worldwildlife.org', 'nature.org',
            '350.org', 'rainforest-alliance.org', 'carbonbrief.org',
            'drawdown.org', 'rmi.org', 'wri.org', 'ceres.org', 'edf.org',
            'google.com', 'microsoft.com', 'apple.com', 'github.com',
            'netlify.com', 'vercel.com', 'cloudflare.com',
        ]
        
        if domain_lower in known_green:
            return 0.9
        
        # Green TLDs
        green_tlds = ['.org', '.gov', '.edu', '.eu', '.nz', '.fi', '.se']
        for tld in green_tlds:
            if domain_lower.endswith(tld):
                return 0.6
        
        return 0.3
    
    def get_score(self, domain: str) -> float:
        """Get carbon score for domain (0 = dirty, 1 = green)."""
        domain = domain.lower().replace('www.', '')
        
        if self._is_cache_valid(domain):
            return self.cache[domain]['score']
        
        score = self._query_api(domain)
        
        if score is None:
            score = self._fallback_heuristic(domain)
            source = 'heuristic'
        else:
            source = 'api'
        
        self.cache[domain] = {
            'score': score,
            'source': source,
            'timestamp': time.time(),
        }
        self._save_cache()
        time.sleep(self.delay)
        
        return score
    
    def label_batch(self, domains: List[str]) -> Dict[str, float]:
        """Label multiple domains, using cache where possible."""
        unique_domains = list(set(domains))
        new_domains = [d for d in unique_domains if not self._is_cache_valid(d)]
        
        if new_domains:
            print(f"  Querying Green Web API for {len(new_domains)} new domains...")
            for domain in tqdm(new_domains, desc="  Carbon labeling"):
                self.get_score(domain)
        
        return {d: self.cache.get(d, {}).get('score', 0.3) for d in domains}


# ============================================================================
# SerpAPI Collector
# ============================================================================

class SerpAPICollector:
    """Collects SERPs via SerpAPI."""
    
    BASE_URL = "https://serpapi.com/search"
    
    def __init__(self, api_key: str, delay: float = 2.0):
        self.api_key = api_key
        self.delay = delay
    
    def search(self, query: str, num_results: int = 10) -> Optional[List[Dict]]:
        """Get organic results from Google."""
        params = {
            'api_key': self.api_key,
            'engine': 'google',
            'q': query,
            'num': num_results,
            'hl': 'en',
            'gl': 'us',
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = []
            organic = data.get('organic_results', [])[:num_results]
            
            for position, result in enumerate(organic):
                url = result.get('link', '')
                domain = self._extract_domain(url)
                
                results.append({
                    'position': position,
                    'title': result.get('title', ''),
                    'url': url,
                    'domain': domain,
                    'snippet': result.get('snippet', ''),
                })
            
            time.sleep(self.delay)
            return results
            
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return ""


# ============================================================================
# Main Collector
# ============================================================================

class ResearchDatasetCollector:
    """
    Collects complete research dataset:
    1. SERPs for all queries
    2. Carbon scores for all domains
    3. Saves incrementally
    """
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.serpapi = SerpAPICollector(config.serpapi_key, config.delay_serpapi)
        self.carbon_labeler = CarbonLabeler(
            config.green_web_api_url,
            config.cache_dir,
            config.delay_green_api,
        )
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    def load_queries(self) -> List[str]:
        """Load queries from file."""
        with open(self.config.queries_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _save_checkpoint(self, data: Dict, checkpoint_id: int):
        """Save intermediate results."""
        checkpoint_file = os.path.join(
            self.config.output_dir,
            f"checkpoint_{checkpoint_id:04d}.json"
        )
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def collect(self, queries: Optional[List[str]] = None, 
                start_from: int = 0) -> Dict:
        """
        Collect full dataset.
        
        Args:
            queries: List of query strings (or None to load from file)
            start_from: Index to resume from (for interrupted runs)
        
        Returns:
            Complete dataset dict
        """
        if queries is None:
            queries = self.load_queries()
        
        print(f"\n{'='*60}")
        print(f"RESEARCH DATASET COLLECTION")
        print(f"{'='*60}")
        print(f"Queries to collect: {len(queries)}")
        print(f"Results per query: {self.config.results_per_query}")
        print(f"Starting from: {start_from}")
        print(f"{'='*60}\n")
        
        # Load existing progress if any
        all_results = []
        all_domains = set()
        
        # Collect SERPs
        print("STEP 1: Collecting SERPs from Google\n")
        
        for i in tqdm(range(start_from, len(queries)), 
                      desc="Collecting SERPs",
                      initial=start_from,
                      total=len(queries)):
            query = queries[i]
            results = self.serpapi.search(query, self.config.results_per_query)
            
            if results:
                query_data = {
                    'query': query,
                    'query_id': f"query_{i:04d}",
                    'results': results,
                    'num_results': len(results),
                    'collected_at': datetime.now().isoformat(),
                }
                all_results.append(query_data)
                
                for r in results:
                    all_domains.add(r['domain'])
            else:
                print(f"\n  Warning: No results for '{query}'")
                # Save empty result to maintain indexing
                all_results.append({
                    'query': query,
                    'query_id': f"query_{i:04d}",
                    'results': [],
                    'num_results': 0,
                    'collected_at': datetime.now().isoformat(),
                })
            
            # Save checkpoint every batch_size queries
            if (i + 1) % self.config.batch_size == 0:
                checkpoint_data = {
                    'queries_collected': len(all_results),
                    'last_index': i,
                    'results': all_results,
                    'all_domains': list(all_domains),
                }
                self._save_checkpoint(checkpoint_data, i + 1)
        
        # Step 2: Carbon labeling
        print(f"\nSTEP 2: Labeling {len(all_domains)} unique domains\n")
        
        carbon_scores = self.carbon_labeler.label_batch(list(all_domains))
        
        # Step 3: Add carbon scores to results
        print("\nSTEP 3: Adding carbon scores to results\n")
        
        for query_data in tqdm(all_results, desc="Adding carbon scores"):
            for result in query_data['results']:
                result['carbon_score'] = carbon_scores.get(
                    result['domain'],
                    self.carbon_labeler._fallback_heuristic(result['domain'])
                )
        
        # Build final dataset
        dataset = {
            'metadata': {
                'collected_at': datetime.now().isoformat(),
                'total_queries': len(queries),
                'successful_queries': sum(1 for q in all_results if q['num_results'] > 0),
                'total_results': sum(q['num_results'] for q in all_results),
                'unique_domains': len(all_domains),
                'serpapi_key_used': bool(self.config.serpapi_key),
            },
            'queries': all_results,
            'all_domains': list(all_domains),
            'carbon_scores': carbon_scores,
        }
        
        # Save final dataset
        output_file = os.path.join(self.config.output_dir, "serp_dataset.json")
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"DATASET SAVED")
        print(f"{'='*60}")
        print(f"File: {output_file}")
        print(f"Queries: {dataset['metadata']['successful_queries']}/{len(queries)}")
        print(f"Results: {dataset['metadata']['total_results']}")
        print(f"Unique domains: {dataset['metadata']['unique_domains']}")
        
        # Carbon stats
        scores = list(carbon_scores.values())
        if scores:
            green = sum(1 for s in scores if s >= 0.5)
            print(f"\nCarbon Distribution:")
            print(f"  Green (≥0.5): {green} ({100*green/len(scores):.1f}%)")
            print(f"  Mean score: {sum(scores)/len(scores):.3f}")
            print(f"  Median score: {sorted(scores)[len(scores)//2]:.3f}")
        
        return dataset


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect research SERP dataset')
    parser.add_argument('--serpapi-key', type=str, required=True,
                       help='SerpAPI key')
    parser.add_argument('--queries-file', type=str, default='data/queries.txt',
                       help='File with queries (one per line)')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--num-queries', type=int, default=None,
                       help='Number of queries to collect (default: all)')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Resume from this query index')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Save checkpoint every N queries')
    
    args = parser.parse_args()
    
    config = CollectorConfig(
        serpapi_key=args.serpapi_key,
        queries_file=args.queries_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    
    collector = ResearchDatasetCollector(config)
    queries = collector.load_queries()
    
    if args.num_queries:
        queries = queries[:args.num_queries]
    
    dataset = collector.collect(queries, start_from=args.start_from)
    
    return dataset


if __name__ == "__main__":
    dataset = main()