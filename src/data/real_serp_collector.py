"""
Real SERP Data Collector
========================
Collects real Google search results and labels domains with Green Web Foundation API.
Outputs a clean JSON dataset for offline bandit training/evaluation.
"""

import json
import time
import os
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
import yaml
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CollectorConfig:
    """Configuration for data collection."""
    # SerpAPI settings (paid but reliable)
    serpapi_key: str = ""  # Get from https://serpapi.com/
    
    # Or use manual scraping (free but fragile)
    use_manual_scraping: bool = True
    
    # Queries to collect
    queries_file: str = "data/queries.txt"
    
    # Output
    output_dir: str = "data/collected"
    output_file: str = "data/collected/serp_dataset.json"
    
    # Rate limiting
    delay_between_queries: float = 2.0  # seconds
    delay_between_api_calls: float = 1.0  # seconds for Green Web API
    
    # Green Web API
    green_web_api_url: str = "https://api.thegreenwebfoundation.org/api/v3/greencheck/"
    
    # Cache
    cache_dir: str = "data/cache"
    carbon_cache_file: str = "data/cache/carbon_scores.json"


# ============================================================================
# Query List
# ============================================================================

DEFAULT_QUERIES = [
    # Navigational queries (high precision needed)
    "facebook login",
    "gmail",
    "youtube",
    "github",
    "wikipedia",
    
    # Informational queries (learning intent)
    "how to reduce carbon footprint",
    "climate change facts 2024",
    "what is renewable energy",
    "best way to learn python",
    "history of internet",
    "how do solar panels work",
    "effects of global warming",
    "what is machine learning",
    "how to compost at home",
    "benefits of electric cars",
    
    # Transactional queries (buying intent)
    "best laptop 2024",
    "cheap flights to europe",
    "buy running shoes online",
    "best phone under 500",
    "sustainable clothing brands",
    "eco friendly products amazon",
    "electric car deals",
    "used books online",
    "organic food delivery",
    "solar panel installation cost",
    
    # Mixed intent
    "tesla model 3 review",
    "patagonia vs north face",
    "green hosting providers",
    "carbon neutral companies",
    "best vpn for privacy",
    "is amazon sustainable",
    "apple environmental report",
    "fast fashion environmental impact",
    "renewable energy stocks",
    "how to invest in green energy",
    
    # Local intent
    "farmers market near me",
    "recycling center hours",
    "electric vehicle charging stations",
    "organic grocery store",
    "bike shop repair",
    
    # News
    "climate news today",
    "environmental policy 2024",
    "cop29 results",
    "carbon tax latest",
    "green new deal update",
    
    # Academic
    "life cycle assessment methodology",
    "carbon accounting standards",
    "renewable energy research papers",
    "sustainability metrics",
    "environmental impact assessment",
]


def create_default_queries_file(filepath: str = "data/queries.txt"):
    """Create a default queries file if none exists."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for query in DEFAULT_QUERIES:
            f.write(query + '\n')
    print(f"Created default queries file with {len(DEFAULT_QUERIES)} queries at {filepath}")


# ============================================================================
# SERP Collector - Manual Scraping (Free)
# ============================================================================

class ManualSERPCollector:
    """
    Collects SERPs by scraping Google HTML.
    WARNING: Fragile. Google may block this. Use SerpAPI for reliability.
    """
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def search(self, query: str, num_results: int = 10) -> Optional[List[Dict]]:
        """
        Scrape Google search results for a query.
        Returns list of {position, title, url, domain, snippet}
        """
        from bs4 import BeautifulSoup
        
        url = f"https://www.google.com/search?q={requests.utils.quote(query)}&num={num_results}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find all search result divs
            # Note: These selectors change frequently. Update as needed.
            result_divs = soup.select('div.g, div[data-sokoban-container]')
            
            position = 0
            for div in result_divs:
                # Extract title
                title_elem = div.select_one('h3')
                if not title_elem:
                    continue
                title = title_elem.get_text()
                
                # Extract URL
                link_elem = div.select_one('a[href^="http"]')
                if not link_elem:
                    continue
                url = link_elem['href']
                
                # Extract domain
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                
                # Extract snippet
                snippet_elem = div.select_one('div.VwiC3b, span.st, div[data-sncf]')
                snippet = snippet_elem.get_text() if snippet_elem else ""
                
                results.append({
                    'position': position,
                    'title': title,
                    'url': url,
                    'domain': domain,
                    'snippet': snippet,
                })
                
                position += 1
                if position >= num_results:
                    break
            
            time.sleep(self.delay)
            return results
            
        except Exception as e:
            print(f"  Error scraping '{query}': {e}")
            return None


# ============================================================================
# SERP Collector - SerpAPI (Paid, Reliable)
# ============================================================================

class SerpAPICollector:
    """
    Collects SERPs using SerpAPI.
    Requires API key from https://serpapi.com/
    Free tier: 100 searches/month
    """
    
    BASE_URL = "https://serpapi.com/search"
    
    def __init__(self, api_key: str, delay: float = 1.0):
        self.api_key = api_key
        self.delay = delay
    
    def search(self, query: str, num_results: int = 10) -> Optional[List[Dict]]:
        """Get organic results from Google via SerpAPI."""
        params = {
            'api_key': self.api_key,
            'engine': 'google',
            'q': query,
            'num': num_results,
            'hl': 'en',
            'gl': 'us',
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for position, result in enumerate(data.get('organic_results', [])[:num_results]):
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
            print(f"  SerpAPI error for '{query}': {e}")
            return None
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain


# ============================================================================
# Carbon Labeler
# ============================================================================

class CarbonLabeler:
    """
    Labels domains with carbon scores using Green Web Foundation API.
    Includes caching to avoid repeated API calls.
    """
    
    def __init__(self, api_url: str, cache_file: str, delay: float = 1.0):
        self.api_url = api_url
        self.cache_file = cache_file
        self.delay = delay
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, dict]:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _is_cache_valid(self, domain: str) -> bool:
        if domain not in self.cache:
            return False
        age = time.time() - self.cache[domain]['timestamp']
        return age < 86400  # 24 hours
    
    def _query_api(self, domain: str) -> Optional[float]:
        """Returns 1.0 if green, 0.0 if not, None if error."""
        try:
            response = requests.get(f"{self.api_url}{domain}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return 1.0 if data.get('green', False) else 0.0
            return None
        except Exception:
            return None
    
    def get_score(self, domain: str) -> float:
        """Get carbon score for a domain."""
        domain = domain.lower().replace('www.', '')
        
        # Check cache
        if self._is_cache_valid(domain):
            return self.cache[domain]['score']
        
        # Query API
        score = self._query_api(domain)
        
        if score is not None:
            self.cache[domain] = {
                'score': score,
                'source': 'api',
                'timestamp': time.time()
            }
            self._save_cache()
            time.sleep(self.delay)
            return score
        
        # Fallback heuristic
        score = self._fallback_heuristic(domain)
        self.cache[domain] = {
            'score': score,
            'source': 'heuristic',
            'timestamp': time.time()
        }
        self._save_cache()
        return score
    
    def _fallback_heuristic(self, domain: str) -> float:
        """Conservative heuristic when API is unavailable."""
        # Known green domains
        green_domains = [
            'ecosia.org', 'treehugger.com', 'ethicalconsumer.org',
            'goodonyou.eco', 'earth911.com', 'grist.org',
            '350.org', 'greenpeace.org', 'sierraclub.org',
            'worldwildlife.org', 'nature.org', 'rainforest-alliance.org',
        ]
        
        if domain in green_domains:
            return 0.9
        
        # Green TLDs
        green_tlds = ['.org', '.gov', '.edu', '.eu']
        for tld in green_tlds:
            if domain.endswith(tld):
                return 0.6
        
        # Green tech companies
        green_tech = ['google.com', 'microsoft.com', 'apple.com', 
                      'github.com', 'netlify.com', 'vercel.com']
        if domain in green_tech:
            return 0.8
        
        return 0.3  # Conservative default
    
    def label_batch(self, domains: List[str]) -> Dict[str, float]:
        """Label multiple domains."""
        unique_domains = list(set(domains))
        scores = {}
        for domain in tqdm(unique_domains, desc="Labeling domains"):
            scores[domain] = self.get_score(domain)
        return scores


# ============================================================================
# Main Collector
# ============================================================================

class SERPDatasetCollector:
    """
    Orchestrates full data collection pipeline.
    """
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        
        # Initialize collector
        if config.use_manual_scraping:
            self.serp_collector = ManualSERPCollector(delay=config.delay_between_queries)
            print("Using manual Google scraping (fragile, may break)")
        else:
            if not config.serpapi_key:
                raise ValueError("SerpAPI key required when use_manual_scraping=False")
            self.serp_collector = SerpAPICollector(
                api_key=config.serpapi_key,
                delay=config.delay_between_queries
            )
            print("Using SerpAPI (reliable)")
        
        # Initialize carbon labeler
        self.carbon_labeler = CarbonLabeler(
            api_url=config.green_web_api_url,
            cache_file=config.carbon_cache_file,
            delay=config.delay_between_api_calls
        )
    
    def load_queries(self) -> List[str]:
        """Load queries from file."""
        if not os.path.exists(self.config.queries_file):
            create_default_queries_file(self.config.queries_file)
        
        with open(self.config.queries_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(queries)} queries from {self.config.queries_file}")
        return queries
    
    def collect(self, queries: Optional[List[str]] = None) -> Dict:
        """
        Collect SERP data and carbon scores for all queries.
        
        Returns:
            Dataset dict with:
            - metadata: collection info
            - queries: list of query results
            - carbon_scores: domain → carbon score mapping
        """
        if queries is None:
            queries = self.load_queries()
        
        dataset = {
            'metadata': {
                'collected_at': datetime.now().isoformat(),
                'num_queries': len(queries),
                'collector': 'serpapi' if not self.config.use_manual_scraping else 'manual',
            },
            'queries': [],
            'all_domains': set(),
        }
        
        # Step 1: Collect SERPs
        print(f"\n{'='*60}")
        print(f"STEP 1: Collecting SERPs for {len(queries)} queries")
        print(f"{'='*60}\n")
        
        successful = 0
        for query in tqdm(queries, desc="Collecting SERPs"):
            results = self.serp_collector.search(query, num_results=10)
            
            if results:
                dataset['queries'].append({
                    'query': query,
                    'results': results,
                    'num_results': len(results),
                })
                for r in results:
                    dataset['all_domains'].add(r['domain'])
                successful += 1
            else:
                print(f"  Failed: '{query}'")
        
        dataset['metadata']['successful_queries'] = successful
        dataset['all_domains'] = list(dataset['all_domains'])
        
        print(f"\n✓ Collected {successful}/{len(queries)} SERPs successfully")
        print(f"✓ Found {len(dataset['all_domains'])} unique domains")
        
        # Step 2: Label domains with carbon scores
        print(f"\n{'='*60}")
        print(f"STEP 2: Labeling {len(dataset['all_domains'])} domains with Green Web API")
        print(f"{'='*60}\n")
        
        carbon_scores = self.carbon_labeler.label_batch(dataset['all_domains'])
        dataset['carbon_scores'] = carbon_scores
        
        # Step 3: Add carbon scores to results
        for query_data in dataset['queries']:
            for result in query_data['results']:
                result['carbon_score'] = carbon_scores.get(
                    result['domain'], 
                    self.carbon_labeler._fallback_heuristic(result['domain'])
                )
        
        # Step 4: Compute relevance scores (position-based proxy)
        for query_data in dataset['queries']:
            for result in query_data['results']:
                # Simple relevance proxy: inverse of position
                result['relevance_score'] = 1.0 - (result['position'] / 10.0)
        
        # Step 5: Save dataset
        os.makedirs(os.path.dirname(self.config.output_file), exist_ok=True)
        # Convert set to list for JSON serialization
        dataset['all_domains'] = list(dataset['all_domains'])
        with open(self.config.output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"DATASET SAVED")
        print(f"{'='*60}")
        print(f"File: {self.config.output_file}")
        print(f"Queries: {successful}")
        print(f"Unique domains: {len(dataset['all_domains'])}")
        print(f"Carbon scores cached: {len(carbon_scores)}")
        
        # Summary statistics
        scores = list(carbon_scores.values())
        green_count = sum(1 for s in scores if s >= 0.5)
        print(f"\nCarbon Score Distribution:")
        print(f"  Green (≥0.5): {green_count} ({100*green_count/len(scores):.1f}%)")
        print(f"  Not green (<0.5): {len(scores)-green_count} ({100*(len(scores)-green_count)/len(scores):.1f}%)")
        print(f"  Mean score: {sum(scores)/len(scores):.3f}")
        
        return dataset


# ============================================================================
# Main
# ============================================================================

def main():
    """Collect real SERP data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect real SERP data with carbon scores')
    parser.add_argument('--serpapi-key', type=str, default='', 
                       help='SerpAPI key (omit for manual scraping)')
    parser.add_argument('--queries-file', type=str, default='data/queries.txt',
                       help='File with one query per line')
    parser.add_argument('--output', type=str, default='data/collected/serp_dataset.json',
                       help='Output JSON file')
    parser.add_argument('--num-queries', type=int, default=50,
                       help='Number of queries to collect (uses first N from file)')
    
    args = parser.parse_args()
    
    config = CollectorConfig(
        serpapi_key=args.serpapi_key,
        use_manual_scraping=(args.serpapi_key == ''),
        queries_file=args.queries_file,
        output_file=args.output,
    )
    
    collector = SERPDatasetCollector(config)
    queries = collector.load_queries()[:args.num_queries]
    dataset = collector.collect(queries)
    
    return dataset


if __name__ == "__main__":
    dataset = main()