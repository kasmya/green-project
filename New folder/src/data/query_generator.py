"""
Query Generator for Research Dataset
=====================================
Generates 1000 diverse queries across 5 intent categories.
Ensures variety in topics, lengths, and search intents.
"""

import json
import os
import random
from typing import List, Dict


# ============================================================================
# Query Templates by Category
# ============================================================================

NAVIGATIONAL_TEMPLATES = [
    # Direct website searches
    "{brand} official website",
    "{brand} login",
    "{brand} homepage",
    "www {brand} com",
    "{brand} .com",
    "go to {brand}",
    "{brand} website",
    "{brand} sign in",
    
    # Specific pages
    "{brand} careers",
    "{brand} investor relations",
    "{brand} annual report",
    "{brand} sustainability report",
    "{brand} contact",
    "{brand} support",
    "{brand} pricing",
    "{brand} documentation",
    "{brand} api",
    "{brand} github",
    "{brand} blog",
    "{brand} about",
]

INFORMATIONAL_TEMPLATES = [
    # How-to queries
    "how to {green_action}",
    "how does {green_tech} work",
    "how to reduce {waste_type}",
    "how to start {green_habit}",
    "how to measure {green_metric}",
    
    # What is queries
    "what is {green_concept}",
    "what are {green_tech_plural}",
    "what causes {env_problem}",
    "what is the best {green_product}",
    
    # Why queries
    "why is {green_concept} important",
    "why should we {green_action}",
    "why are {green_tech_plural} better",
    
    # General information
    "benefits of {green_concept}",
    "advantages of {green_tech}",
    "disadvantages of {env_problem}",
    "{green_concept} explained",
    "guide to {green_action}",
    "introduction to {green_concept}",
    "history of {green_movement}",
    "future of {green_tech}",
    "{green_concept} statistics {current_year}",
    "{green_concept} facts",
    "{green_concept} vs {traditional_alternative}",
    "difference between {green_concept} and {traditional_alternative}",
    "examples of {green_concept}",
    "types of {green_tech_plural}",
]

TRANSACTIONAL_TEMPLATES = [
    # Buy queries
    "buy {green_product} online",
    "best {green_product} to buy",
    "where to buy {green_product}",
    "{green_product} for sale",
    "cheap {green_product}",
    "affordable {green_product}",
    "discount {green_product}",
    "{green_product} coupon",
    "{green_product} deals",
    "order {green_product}",
    
    # Comparison queries
    "best {green_product} {current_year}",
    "top 10 {green_product}",
    "{green_product} reviews",
    "{green_product} comparison",
    "{brand} vs {competitor}",
    "best rated {green_product}",
    "budget {green_product}",
    
    # Price queries
    "{green_product} price",
    "how much does {green_product} cost",
    "{green_product} cost comparison",
    "cheapest {green_product}",
    
    # Service queries
    "{green_service} near me",
    "hire {green_service}",
    "find {green_service}",
    "best {green_service} company",
]

LOCAL_TEMPLATES = [
    "farmers market near me",
    "organic grocery store near me",
    "recycling center near me",
    "electric vehicle charging station near me",
    "bike repair shop near me",
    "community garden near me",
    "composting facility near me",
    "solar panel installer near me",
    "zero waste store near me",
    "eco friendly restaurant near me",
    "thrift store near me",
    "local organic farm near me",
    "green energy provider near me",
    "environmental volunteer near me",
    "sustainable salon near me",
    "plant nursery near me",
    "bulk food store near me",
    "repair cafe near me",
    "bike share near me",
    "public transit near me",
]

ACADEMIC_TEMPLATES = [
    # Research queries
    "{green_concept} research paper",
    "{green_concept} literature review",
    "{green_concept} meta analysis",
    "recent advances in {green_concept}",
    "state of the art {green_concept}",
    
    # Methodology
    "life cycle assessment methodology",
    "carbon footprint calculation methods",
    "environmental impact assessment framework",
    "sustainability metrics and indicators",
    "carbon accounting standards",
    
    # Specific research areas
    "renewable energy integration challenges",
    "circular economy implementation",
    "sustainable supply chain management",
    "green building certification comparison",
    "carbon capture and storage technologies",
    "sustainable agriculture practices",
    "biodiversity conservation strategies",
    "climate change mitigation policies",
    "environmental justice framework",
    "corporate sustainability reporting standards",
    
    # Data and statistics
    "global carbon emissions dataset",
    "renewable energy statistics {current_year}",
    "deforestation rates by country",
    "ocean acidification data",
    "global temperature anomaly data",
]


# ============================================================================
# Fill-in Values
# ============================================================================

FILL_VALUES = {
    "brand": [
        "patagonia", "allbirds", "ecosia", "tesla", "beyond meat",
        "seventh generation", "method", "dr bronners", "klean kanteen",
        "tentree", "ecostyle", "green toys", "etiko", "outerknown",
        "reformation", "veja", "nudie jeans", "pact", "thought clothing",
        "ecoalf", "manduka", "prana", "cotopaxi", "hydro flask",
        "who gives a crap", "blueland", "grove collaborative", "dropps",
        "asana", "avocado mattress", "brooklinen", "coyuchi", "ettitude",
    ],
    "competitor": [
        "nike", "amazon basics", "walmart brand", "conventional alternative",
        "traditional brand", "mainstream option", "standard product",
    ],
    "green_action": [
        "reduce plastic waste", "compost at home", "save water",
        "conserve energy", "recycle properly", "start a garden",
        "go zero waste", "reduce carbon footprint", "shop sustainably",
        "use less electricity", "conserve biodiversity", "protect pollinators",
        "reduce food waste", "harvest rainwater", "use public transport",
    ],
    "green_tech": [
        "solar panel", "wind turbine", "electric vehicle", "heat pump",
        "geothermal energy", "biomass energy", "tidal power",
        "green hydrogen", "battery storage", "smart grid",
    ],
    "green_tech_plural": [
        "solar panels", "wind turbines", "electric vehicles", "heat pumps",
        "geothermal systems", "battery storage systems", "smart meters",
        "LED lights", "energy efficient appliances", "composting toilets",
    ],
    "waste_type": [
        "plastic waste", "food waste", "electronic waste", "textile waste",
        "paper waste", "metal waste", "glass waste", "organic waste",
        "hazardous waste", "construction waste",
    ],
    "green_habit": [
        "composting", "recycling", "zero waste lifestyle", "minimalism",
        "sustainable diet", "plant based eating", "conscious consumption",
        "urban gardening", "bike commuting", "solar cooking",
    ],
    "green_metric": [
        "carbon footprint", "water footprint", "ecological footprint",
        "energy efficiency", "waste diversion rate", "recycling rate",
        "sustainability score", "ESG rating", "circularity index",
    ],
    "green_concept": [
        "renewable energy", "sustainable development", "carbon neutrality",
        "circular economy", "green building", "sustainable agriculture",
        "biodiversity", "climate resilience", "environmental justice",
        "sustainable transport", "clean energy", "regenerative farming",
        "permaculture", "agroforestry", "blue economy",
        "sustainable fashion", "ethical consumerism", "fair trade",
        "carbon offsetting", "net zero emissions",
    ],
    "env_problem": [
        "climate change", "global warming", "deforestation", "ocean pollution",
        "air pollution", "water scarcity", "soil degradation",
        "biodiversity loss", "plastic pollution", "coral bleaching",
        "species extinction", "desertification", "acid rain",
        "ozone depletion", "microplastic contamination",
    ],
    "green_movement": [
        "environmental movement", "conservation movement", "climate movement",
        "zero waste movement", "permaculture movement", "green peace",
        "earth day", "sustainability movement", "organic movement",
    ],
    "traditional_alternative": [
        "fossil fuels", "conventional agriculture", "fast fashion",
        "single use plastic", "gas vehicles", "coal power",
        "industrial farming", "landfill waste", "chemical pesticides",
    ],
    "green_product": [
        "solar panel", "electric car", "bamboo toothbrush", "reusable water bottle",
        "compostable phone case", "organic cotton t-shirt", "LED light bulb",
        "smart thermostat", "rainwater collection barrel", "solar charger",
        "beeswax wrap", "metal straw", "cloth shopping bag", "plant based detergent",
        "energy efficient refrigerator", "eco friendly laptop", "sustainable shoes",
        "recycled backpack", "biodegradable soap", "natural sunscreen",
    ],
    "green_service": [
        "solar panel installation", "energy audit", "green cleaning service",
        "organic landscaping", "sustainable architecture", "environmental consulting",
        "carbon offset provider", "green web hosting", "eco friendly pest control",
        "sustainable catering",
    ],
    "current_year": ["2024", "2025", "2026"],
}


# ============================================================================
# Generator
# ============================================================================

class QueryGenerator:
    """Generates diverse search queries for research dataset."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.queries = set()  # Avoid duplicates
    
    def _fill_template(self, template: str) -> str:
        """Replace placeholders with random values."""
        filled = template
        for key, values in FILL_VALUES.items():
            if "{" + key + "}" in filled:
                filled = filled.replace("{" + key + "}", random.choice(values))
        return filled
    
    def _generate_category(self, templates: List[str], count: int, category: str) -> List[Dict]:
        """Generate queries for a specific category."""
        queries = []
        attempts = 0
        max_attempts = count * 10
        
        while len(queries) < count and attempts < max_attempts:
            template = random.choice(templates)
            query = self._fill_template(template)
            
            # Normalize
            query = query.strip().lower()
            query = " ".join(query.split())  # Remove extra spaces
            
            if query not in self.queries and len(query) > 5:
                self.queries.add(query)
                queries.append({
                    "query": query,
                    "category": category,
                    "id": f"{category}_{len(queries):04d}",
                })
            
            attempts += 1
        
        return queries
    
    def generate(self, queries_per_category: int = 200) -> List[Dict]:
        """
        Generate full query dataset.
        
        Returns list of {query, category, id}
        """
        all_queries = []
        
        categories = [
            ("navigational", NAVIGATIONAL_TEMPLATES),
            ("informational", INFORMATIONAL_TEMPLATES),
            ("transactional", TRANSACTIONAL_TEMPLATES),
            ("local", LOCAL_TEMPLATES),
            ("academic", ACADEMIC_TEMPLATES),
        ]
        
        for category, templates in categories:
            print(f"Generating {queries_per_category} {category} queries...")
            category_queries = self._generate_category(
                templates, queries_per_category, category
            )
            all_queries.extend(category_queries)
            print(f"  Generated {len(category_queries)} unique queries")
        
        # Shuffle
        random.shuffle(all_queries)
        
        # Add sequential IDs
        for i, q in enumerate(all_queries):
            q['id'] = f"query_{i:04d}"
        
        return all_queries
    
    def save(self, queries: List[Dict], filepath: str = "data/queries.json"):
        """Save queries to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(queries, f, indent=2)
        
        # Also save a simple text file for the collector
        txt_path = filepath.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            for q in queries:
                f.write(q['query'] + '\n')
        
        print(f"Saved {len(queries)} queries to {filepath}")
        print(f"Saved text version to {txt_path}")


# ============================================================================
# Statistics
# ============================================================================

def compute_query_stats(queries: List[Dict]) -> Dict:
    """Compute statistics about generated queries."""
    stats = {
        "total_queries": len(queries),
        "by_category": {},
        "avg_length": 0,
        "length_distribution": {},
    }
    
    lengths = []
    for q in queries:
        category = q['category']
        if category not in stats['by_category']:
            stats['by_category'][category] = 0
        stats['by_category'][category] += 1
        
        length = len(q['query'].split())
        lengths.append(length)
        
        length_bin = f"{length // 2 * 2}-{(length // 2 * 2) + 1} words"
        if length_bin not in stats['length_distribution']:
            stats['length_distribution'][length_bin] = 0
        stats['length_distribution'][length_bin] += 1
    
    stats['avg_length'] = sum(lengths) / len(lengths)
    stats['min_length'] = min(lengths)
    stats['max_length'] = max(lengths)
    
    return stats


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    generator = QueryGenerator(seed=42)
    
    # Generate 1000 queries (200 per category)
    queries = generator.generate(queries_per_category=200)
    
    # Save
    generator.save(queries, "data/queries.json")
    
    # Stats
    stats = compute_query_stats(queries)
    print(f"\n{'='*50}")
    print("QUERY DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Average length: {stats['avg_length']:.1f} words")
    print(f"Length range: {stats['min_length']}-{stats['max_length']} words")
    print(f"\nBy Category:")
    for cat, count in stats['by_category'].items():
        print(f"  {cat}: {count}")