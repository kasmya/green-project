"""
Dataset Builder
===============
Creates train/val/test splits from scored dataset.
Ensures query-level splitting (no result leakage).
"""

import json
import os
import random
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


class DatasetBuilder:
    """
    Builds train/val/test splits for research evaluation.
    """
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "data/processed",
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        print(f"Loaded dataset: {len(self.dataset['queries'])} queries")
    
    def _stratify_by_category(self, queries: List[Dict]) -> Dict[str, List[Dict]]:
        """Group queries by category (if available) or by query type."""
        by_category = {}
        
        for q in queries:
            # Try to get category from metadata
            category = q.get('category', 'unknown')
            
            # If no category, infer from query characteristics
            if category == 'unknown':
                query = q['query'].lower()
                if any(w in query for w in ['buy', 'price', 'shop', 'deal']):
                    category = 'transactional'
                elif any(w in query for w in ['how', 'what', 'why', 'guide']):
                    category = 'informational'
                elif any(w in query for w in ['login', 'official', 'website']):
                    category = 'navigational'
                elif any(w in query for w in ['near me', 'local']):
                    category = 'local'
                else:
                    category = 'general'
            
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(q)
        
        return by_category
    
    def _split_queries(self, queries: List[Dict]) -> Tuple[List, List, List]:
        """
        Split queries into train/val/test.
        Maintains category distribution if possible.
        """
        random.seed(self.seed)
        random.shuffle(queries)
        
        n = len(queries)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        train = queries[:n_train]
        val = queries[n_train:n_train + n_val]
        test = queries[n_train + n_val:]
        
        return train, val, test
    
    def _compute_split_stats(self, queries: List[Dict], name: str) -> Dict:
        """Compute statistics for a split."""
        stats = {
            'name': name,
            'num_queries': len(queries),
            'num_results': sum(q.get('num_results', len(q.get('results', []))) for q in queries),
        }
        
        # Carbon stats
        carbons = []
        relevances = []
        for q in queries:
            for r in q.get('results', []):
                if 'carbon_score' in r:
                    carbons.append(r['carbon_score'])
                if 'relevance_score' in r:
                    relevances.append(r['relevance_score'])
        
        if carbons:
            stats['mean_carbon'] = float(np.mean(carbons))
            stats['std_carbon'] = float(np.std(carbons))
        if relevances:
            stats['mean_relevance'] = float(np.mean(relevances))
            stats['std_relevance'] = float(np.std(relevances))
        
        return stats
    
    def build(self) -> Dict:
        """Build and save dataset splits."""
        queries = self.dataset['queries']
        
        print(f"\n{'='*60}")
        print(f"BUILDING DATASET SPLITS")
        print(f"{'='*60}")
        print(f"Train ratio: {self.train_ratio}")
        print(f"Val ratio:   {self.val_ratio}")
        print(f"Test ratio:  {self.test_ratio}")
        print(f"Random seed: {self.seed}")
        
        # Try stratified split
        by_category = self._stratify_by_category(queries)
        print(f"\nCategories found: {list(by_category.keys())}")
        
        if len(by_category) > 1:
            # Stratified split
            print("Using stratified split by category")
            train_queries, val_queries, test_queries = [], [], []
            
            for category, cat_queries in by_category.items():
                n = len(cat_queries)
                n_train = int(n * self.train_ratio)
                n_val = int(n * self.val_ratio)
                
                random.shuffle(cat_queries)
                train_queries.extend(cat_queries[:n_train])
                val_queries.extend(cat_queries[n_train:n_train + n_val])
                test_queries.extend(cat_queries[n_train + n_val:])
                
                print(f"  {category}: {n} → {n_train}/{n_val}/{n - n_train - n_val}")
        else:
            # Simple random split
            print("Using simple random split")
            train_queries, val_queries, test_queries = self._split_queries(queries)
        
        # Shuffle each split
        random.shuffle(train_queries)
        random.shuffle(val_queries)
        random.shuffle(test_queries)
        
        # Compute stats
        train_stats = self._compute_split_stats(train_queries, 'train')
        val_stats = self._compute_split_stats(val_queries, 'val')
        test_stats = self._compute_split_stats(test_queries, 'test')
        
        # Build output
        splits = {
            'metadata': {
                **self.dataset.get('metadata', {}),
                'split_info': {
                    'train_ratio': self.train_ratio,
                    'val_ratio': self.val_ratio,
                    'test_ratio': self.test_ratio,
                    'seed': self.seed,
                    'created_at': str(np.datetime64('now')),
                }
            },
            'train': {
                'queries': train_queries,
                'stats': train_stats,
            },
            'val': {
                'queries': val_queries,
                'stats': val_stats,
            },
            'test': {
                'queries': test_queries,
                'stats': test_stats,
            },
            'carbon_scores': self.dataset.get('carbon_scores', {}),
        }
        
        # Save
        output_file = os.path.join(self.output_dir, "dataset_splits.json")
        with open(output_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        # Also save individual splits
        for split_name, split_data in [('train', train_queries), 
                                        ('val', val_queries), 
                                        ('test', test_queries)]:
            split_file = os.path.join(self.output_dir, f"{split_name}.json")
            with open(split_file, 'w') as f:
                json.dump({
                    'queries': split_data,
                    'carbon_scores': self.dataset.get('carbon_scores', {}),
                }, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"DATASET SPLITS SAVED")
        print(f"{'='*60}")
        print(f"\n{'Split':<10} {'Queries':<10} {'Results':<10} {'Mean Carbon':<12} {'Mean Relevance':<15}")
        print("-" * 60)
        for stats in [train_stats, val_stats, test_stats]:
            print(f"{stats['name']:<10} {stats['num_queries']:<10} {stats['num_results']:<10} "
                  f"{stats.get('mean_carbon', 0):<12.4f} {stats.get('mean_relevance', 0):<15.4f}")
        
        print(f"\nOutput: {output_file}")
        
        return splits


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build dataset splits')
    parser.add_argument('--dataset', type=str,
                       default='data/raw/serp_dataset.json',
                       help='Path to scored dataset')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed',
                       help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    builder = DatasetBuilder(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        seed=args.seed,
    )
    
    splits = builder.build()