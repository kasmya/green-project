"""
Relevance Scorer using Sentence-BERT
=====================================
Computes semantic similarity between queries and search results.
Uses all-MiniLM-L6-v2 for efficient inference.
"""

import json
import os
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


class RelevanceScorer:
    """
    Scores relevance of search results using Sentence-BERT.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 batch_size: int = 64,
                 device: str = None):
        """
        Args:
            model_name: Sentence-BERT model to use
            batch_size: Batch size for encoding
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        
        print(f"Loaded model: {model_name}")
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def _prepare_texts(self, query: str, results: List[Dict]) -> List[str]:
        """
        Prepare text pairs for encoding.
        Combines title and snippet for better relevance signal.
        """
        texts = []
        for result in results:
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            # Combine title and snippet (title is more important)
            result_text = f"{title} {title} {snippet}"  # Repeat title for emphasis
            texts.append(result_text)
        return texts
    
    def score_query(self, query: str, results: List[Dict]) -> np.ndarray:
        """
        Compute relevance scores for all results of a single query.
        
        Returns:
            Array of relevance scores (0-1), same length as results
        """
        # Prepare texts
        result_texts = self._prepare_texts(query, results)
        
        # Encode query and results
        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True
        )[0]
        
        result_embeddings = self.model.encode(
            result_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        # Cosine similarity
        similarities = np.dot(result_embeddings, query_embedding)
        
        # Normalize to 0-1 range (similarities are already -1 to 1 with normalized embeddings)
        similarities = np.clip((similarities + 1) / 2, 0, 1)
        
        return similarities
    
    def score_dataset(self, dataset_path: str, output_path: str = None) -> Dict:
        """
        Add relevance scores to entire dataset.
        
        Args:
            dataset_path: Path to collected SERP dataset
            output_path: Path to save scored dataset (default: overwrite)
        
        Returns:
            Dataset with relevance scores added
        """
        print(f"\n{'='*60}")
        print(f"RELEVANCE SCORING")
        print(f"{'='*60}")
        print(f"Loading dataset: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        queries = dataset['queries']
        total_results = sum(q['num_results'] for q in queries)
        
        print(f"Queries: {len(queries)}")
        print(f"Total results: {total_results}")
        print(f"Scoring relevance...\n")
        
        all_relevance_scores = []
        
        for query_data in tqdm(queries, desc="Scoring relevance"):
            query = query_data['query']
            results = query_data['results']
            
            if results:
                scores = self.score_query(query, results)
                
                for i, result in enumerate(results):
                    result['relevance_score'] = float(scores[i])
                
                all_relevance_scores.extend(scores.tolist())
            else:
                # No results for this query
                for result in query_data.get('results', []):
                    result['relevance_score'] = 0.0
        
        # Update metadata
        dataset['metadata']['relevance_scorer'] = 'Sentence-BERT all-MiniLM-L6-v2'
        dataset['metadata']['relevance_scored_at'] = str(np.datetime64('now'))
        
        if all_relevance_scores:
            dataset['metadata']['mean_relevance'] = float(np.mean(all_relevance_scores))
            dataset['metadata']['std_relevance'] = float(np.std(all_relevance_scores))
            dataset['metadata']['min_relevance'] = float(np.min(all_relevance_scores))
            dataset['metadata']['max_relevance'] = float(np.max(all_relevance_scores))
        
        # Save
        if output_path is None:
            output_path = dataset_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"RELEVANCE SCORES ADDED")
        print(f"{'='*60}")
        print(f"Output: {output_path}")
        
        if all_relevance_scores:
            print(f"\nRelevance Distribution:")
            print(f"  Mean: {np.mean(all_relevance_scores):.4f}")
            print(f"  Std:  {np.std(all_relevance_scores):.4f}")
            print(f"  Min:  {np.min(all_relevance_scores):.4f}")
            print(f"  Max:  {np.max(all_relevance_scores):.4f}")
        
        return dataset


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Score relevance using Sentence-BERT')
    parser.add_argument('--dataset', type=str, 
                       default='data/raw/serp_dataset.json',
                       help='Path to SERP dataset')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: overwrite input)')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence-BERT model name')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for encoding')
    
    args = parser.parse_args()
    
    scorer = RelevanceScorer(
        model_name=args.model,
        batch_size=args.batch_size,
    )
    
    dataset = scorer.score_dataset(args.dataset, args.output)