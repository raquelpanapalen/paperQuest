from models.base_class import BaseRetriever
from rank_bm25 import BM25Okapi
import os
import numpy as np
import pickle
import hashlib

class BM25Retriever(BaseRetriever):
    """Baseline BM25 retriever"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        
        # Initialize BM25 model
        self._init_bm25()
    
    def _get_cache_filename(self):
        """Generate a unique cache filename based on collection size and content"""
        # Create a unique identifier for this specific collection
        collection_size = len(self.collection_df)
        
        # Use first few cord_uids as a fingerprint of the dataset
        max_sample = min(100, len(self.collection_df))
        ids_sample = "_".join(self.collection_df['cord_uid'].iloc[:max_sample])
        data_hash = hashlib.md5(ids_sample.encode()).hexdigest()[:8]
        
        # Include collection size in the cache filename
        cache_file = os.path.join(self.config.cache_dir, f'bm25_baseline_{collection_size}_{data_hash}.pkl')
        return cache_file
    
    def _init_bm25(self):
        """Initialize the BM25 model"""
        cache_file = self._get_cache_filename()
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.bm25 = pickle.load(f)
        else:
            # Extract text from collection
            corpus = self.collection_df['text'].tolist()
            tokenized_corpus = [doc.split() for doc in corpus]
            
            # Create BM25 model
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            # Cache model
            with open(cache_file, 'wb') as f:
                pickle.dump(self.bm25, f)
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query"""
        if top_k is None:
            top_k = self.config.top_k
            
        # Tokenize query
        tokenized_query = query_text.split()
        
        # Get scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k document indices
        top_indices = np.argsort(-doc_scores)[:top_k]
        
        # Ensure indices are within bounds
        valid_indices = [idx for idx in top_indices if idx < len(self.cord_uids)]
        
        # Return top document IDs
        return [self.cord_uids[idx] for idx in valid_indices]