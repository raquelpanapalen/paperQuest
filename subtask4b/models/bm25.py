from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import TextPreprocessor

from models.base import BaseRetriever, CachedModelMixin, prepare_text_simple


class BM25Retriever(BaseRetriever, CachedModelMixin):
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "bm25_baseline"
        
        # Use BM25-specific preprocessing
        self.preprocessor = TextPreprocessor()
        self.processed_df = self.preprocessor.preprocess_collection(
            collection_df, 
            model_type='bm25'
        )
        
        self.bm25 = self._load_or_create_model(self._create_bm25)
    
    def _create_bm25(self):
        corpus = self.processed_df['text'].tolist()
        tokenized_corpus = [doc.split() for doc in corpus]
        return BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query"""
        if top_k is None:
            top_k = self.config.top_k
        
        # ENHANCED: Use tweet preprocessing for queries
        processed_query = self.preprocessor.preprocess_tweet_query(query_text)
        tokenized_query = processed_query.split()
        
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-doc_scores)[:top_k]
        
        # Ensure indices are within bounds
        valid_indices = [idx for idx in top_indices if idx < len(self.cord_uids)]
        return [self.cord_uids[idx] for idx in valid_indices]


class EnhancedBM25Retriever(BaseRetriever, CachedModelMixin):
    """Enhanced BM25: EXACT same as baseline + reranking only"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "enhanced_bm25"
        
        self.preprocessor = TextPreprocessor()
        self.processed_df = self.preprocessor.preprocess_collection(
            collection_df, 
            model_type='bm25'
        )
        
        self.bm25 = self._load_or_create_model(self._create_bm25)
        
        self.use_reranking = getattr(config, 'use_reranking', True)
        if self.use_reranking:
            self.device = self._get_device()
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(
                getattr(config, 'reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                device=self.device
            )
            self.candidate_count = getattr(config, 'candidate_count', 50)
    
    def _get_device(self):
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _create_bm25(self):
        # IDENTICAL to baseline
        corpus = self.processed_df['text'].tolist()
        tokenized_corpus = [doc.split() for doc in corpus]
        return BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query_text, top_k=None):
        """IDENTICAL to baseline BM25 + optional reranking"""
        if top_k is None:
            top_k = self.config.top_k
        
        processed_query = self.preprocessor.preprocess_tweet_query(query_text)
        tokenized_query = processed_query.split()
        
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        if not self.use_reranking:
            top_indices = np.argsort(-doc_scores)[:top_k]
            valid_indices = [idx for idx in top_indices if idx < len(self.cord_uids)]
            return [self.cord_uids[idx] for idx in valid_indices]
        
        # reranking
        retrieval_count = min(self.candidate_count, len(self.cord_uids))
        top_indices = np.argsort(-doc_scores)[:retrieval_count]
        
        # Get candidates
        candidates = []
        pairs = []
        for idx in top_indices:
            if idx < len(self.cord_uids):
                uid = self.cord_uids[idx]
                candidates.append(uid)
                
                # Create reranking pair
                try:
                    row = self.collection_df.iloc[idx]
                    # FIXED: Use prepare_text_simple instead of arbitrary truncation
                    doc_text = prepare_text_simple(row.get('title', ''), row.get('abstract', ''))
                    pairs.append([query_text, doc_text])  # Original query
                except:
                    pairs.append([query_text, ""])
        
        if not pairs:
            return candidates[:top_k]
        
        # Rerank
        try:
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return [uid for uid, _ in scored_candidates[:top_k]]
        except:
            return candidates[:top_k]