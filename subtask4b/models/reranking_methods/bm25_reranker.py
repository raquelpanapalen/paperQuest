from rank_bm25 import BM25Okapi
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from preprocessing import TextPreprocessor
from models.base import BaseRetriever, CachedModelMixin, prepare_text_simple


class BM25Reranker(BaseRetriever, CachedModelMixin):
    """BM25 retrieval with neural reranking"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "bm25_reranker"
        
        # Preprocess collection 
        self.preprocessor = TextPreprocessor()
        self.processed_df = self.preprocessor.prepare_collection_cleaned(collection_df)
        
        # Create BM25 index
        self.bm25 = self._load_or_create_model(self._create_bm25)
        
        # Initialize reranker
        self.use_reranking = getattr(config, 'use_reranking', True)
        if self.use_reranking:
            self.device = self._get_device()
            self.reranker = CrossEncoder(
                getattr(config, 'reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                device=self.device
            )
            self.candidate_count = getattr(config, 'candidate_count', 50)
    
    def _get_device(self):
        """Get optimal device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _create_bm25(self):
        """Create BM25 index"""
        corpus = self.processed_df['text'].tolist()
        tokenized_corpus = [doc.split() for doc in corpus]
        return BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve with BM25 then rerank"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Get BM25 scores 
        processed_query = self.preprocessor.social_media_preprocessing(query_text)
        tokenized_query = processed_query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Return BM25 results if no reranking
        if not self.use_reranking:
            top_indices = np.argsort(-doc_scores)[:top_k]
            valid_indices = [idx for idx in top_indices if idx < len(self.cord_uids)]
            return [self.cord_uids[idx] for idx in valid_indices]
        
        # Get candidates for reranking
        retrieval_count = min(self.candidate_count, len(self.cord_uids))
        top_indices = np.argsort(-doc_scores)[:retrieval_count]
        
        # Prepare candidates and text pairs
        candidates = []
        pairs = []
        for idx in top_indices:
            if idx < len(self.cord_uids):
                uid = self.cord_uids[idx]
                candidates.append(uid)
                
                try:
                    row = self.collection_df.iloc[idx]
                    doc_text = prepare_text_simple(row.get('title', ''), row.get('abstract', ''))
                    pairs.append([query_text, doc_text])
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
        except Exception as e:
            print(f"BM25 reranking failed: {e}")
            return candidates[:top_k]