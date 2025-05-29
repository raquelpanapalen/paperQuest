# Replace CIIVERRetriever with this better implementation

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import logging

from models.base import BaseRetriever, CachedModelMixin, prepare_text_simple
from preprocessing import TextPreprocessor

# Disable progress bars globally
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


class HybridRetriever(BaseRetriever, CachedModelMixin):
    """
    Three-stage hybrid retrieval: BM25 + Dense Vector Search + CrossEncoder Reranking
    
    Stage 1: Get candidates from both sparse (BM25) and dense (FAISS) retrieval
    Stage 2: Fuse candidates using reciprocal rank fusion  
    Stage 3: Rerank top candidates with CrossEncoder
    """
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "hybrid_dense_sparse_retriever"
        
        # Device setup
        self.device = self._get_device()
        print(f"Hybrid retriever using device: {self.device}")
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Initialize models with consistent device
        self.dense_encoder = SentenceTransformer(
            getattr(config, 'embedding_model', 'sentence-transformers/allenai-specter'), 
            device=self.device
        )
        
        self.reranker = CrossEncoder(
            getattr(config, 'reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'), 
            device=self.device
        )
        
        # Initialize retrieval components
        self._build_retrieval_indices()
        
        # Fusion parameters
        self.sparse_weight = getattr(config, 'sparse_weight', 0.5)
        self.dense_weight = 1.0 - self.sparse_weight
        self.candidate_count = getattr(config, 'candidate_count', 100)
        
        # Cache for query embeddings
        self.query_embedding_cache = {}
    
    def _get_device(self):
        """Get consistent device across all components"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _build_retrieval_indices(self):
        """Build both sparse and dense retrieval indices"""
        indices = self._load_or_create_model(
            self._create_retrieval_indices,
            suffix="hybrid_indices_no_limits"
        )
        
        self.sparse_index = indices['sparse_index']
        self.dense_index = indices['dense_index'] 
        self.document_embeddings = indices['document_embeddings']
        self.processed_documents = indices['processed_documents']
    
    def _create_retrieval_indices(self):
        """Create both BM25 and FAISS indices with consistent preprocessing"""
        print("Building hybrid retrieval indices...")
        
        # FIXED: Use prepare_text_simple instead of preprocessing collection
        documents = [prepare_text_simple(row.get('title', ''), row.get('abstract', '')) 
                    for _, row in self.collection_df.iterrows()]
        
        # Build sparse index (BM25)
        print("Building sparse index (BM25)...")
        tokenized_docs = [doc.split() for doc in documents]
        sparse_index = BM25Okapi(tokenized_docs)
        
        # Build dense index (FAISS)
        print("Building dense embeddings...")
        document_embeddings = self.dense_encoder.encode(
            documents,
            batch_size=getattr(self.config, 'batch_size', 32),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
        )
        
        print("Building FAISS index...")
        dimension = document_embeddings.shape[1]
        dense_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        dense_index.add(document_embeddings.astype(np.float32))
        
        return {
            'sparse_index': sparse_index,
            'dense_index': dense_index,
            'document_embeddings': document_embeddings,
            'processed_documents': documents
        }
    
    def _encode_query(self, query_text):
        """Encode query using dense encoder with caching"""
        if query_text in self.query_embedding_cache:
            return self.query_embedding_cache[query_text]
        
        # Minimal preprocessing for query
        processed_query = self._minimal_query_preprocessing(query_text)
        
        embedding = self.dense_encoder.encode(
            processed_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device
        ).reshape(1, -1)
        
        self.query_embedding_cache[query_text] = embedding
        return embedding
    
    def _minimal_query_preprocessing(self, text):
        """Minimal query preprocessing - just remove URLs"""
        if not text:
            return ""
        
        import re
        # Only remove URLs, keep everything else
        text = re.sub(r'http\S+|www\.\S+|t\.co/\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _sparse_search(self, query_text, num_candidates):
        """BM25 sparse search"""
        processed_query = self.preprocessor.preprocess_tweet_query(query_text)
        tokenized_query = processed_query.split()
        
        scores = self.sparse_index.get_scores(tokenized_query)
        
        # Get top candidates with bounds checking
        valid_candidates = min(num_candidates, len(scores))
        top_indices = np.argsort(-scores)[:valid_candidates]
        
        return [(idx, scores[idx]) for idx in top_indices if idx < len(self.cord_uids)]
    
    def _dense_search(self, query_text, num_candidates):
        """FAISS dense search"""
        query_embedding = self._encode_query(query_text)
        
        # Search with bounds checking
        valid_candidates = min(num_candidates, self.dense_index.ntotal)
        
        similarities, indices = self.dense_index.search(
            query_embedding.astype(np.float32), 
            valid_candidates
        )
        
        return [(idx, score) for idx, score in zip(indices[0], similarities[0]) 
                if idx < len(self.cord_uids)]
    
    def _fuse_candidates(self, sparse_results, dense_results):
        """Fuse candidates using reciprocal rank fusion"""
        scores = {}
        
        # Add sparse scores
        for rank, (doc_idx, score) in enumerate(sparse_results):
            if doc_idx < len(self.cord_uids):
                uid = self.cord_uids[doc_idx]
                rrf_score = 1.0 / (60 + rank + 1)  # RRF with k=60
                scores[uid] = scores.get(uid, 0) + self.sparse_weight * rrf_score
        
        # Add dense scores  
        for rank, (doc_idx, score) in enumerate(dense_results):
            if doc_idx < len(self.cord_uids):
                uid = self.cord_uids[doc_idx]
                rrf_score = 1.0 / (60 + rank + 1)  # RRF with k=60
                scores[uid] = scores.get(uid, 0) + self.dense_weight * rrf_score
        
        # Sort by combined score
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [uid for uid, _ in sorted_candidates]
    
    def _rerank_with_cross_encoder(self, query_text, candidate_uids, top_k):
        """Rerank candidates using CrossEncoder"""
        if not candidate_uids:
            return []
        
        # Prepare query-document pairs
        pairs = []
        valid_uids = []
        
        max_candidates = min(len(candidate_uids), self.config.candidate_count)  
        for uid in candidate_uids[:max_candidates]:
            try:
                idx = self.cord_uids.index(uid)
                doc = self.collection_df.iloc[idx]
                
                # FIXED: Use prepare_text_simple instead of arbitrary truncation
                doc_text = prepare_text_simple(doc.get('title', ''), doc.get('abstract', ''))
                
                pairs.append([query_text, doc_text])  # Use original query
                valid_uids.append(uid)
                
            except (IndexError, ValueError):
                continue
        
        if not pairs:
            return candidate_uids[:top_k]
        
        try:
            # Batch reranking
            scores = self.reranker.predict(
                pairs,
                batch_size=getattr(self.config, 'reranker_batch_size', 32),
                show_progress_bar=False
            )
            
            # Sort by reranker scores
            scored_results = list(zip(valid_uids, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            return [uid for uid, _ in scored_results[:top_k]]
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            return candidate_uids[:top_k]
    
    def retrieve(self, query_text, top_k=None):
        """
        Three-stage hybrid retrieval:
        1. Get candidates from both BM25 and dense search
        2. Fuse candidates using reciprocal rank fusion
        3. Rerank with CrossEncoder
        """
        if top_k is None:
            top_k = self.config.top_k
        
        num_candidates = min(self.candidate_count, len(self.cord_uids))
        
        try:
            # Stage 1: Parallel candidate retrieval
            with ThreadPoolExecutor(max_workers=2) as executor:
                sparse_future = executor.submit(self._sparse_search, query_text, num_candidates)
                dense_future = executor.submit(self._dense_search, query_text, num_candidates)
                
                sparse_candidates = sparse_future.result()
                dense_candidates = dense_future.result()
        
        except Exception as e:
            print(f"Candidate retrieval failed: {e}")
            return []
        
        # Stage 2: Fuse candidates
        fused_candidates = self._fuse_candidates(sparse_candidates, dense_candidates)
        
        if not fused_candidates:
            return []
        
        # Stage 3: Rerank with CrossEncoder
        return self._rerank_with_cross_encoder(query_text, fused_candidates, top_k)