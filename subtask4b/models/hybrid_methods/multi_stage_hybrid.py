import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import logging

from models.base import BaseRetriever, CachedModelMixin, prepare_text_simple
from preprocessing import TextPreprocessor

# Disable verbose logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


class MultiStageHybrid(BaseRetriever, CachedModelMixin):
    """
    Hybrid retrieval combining BM25 (sparse) + Dense vectors + CrossEncoder reranking
    
    Three stages:
    1. Get candidates from BM25 and dense search
    2. Fuse using reciprocal rank fusion  
    3. Rerank with CrossEncoder
    """
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)

        self.model_name = "multi_stage_hybrid"
        
        # Device setup
        self.device = self._get_device()
        print(f"Hybrid retriever using device: {self.device}")
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Initialize models
        self.dense_encoder = SentenceTransformer(
            getattr(config, 'embedding_model', 'sentence-transformers/allenai-specter'), 
            device=self.device
        )
        
        self.reranker = CrossEncoder(
            getattr(config, 'reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'), 
            device=self.device
        )
        
        # Build retrieval indices
        self._build_retrieval_indices()
        
        # Fusion settings
        self.sparse_weight = getattr(config, 'sparse_weight', 0.5)
        self.dense_weight = 1.0 - self.sparse_weight
        self.candidate_count = getattr(config, 'candidate_count', 100)
        
        # Query cache
        self.query_embedding_cache = {}
    
    def _get_device(self):
        """Get optimal device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _build_retrieval_indices(self):
        """Build both sparse and dense indices"""
        indices = self._load_or_create_model(
            self._create_retrieval_indices,
            suffix="hybrid_indices"
        )
        
        self.sparse_index = indices['sparse_index']
        self.dense_index = indices['dense_index'] 
        self.document_embeddings = indices['document_embeddings']
        self.processed_documents = indices['processed_documents']
    
    def _create_retrieval_indices(self):
        """Create BM25 and FAISS indices"""
        print("Building hybrid retrieval indices...")
        
        # Prepare documents
        documents = [prepare_text_simple(row.get('title', ''), row.get('abstract', '')) 
                    for _, row in self.collection_df.iterrows()]
        
        # Build BM25 index
        print("Building sparse index (BM25)...")
        tokenized_docs = [doc.split() for doc in documents]
        sparse_index = BM25Okapi(tokenized_docs)
        
        # Build dense embeddings
        print("Building dense embeddings...")
        document_embeddings = self.dense_encoder.encode(
            documents,
            batch_size=getattr(self.config, 'batch_size', 32),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
        )
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = document_embeddings.shape[1]
        dense_index = faiss.IndexFlatIP(dimension)  
        dense_index.add(document_embeddings.astype(np.float32))
        
        return {
            'sparse_index': sparse_index,
            'dense_index': dense_index,
            'document_embeddings': document_embeddings,
            'processed_documents': documents
        }
    
    def _encode_query(self, query_text):
        """Encode query with caching"""
        if query_text in self.query_embedding_cache:
            return self.query_embedding_cache[query_text]
        

        processed_query = self.preprocessor.social_media_preprocessing(query_text)
        
        embedding = self.dense_encoder.encode(
            processed_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device
        ).reshape(1, -1)
        
        self.query_embedding_cache[query_text] = embedding
        return embedding
    
    def _sparse_search(self, query_text, num_candidates):
        """BM25 search"""
        processed_query = self.preprocessor.social_media_preprocessing(query_text)
        tokenized_query = processed_query.split()
        
        scores = self.sparse_index.get_scores(tokenized_query)
        
        # Get top candidates
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
        """Fuse using reciprocal rank fusion"""
        scores = {}
        rrf_k = getattr(self.config, 'rrf_k', 60)  
        
        # Add sparse scores
        for rank, (doc_idx, score) in enumerate(sparse_results):
            if doc_idx < len(self.cord_uids):
                uid = self.cord_uids[doc_idx]
                rrf_score = 1.0 / (rrf_k + rank + 1)
                scores[uid] = scores.get(uid, 0) + self.sparse_weight * rrf_score
        
        # Add dense scores  
        for rank, (doc_idx, score) in enumerate(dense_results):
            if doc_idx < len(self.cord_uids):
                uid = self.cord_uids[doc_idx]
                rrf_score = 1.0 / (rrf_k + rank + 1)
                scores[uid] = scores.get(uid, 0) + self.dense_weight * rrf_score
        
        # Sort by combined score
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [uid for uid, _ in sorted_candidates]
    
    def _rerank_with_cross_encoder(self, query_text, candidate_uids, top_k):
        """Rerank using CrossEncoder"""
        if not candidate_uids:
            return []
        
        # Prepare query-document pairs
        pairs = []
        valid_uids = []
        
        max_candidates = min(len(candidate_uids), self.candidate_count)  
        for uid in candidate_uids[:max_candidates]:
            try:
                idx = self.cord_uids.index(uid)
                doc = self.collection_df.iloc[idx]
                
                doc_text = prepare_text_simple(doc.get('title', ''), doc.get('abstract', ''))
                
                pairs.append([query_text, doc_text])
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
            
            # Sort by scores
            scored_results = list(zip(valid_uids, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            return [uid for uid, _ in scored_results[:top_k]]
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            return candidate_uids[:top_k]
    
    def retrieve(self, query_text, top_k=None):
        """
        Three-stage hybrid retrieval:
        1. Get candidates from BM25 and dense search
        2. Fuse with reciprocal rank fusion
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
        
        # Stage 3: Rerank
        return self._rerank_with_cross_encoder(query_text, fused_candidates, top_k)