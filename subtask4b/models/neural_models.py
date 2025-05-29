import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import logging

from models.base import BaseRetriever, CachedModelMixin, EmbeddingMixin, prepare_text_simple
from models.bm25 import BM25Retriever, EnhancedBM25Retriever

from preprocessing import TextPreprocessor

# Disable sentence transformers progress bars globally
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


class DenseRetriever(BaseRetriever, CachedModelMixin, EmbeddingMixin):
    """Dense retriever using transformer embeddings"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "dense_retriever"
        
        # Initialize device first
        self.device = self._get_device()
        
        # Initialize preprocessor for tweet queries
        self.preprocessor = TextPreprocessor()
        
        # Initialize model with consistent device
        self.model = self._init_model()
        
        # Get document embeddings
        self.doc_embeddings = self._load_or_create_model(
            self._create_document_embeddings, 
            suffix=f"embeddings_{self.config.embedding_model.replace('/', '_')}_batch{self.config.batch_size}_no_limits"
        )
    
    def _get_device(self):
        """Get consistent device across all components"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _init_model(self):
        """Initialize the embedding model with consistent device"""
        model = SentenceTransformer(self.config.embedding_model, device=self.device)
        return model
    
    def _create_document_embeddings(self):
        # FIXED: Use prepare_text_simple instead of arbitrary truncation
        docs = [prepare_text_simple(row.get('title', ''), row.get('abstract', '')) 
                for _, row in self.collection_df.iterrows()]
        
        embeddings = self.model.encode(
            docs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
        )
        
        return np.array(embeddings)
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents using dense embeddings"""
        if top_k is None:
            top_k = self.config.top_k
        
        # ENHANCED: Use tweet preprocessing for queries        
        processed_query = self.preprocessor.preprocess_tweet_query(query_text)
        
        query_embedding = self.model.encode(
            processed_query, 
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device
        )
        
        # Calculate similarity scores
        cos_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        # Get top-k document indices
        top_indices = torch.topk(cos_scores, k=top_k).indices.cpu().numpy()
        
        return [self.cord_uids[idx] for idx in top_indices]


class NeuralReranker(BaseRetriever):
    """Neural reranker using a two-stage retrieval approach"""
    
    def __init__(self, collection_df, config=None, first_stage_model=None):
        super().__init__(collection_df, config)
        self.model_name = "neural_reranker"
        
        # Initialize device first
        self.device = self._get_device()
        
        # Use TextPreprocessor to prepare collection
        self.preprocessor = TextPreprocessor()
        
        # Create a copy of the collection_df to avoid modifying the original
        self.collection_df = collection_df.copy()
        
        # Ensure 'text' column exists if not already present
        if 'text' not in self.collection_df.columns:
            # FIXED: Use prepare_text_simple instead of f-string concatenation
            self.collection_df['text'] = self.collection_df.apply(
                lambda x: prepare_text_simple(x.get('title', ''), x.get('abstract', '')), 
                axis=1
            )
        
        # Initialize first-stage retriever
        if first_stage_model is None:
            self.first_stage = BM25Retriever(self.collection_df, config)
        else:
            self.first_stage = first_stage_model
        
        # Initialize reranker model
        self._init_reranker()
    
    def _get_device(self):
        """Get consistent device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _init_reranker(self):
        """Initialize the reranker model with consistent device"""
        model_name = self.config.reranker_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Consistent device placement
        self.model.to(self.device)
    
    def _batch_inference(self, pairs, batch_size=None):
        """Run batched inference on pairs of query-document texts"""
        if batch_size is None:
            batch_size = self.config.reranker_batch_size
            
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                # Tokenize - let the tokenizer handle truncation automatically
                features = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                # Get scores
                outputs = self.model(**features)
                scores = outputs.logits.squeeze(-1).cpu().numpy()
                all_scores.extend(scores)
                
        return all_scores
    
    def retrieve(self, query_text, top_k=None):
        """Two-stage retrieval: BM25 followed by neural reranking"""
        if top_k is None:
            top_k = self.config.top_k
        
        # ENHANCED: Use tweet preprocessing for first stage
        processed_query = self.preprocessor.preprocess_tweet_query(query_text)
        
        # Get first-stage candidates using processed query
        candidates = self.first_stage.retrieve(processed_query, top_k=self.config.candidate_count)
        
        # Better error handling for missing UIDs
        candidate_indices = []
        for uid in candidates:
            try:
                idx = self.cord_uids.index(uid)
                candidate_indices.append(idx)
            except ValueError:
                continue
        
        if not candidate_indices:
            return []
        
        # Create texts for reranking
        candidate_texts = []
        for idx in candidate_indices:
            row = self.collection_df.iloc[idx]
            
            # FIXED: Use prepare_text_simple instead of arbitrary truncation
            if 'text' in row and row['text']:
                text = str(row['text'])
            else:
                text = prepare_text_simple(row.get('title', ''), row.get('abstract', ''))
            
            candidate_texts.append(text)
        
        # Create pairs of (query, document) for the reranker - use original query
        pairs = [[query_text, doc] for doc in candidate_texts]
        
        # Use batched inference
        all_scores = self._batch_inference(pairs)
        
        # Get indices of top-k scores
        reranked_indices = np.argsort(-np.array(all_scores))[:top_k]
        
        # Map back to original document indices
        final_indices = [candidate_indices[idx] for idx in reranked_indices]
        
        return [self.cord_uids[idx] for idx in final_indices]


class HybridNeuralReranker(BaseRetriever, CachedModelMixin):
    """Hybrid Dense-Sparse Reranker combining BM25 and dense retrieval"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "scientific_tweet_hybrid_retriever"
        
        # Create a copy to avoid issues
        self.collection_df = collection_df.copy()
        
        # Initialize preprocessor for tweet queries
        self.preprocessor = TextPreprocessor()
        
        # Ensure text column exists for compatibility
        if 'text' not in self.collection_df.columns:
            # FIXED: Use prepare_text_simple instead of f-string concatenation
            self.collection_df['text'] = self.collection_df.apply(
                lambda x: prepare_text_simple(x.get('title', ''), x.get('abstract', '')), 
                axis=1
            )
        
        # Consistent device setup
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        print(f"Using encoder: {self.config.embedding_model}")
        
        # Initialize sparse and dense retrievers with the prepared collection
        self.sparse_retriever = EnhancedBM25Retriever(self.collection_df, config)
        self.dense_retriever = DenseRetriever(self.collection_df, config)
        
        # Consistent device for CrossEncoder
        self.reranker = CrossEncoder(config.reranker_model, device=self.device)
        
        # Fusion parameters
        self.rrf_k = getattr(config, 'rrf_k', 60)
        self.sparse_weight = getattr(config, 'sparse_weight', 0.5)
        
        # Cache fusion results
        self._fusion_cache = {}
    
    def _get_device(self):
        """Get consistent device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve using hybrid sparse-dense approach with neural reranking"""
        if top_k is None:
            top_k = self.config.top_k
        
        # ENHANCED: Preprocess tweet query
        processed_query = self.preprocessor.preprocess_tweet_query(query_text)
        
        # Check cache
        cache_key = f"{processed_query}_{self.config.candidate_count}"
        if cache_key in self._fusion_cache:
            fused_candidates = self._fusion_cache[cache_key]
        else:
            # Get candidates from both retrievers using processed query
            sparse_candidates = self.sparse_retriever.retrieve(
                processed_query, top_k=self.config.candidate_count
            )
            dense_candidates = self.dense_retriever.retrieve(
                processed_query, top_k=self.config.candidate_count
            )
            
            # Combine using reciprocal rank fusion
            fused_candidates = self._reciprocal_rank_fusion(
                sparse_candidates, dense_candidates
            )
            
            # Cache the fusion result
            self._fusion_cache[cache_key] = fused_candidates
        
        # Prepare for neural reranking
        rerank_candidates = fused_candidates[:self.config.candidate_count]
        
        if not rerank_candidates:
            return []
        
        # Neural reranking using original query
        return self._neural_rerank(query_text, rerank_candidates, top_k)
    
    def _reciprocal_rank_fusion(self, sparse_results, dense_results):
        """Combine results using Reciprocal Rank Fusion (RRF)"""
        scores = {}
        
        # Score sparse results  
        for rank, doc_id in enumerate(sparse_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + self.sparse_weight * rrf_score
        
        # Score dense results
        for rank, doc_id in enumerate(dense_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + (1 - self.sparse_weight) * rrf_score
        
        # Sort by combined score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs]
    
    def _neural_rerank(self, query_text, candidates, top_k):
        """Rerank candidates using neural model"""
        # Prepare query-document pairs
        pairs = []
        valid_indices = []
        
        for i, uid in enumerate(candidates):
            try:
                idx = self.cord_uids.index(uid)
                # FIXED: Use prepare_text_simple instead of arbitrary truncation
                row = self.collection_df.iloc[idx]
                doc_text = prepare_text_simple(row.get('title', ''), row.get('abstract', ''))
                pairs.append([query_text, doc_text])
                valid_indices.append(i)
            except ValueError:
                continue
        
        if not pairs:
            return []
        
        # Use config batch size for consistency, disable progress
        scores = self.reranker.predict(
            pairs, 
            batch_size=self.config.reranker_batch_size,
            show_progress_bar=False
        )
        
        # Sort by scores
        sorted_indices = np.argsort(-scores)[:top_k]
        
        # Map back to original candidates
        return [candidates[valid_indices[idx]] for idx in sorted_indices]