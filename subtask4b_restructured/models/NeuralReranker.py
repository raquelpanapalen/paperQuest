from models.base_class import BaseRetriever
from models.BM25Retriever import BM25Retriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np


class NeuralReranker(BaseRetriever):
    """Neural reranker using a two-stage retrieval approach"""
    
    def __init__(self, collection_df, config=None, first_stage_model=None):
        super().__init__(collection_df, config)
        
        # Initialize first-stage retriever
        if first_stage_model is None:
            self.first_stage = BM25Retriever(collection_df, config)
        else:
            self.first_stage = first_stage_model
        
        # Initialize reranker model
        self._init_reranker()
    
    def _init_reranker(self):
        """Initialize the reranker model"""
        model_name = self.config.reranker_model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Use GPU if available and configured
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        self.device = device
        self.model.to(self.device)
    
    def _get_candidates(self, query, candidate_count=None):
        """First-stage retrieval to get initial candidates"""
        if candidate_count is None:
            candidate_count = self.config.candidate_count
        
        # Get top-k documents from first-stage retriever
        candidates = self.first_stage.retrieve(query, top_k=candidate_count)
        
        # Convert cord_uids to indices
        candidate_indices = [self.cord_uids.index(uid) for uid in candidates]
        
        return candidate_indices
    
    def _batch_inference(self, pairs, batch_size=None):
        """Run batched inference on pairs of query-document texts"""
        if batch_size is None:
            batch_size = self.config.reranker_batch_size
            
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                # Tokenize
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
    
    def _rerank_candidates(self, query, candidate_indices, top_k=None):
        """Rerank candidates using the neural reranker"""
        if top_k is None:
            top_k = self.config.top_k
            
        candidate_texts = []
        for idx in candidate_indices:
            doc = self.collection_df.iloc[idx]
            text = doc['text'][:512]  # Truncate to fit in context
            candidate_texts.append(text)
        
        # Create pairs of (query, document) for the reranker
        pairs = [[query, doc] for doc in candidate_texts]
        
        # Use batched inference for efficiency
        all_scores = self._batch_inference(pairs)
        
        # Get indices of top-k scores
        reranked_indices = np.argsort(-np.array(all_scores))[:top_k]
        
        # Map back to original document indices
        final_indices = [candidate_indices[idx] for idx in reranked_indices]
        
        return [self.cord_uids[idx] for idx in final_indices]
    
    def retrieve(self, query_text, top_k=None):
        """Two-stage retrieval: BM25 followed by neural reranking"""
        if top_k is None:
            top_k = self.config.top_k
            
        # Get first-stage candidates
        candidate_indices = self._get_candidates(query_text)
        
        # Rerank candidates
        results = self._rerank_candidates(query_text, candidate_indices, top_k)
        
        return results