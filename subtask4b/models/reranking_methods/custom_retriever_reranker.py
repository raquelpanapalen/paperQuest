import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import logging
from models.base import BaseRetriever, prepare_text_simple
from models.representation_learning.custom_retriever import CustomRetriever  
from preprocessing import TextPreprocessor

# Disable verbose logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


class CustomRetrieverReranker(BaseRetriever):
    """Semantic Retrieval first + neural reranking"""
    
    def __init__(self, collection_df, config=None, first_stage_model=None):
        super().__init__(collection_df, config)
        self.model_name = "semantic_reranker"
        
        # Initialize device
        self.device = self._get_device()
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Prepare collection
        self.collection_df = collection_df.copy()
        
        # Ensure text column exists
        if 'text' not in self.collection_df.columns:
            self.collection_df['text'] = self.collection_df.apply(
                lambda x: prepare_text_simple(x.get('title', ''), x.get('abstract', '')), 
                axis=1
            )
        
        # first-stage retriever as CustomRetriever
        if first_stage_model is None:
            self.first_stage = CustomRetriever(self.collection_df, config)
        else:
            self.first_stage = first_stage_model
        
        # Initialize neural reranker
        self._init_reranker()
    
    def _get_device(self):
        """Get optimal device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _init_reranker(self):
        """Initialize neural reranking model"""
        model_name = self.config.reranker_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to device
        self.model.to(self.device)
    
    def _batch_inference(self, pairs, batch_size=None):
        """Run batched inference on query-document pairs"""
        if batch_size is None:
            batch_size = self.config.reranker_batch_size
            
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                # Tokenize pairs
                features = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                # Get reranking scores
                outputs = self.model(**features)
                scores = outputs.logits.squeeze(-1).cpu().numpy()
                all_scores.extend(scores)
                
        return all_scores
    
    def retrieve(self, query_text, top_k=None):
        """Two-stage retrieval: Semantic to Neural reranking"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Use minimal preprocessing for semantic queries (preserve semantic information)
        processed_query = self.preprocessor.social_media_preprocessing(query_text)
        candidates = self.first_stage.retrieve(processed_query, top_k=self.config.candidate_count)
        
        # Handle missing documents
        candidate_indices = []
        for uid in candidates:
            try:
                idx = self.cord_uids.index(uid)
                candidate_indices.append(idx)
            except ValueError:
                continue
        
        if not candidate_indices:
            return []
        
        # Prepare texts for reranking
        candidate_texts = []
        for idx in candidate_indices:
            row = self.collection_df.iloc[idx]
            
            # Use existing text or create from title/abstract
            if 'text' in row and row['text']:
                text = str(row['text'])
            else:
                text = prepare_text_simple(row.get('title', ''), row.get('abstract', ''))
            
            candidate_texts.append(text)
        
        # Neural reranking stage
        pairs = [[query_text, doc] for doc in candidate_texts]  # Use original query for reranking
        
        # Get reranking scores
        try:
            all_scores = self._batch_inference(pairs)
            
            # Sort by scores and return top-k
            reranked_indices = np.argsort(-np.array(all_scores))[:top_k]
            
            # Map back to document IDs
            final_indices = [candidate_indices[idx] for idx in reranked_indices]
            
            return [self.cord_uids[idx] for idx in final_indices]
            
        except Exception as e:
            print(f"Semantic reranking failed: {e}")
            return candidates[:top_k]