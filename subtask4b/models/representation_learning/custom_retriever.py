import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import logging

from models.base import BaseRetriever, CachedModelMixin, EmbeddingMixin, prepare_text_simple
from preprocessing import TextPreprocessor

# Disable verbose logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


class CustomRetriever(BaseRetriever, CachedModelMixin, EmbeddingMixin):
    """Dense retriever using transformer-based semantic embeddings"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "semantic_retriever"
        
        # Initialize device
        self.device = self._get_device()
        
        # Initialize preprocessor for queries
        self.preprocessor = TextPreprocessor()
        
        # Initialize embedding model
        self.model = self._init_model()
        
        # Get document embeddings
        self.doc_embeddings = self._load_or_create_model(
            self._create_document_embeddings, 
            suffix=f"embeddings_{self.config.embedding_model.replace('/', '_')}_batch{self.config.batch_size}"
        )
    
    def _get_device(self):
        """Get optimal device for computation"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _init_model(self):
        """Initialize the sentence transformer model"""
        model = SentenceTransformer(self.config.embedding_model, device=self.device)
        return model
    
    def _create_document_embeddings(self):
        """Create embeddings for all documents"""
        # Prepare document texts
        docs = [prepare_text_simple(row.get('title', ''), row.get('abstract', '')) 
                for _, row in self.collection_df.iterrows()]
        
        # Create embeddings
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
        """Retrieve documents using semantic similarity"""
        if top_k is None:
            top_k = self.config.top_k

        processed_query = self.preprocessor.social_media_preprocessing(query_text)
        
        # Create query embedding
        query_embedding = self.model.encode(
            processed_query, 
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device
        )
        
        # Calculate cosine similarities
        cos_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        # Get top-k documents
        top_indices = torch.topk(cos_scores, k=top_k).indices.cpu().numpy()
        
        return [self.cord_uids[idx] for idx in top_indices]