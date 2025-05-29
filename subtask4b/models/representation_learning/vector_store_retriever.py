import os
import torch
import logging
from models.base import BaseRetriever, VectorStoreMixin
from vector_db.vector_manager import VectorStoreManager

# Disable verbose logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


class VectorStoreRetriever(BaseRetriever, VectorStoreMixin):
    """Vector database retriever using embeddings"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "vector_store_retriever"
        
        # Initialize device
        self.device = self._get_device()
        
        # Initialize vector store
        self.vector_store_manager = VectorStoreManager(self.config)
        
        # Set up vector store and retriever
        self.vector_store, was_loaded = self._initialize_vector_store(self.vector_store_manager)
        
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.candidate_count}
        )
    
    def _get_device(self):
        """Get optimal device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve documents from vector store"""
        if top_k is None:
            top_k = self.config.top_k
            
        try:
            # Get documents from vector store
            retrieved_docs = self.base_retriever.invoke(query_text)[:top_k]
            
            # Extract document IDs
            results = []
            for doc in retrieved_docs:
                if hasattr(doc, 'metadata') and 'cord_uid' in doc.metadata:
                    results.append(doc.metadata["cord_uid"])
                    
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in vector store retrieval: {e}")
            return []