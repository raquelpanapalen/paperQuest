import os
import torch
import logging
from sentence_transformers import CrossEncoder
from models.base import BaseRetriever, VectorStoreMixin
from vector_db.vector_manager import VectorStoreManager

# Disable verbose logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


class VectorStoreReranker(BaseRetriever, VectorStoreMixin):
    """Vector store retrieval with neural reranking"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "vector_store_reranker"
        
        # Initialize device
        self.device = self._get_device()
        
        # Initialize vector store
        self.vector_store_manager = VectorStoreManager(self.config)
        self.vector_store, was_loaded = self._initialize_vector_store(self.vector_store_manager)
        
        # Set up base retriever
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.candidate_count}
        )
        
        # Initialize neural reranker
        self.reranker = CrossEncoder(self.config.reranker_model, device=self.device)
    
    def _get_device(self):
        """Get optimal device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _rerank_documents(self, query, retrieved_docs, top_k=None):
        """Rerank documents using cross-encoder"""
        if top_k is None:
            top_k = self.config.top_k
            
        if not retrieved_docs:
            return []
            
        # Create query-document pairs
        pairs = []
        valid_docs = []
        
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content') and doc.page_content:
                content = str(doc.page_content)
                pairs.append([query, content])
                valid_docs.append(doc)
        
        if not pairs:
            return retrieved_docs[:top_k]
        
        # Process in batches
        batch_size = self.config.reranker_batch_size
        all_scores = []
        
        try:
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                batch_scores = self.reranker.predict(batch_pairs, show_progress_bar=False)
                
                if isinstance(batch_scores, list):
                    all_scores.extend(batch_scores)
                else:
                    all_scores.extend(batch_scores.tolist())
        except Exception as e:
            print(f"Reranking error: {e}")
            return retrieved_docs[:top_k]
        
        # Sort by reranking scores
        scored_docs = sorted(zip(valid_docs, all_scores), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve with vector store then rerank"""
        if top_k is None:
            top_k = self.config.top_k
            
        try:
            # Get candidates from vector store
            retrieved_docs = self.base_retriever.invoke(query_text)
            
            # Rerank candidates
            reranked_docs = self._rerank_documents(query_text, retrieved_docs, top_k=top_k)
            
            # Extract document IDs
            results = []
            for doc in reranked_docs:
                if hasattr(doc, 'metadata') and 'cord_uid' in doc.metadata:
                    results.append(doc.metadata["cord_uid"])
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in vector store reranker: {e}")
            return []