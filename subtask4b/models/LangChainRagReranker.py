from sentence_transformers import CrossEncoder
from models.LangChainBaseRetriever import LangChainBaseRetriever


class LangChainRerankerRetriever(LangChainBaseRetriever):
    """Advanced RAG retriever using LangChain with embeddings and reranking"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        
        self.reranker = self._load_reranker()
    
    def _load_reranker(self):
        """Load the reranker model"""
        reranker_model = self.config.reranker_model

        return CrossEncoder(reranker_model)
    
    def _rerank_documents(self, query, retrieved_docs, top_k=None):
        """Rerank documents using the cross-encoder"""
        if top_k is None:
            top_k = self.config.top_k
            
        # Create pairs of query and document text
        pairs = [(query, doc.page_content) for doc in retrieved_docs]
        
        # Process in batches for more efficient inference
        batch_size = self.config.reranker_batch_size
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_scores = self.reranker.predict(batch_pairs)
            
            if isinstance(batch_scores, list):
                all_scores.extend(batch_scores)
            else:
                # Handle case where predict returns a single score or numpy array
                all_scores.extend(batch_scores.tolist())
        
        # Sort documents by score
        scored_docs = sorted(zip(retrieved_docs, all_scores), key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query with reranking"""
        if top_k is None:
            top_k = self.config.top_k
            
        # Retrieve initial candidates (more than we need for reranking)
        retrieved_docs = self.base_retriever.get_relevant_documents(query_text)
        
        # Rerank candidates
        reranked_docs = self._rerank_documents(query_text, retrieved_docs, top_k=top_k)
        
        # Extract cord_uids
        return [doc.metadata["cord_uid"] for doc in reranked_docs]