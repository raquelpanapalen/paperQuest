import os
from sentence_transformers import CrossEncoder
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field
import regex as re
from tqdm import tqdm

from models.base import BaseRetriever, VectorStoreMixin
from vector_db.vector_manager import VectorStoreManager


class LangChainBaseRetriever(BaseRetriever, VectorStoreMixin):
    """Base class for LangChain retrievers with embeddings"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        
        # Initialize vector store manager
        self.vector_store_manager = VectorStoreManager(self.config)
        
        # Initialize vector store
        self.vector_store, was_loaded = self._initialize_vector_store(self.vector_store_manager)
        
        # Set up the retriever
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.candidate_count}
        )
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query"""
        if top_k is None:
            top_k = self.config.top_k
            
        # Retrieve documents from vector store
        retrieved_docs = self.base_retriever.get_relevant_documents(query_text)[:top_k]
        
        # Extract cord_uids
        return [doc.metadata["cord_uid"] for doc in retrieved_docs]


class LangChainRAGRetriever(LangChainBaseRetriever):
    """Simple RAG retriever using LangChain with embeddings only"""
    pass


class LangChainRerankerRetriever(LangChainBaseRetriever):
    """Advanced retriever using LangChain with embeddings and reranking"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.reranker = CrossEncoder(self.config.reranker_model)
    
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
                all_scores.extend(batch_scores.tolist())
        
        # Sort documents by score
        scored_docs = sorted(zip(retrieved_docs, all_scores), key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query with reranking"""
        if top_k is None:
            top_k = self.config.top_k
            
        # Retrieve initial candidates
        retrieved_docs = self.base_retriever.get_relevant_documents(query_text)
        
        # Rerank candidates
        reranked_docs = self._rerank_documents(query_text, retrieved_docs, top_k=top_k)
        
        # Extract cord_uids
        return [doc.metadata["cord_uid"] for doc in reranked_docs]


from ollama import chat
from pydantic import BaseModel
from tqdm import tqdm
import json

class ScientificQueryExpansion(BaseModel):
    """Schema for scientific query expansions"""
    title: str 
    abstract: str


class LangChainQueryExpansionRetriever(BaseRetriever, VectorStoreMixin):
    """Retriever with query expansion using direct Ollama chat API with structured output"""
    
    EXPANSION_MODEL = "llama3.2"  # Updated to match your working example
    TEMPERATURE = 0.15  # Lower temperature for more focused outputs
    
    def __init__(self, collection_df, queries_df=None, config=None):
        super().__init__(collection_df, config)
        
        # Initialize vector store manager and store
        self.vector_store_manager = VectorStoreManager(self.config)
        self.vector_store, _ = self._initialize_vector_store(self.vector_store_manager)
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.candidate_count}
        )
        
        # Store model configuration
        self.model_config = {
            'model': self.EXPANSION_MODEL,
            'options': {'temperature': self.TEMPERATURE},
            'format': ScientificQueryExpansion.model_json_schema()
        }
        
        self.query_expansions = {}
        
        # Batch expand queries if provided
        if queries_df is not None:
            self._batch_expand_queries(queries_df)
    
    
    def _get_expansion_prompt(self, query_text):
        """Generate prompt for query expansion"""
        return f"Create one scientific variation out of your imagination of this tweet in form of a short abstract - only take information given from this tweet: {query_text}"
    

    def _batch_expand_queries(self, queries_df):
        """Expand all queries at once and cache the results"""
        cache_key = self.get_cache_key("query_expansions_structured")
        cached = self.cache_manager.load_cached(cache_key)
        if cached:
            self.query_expansions = cached
            return
        
        unique_queries = queries_df['tweet_text'].unique()
        
        for query in tqdm(unique_queries, desc="Expanding scientific queries"):
            if query not in self.query_expansions:
                self.query_expansions[query] = self._expand_single_query(query)
                
                # Save periodically
                if len(self.query_expansions) % 100 == 0:
                    self.cache_manager.save_to_cache(cache_key, self.query_expansions)
        
        # Save final cache
        self.cache_manager.save_to_cache(cache_key, self.query_expansions)
    

    def _expand_single_query(self, query_text: str):
        """Expand a single query into scientific abstracts using structured output"""
        try:
            prompt = self._get_expansion_prompt(query_text)
            
            # Create up to 3 variations
            expansions = []
            for i in range(3):  # Generate 3 expansions
                response = chat(
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                        }
                    ],
                    **self.model_config
                )
                
                # Parse structured response
                expansion = ScientificQueryExpansion.model_validate_json(response.message.content)
                
                # Use abstract as the expansion (ignore title as you mentioned)
                if expansion.abstract and expansion.abstract not in expansions:
                    expansions.append(expansion.abstract)
            
            # Always include original query first
            return [query_text] + expansions if expansions else [query_text]
            
        except Exception as e:
            print(f"Error in query expansion: {e}")
            return [query_text]
    

    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents using the expanded queries"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Get or generate expansions
        if query_text in self.query_expansions:
            expansions = self.query_expansions[query_text]
        else:
            expansions = self._expand_single_query(query_text)
            self.query_expansions[query_text] = expansions
        
        # Combine original query with expansions
        combined_query = "\n".join(expansions)
        
        # Retrieve documents
        docs = self.base_retriever.get_relevant_documents(combined_query)
        
        # Extract unique cord_uids
        seen, results = set(), []
        for doc in docs:
            uid = doc.metadata["cord_uid"]
            if uid not in seen:
                seen.add(uid)
                results.append(uid)
                if len(results) >= top_k:
                    break
        
        return results