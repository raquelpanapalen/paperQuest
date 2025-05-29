import os
from sentence_transformers import CrossEncoder
from langchain_ollama.llms import OllamaLLM
import regex as re
from tqdm import tqdm
import torch

from models.base import BaseRetriever, VectorStoreMixin, prepare_text_simple
from vector_db.vector_manager import VectorStoreManager
from ollama import chat
import os
import logging

# Disable progress bars and logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

class LangChainBaseRetriever(BaseRetriever, VectorStoreMixin):
    """Base class for LangChain retrievers with embeddings"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        
        # FIXED: Consistent device detection
        self.device = self._get_device()
        
        # Initialize vector store manager
        self.vector_store_manager = VectorStoreManager(self.config)
        
        # Initialize vector store
        self.vector_store, was_loaded = self._initialize_vector_store(self.vector_store_manager)
        
        # Set up the retriever
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.candidate_count}
        )
    
    def _get_device(self):
        """Get consistent device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query"""
        if top_k is None:
            top_k = self.config.top_k
            
        try:
            # Retrieve documents from vector store
            retrieved_docs = self.base_retriever.get_relevant_documents(query_text)[:top_k]
            
            # Extract cord_uids with error handling
            results = []
            for doc in retrieved_docs:
                if hasattr(doc, 'metadata') and 'cord_uid' in doc.metadata:
                    results.append(doc.metadata["cord_uid"])
                    
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in LangChain retrieval: {e}")
            return []


class LangChainRAGRetriever(LangChainBaseRetriever):
    """Simple RAG retriever using LangChain with embeddings only"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "langchain_rag"


class LangChainRerankerRetriever(LangChainBaseRetriever):
    """Advanced retriever using LangChain with embeddings and reranking"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "langchain_reranker"
        
        # FIXED: Consistent device for reranker
        self.reranker = CrossEncoder(self.config.reranker_model, device=self.device)
    
    def _rerank_documents(self, query, retrieved_docs, top_k=None):
        """Rerank documents using the cross-encoder"""
        if top_k is None:
            top_k = self.config.top_k
            
        if not retrieved_docs:
            return []
            
        # Create pairs of query and document text
        pairs = []
        valid_docs = []
        
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content') and doc.page_content:
                content = str(doc.page_content)
                pairs.append([query, content])
                valid_docs.append(doc)
        
        if not pairs:
            return retrieved_docs[:top_k]
        
        # Process in batches for more efficient inference
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
        
        # Sort documents by score
        scored_docs = sorted(zip(valid_docs, all_scores), key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query with reranking"""
        if top_k is None:
            top_k = self.config.top_k
            
        try:
            # Retrieve initial candidates
            retrieved_docs = self.base_retriever.get_relevant_documents(query_text)
            
            # Rerank candidates
            reranked_docs = self._rerank_documents(query_text, retrieved_docs, top_k=top_k)
            
            # Extract cord_uids
            results = []
            for doc in reranked_docs:
                if hasattr(doc, 'metadata') and 'cord_uid' in doc.metadata:
                    results.append(doc.metadata["cord_uid"])
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in reranker retrieval: {e}")
            return []


from pydantic import BaseModel, Field

class ScientificQueryExpansion(BaseModel):
    """Schema for scientific query expansions"""
    title: str 
    abstract: str


class LangChainQueryExpansionRetriever(BaseRetriever, VectorStoreMixin):
    """Retriever with query expansion using direct Ollama chat API with structured output"""
    
    EXPANSION_MODEL = "llama3.2"
    TEMPERATURE = 0.15
    
    def __init__(self, collection_df, queries_df=None, config=None):
        super().__init__(collection_df, config)
        self.model_name = "langchain_query_expansion"
        
        # FIXED: Consistent device detection
        self.device = self._get_device()
        
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
    
    def _get_device(self):
        """Get consistent device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
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
        
        for query in tqdm(unique_queries, desc="Expanding scientific queries", disable=True):
            if query not in self.query_expansions:
                self.query_expansions[query] = self._expand_single_query(query)
                
                # Save periodically
                if len(self.query_expansions) % 100 == 0:
                    self.cache_manager.save_to_cache(cache_key, self.query_expansions)
        
        # Save final cache
        self.cache_manager.save_to_cache(cache_key, self.query_expansions)
    
    def _expand_single_query(self, query_text):
        """Expand a single query into scientific abstracts using structured output"""
        try:
            prompt = self._get_expansion_prompt(query_text)
            
            # Create up to 3 variations
            expansions = []
            for i in range(3):  # Generate 3 expansions
                try:
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
                    
                    # Use abstract as the expansion
                    if expansion.abstract and expansion.abstract not in expansions:
                        expansions.append(expansion.abstract)
                        
                except Exception as e:
                    print(f"Expansion attempt {i} failed: {e}")
                    continue
            
            # Always include original query first
            return [query_text] + expansions if expansions else [query_text]
            
        except Exception as e:
            print(f"Error in query expansion: {e}")
            return [query_text]
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents using the expanded queries"""
        if top_k is None:
            top_k = self.config.top_k
        
        try:
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
                if hasattr(doc, 'metadata') and 'cord_uid' in doc.metadata:
                    uid = doc.metadata["cord_uid"]
                    if uid not in seen:
                        seen.add(uid)
                        results.append(uid)
                        if len(results) >= top_k:
                            break
            
            return results
            
        except Exception as e:
            print(f"Error in query expansion retrieval: {e}")
            return []