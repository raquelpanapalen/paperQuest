import torch
from tqdm import tqdm
from pydantic import BaseModel, Field
from models.base import BaseRetriever, VectorStoreMixin
from vector_db.vector_manager import VectorStoreManager
from ollama import chat
import os
import logging

# Disable verbose logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)


class ScientificQueryExpansion(BaseModel):
    """Schema for scientific query expansions"""
    title: str 
    abstract: str


class QueryExpansionRetriever(BaseRetriever, VectorStoreMixin):
    """Retriever with query expansion using Ollama for scientific abstracts"""
    
    EXPANSION_MODEL = "llama3.2"
    TEMPERATURE = 0.15
    
    def __init__(self, collection_df, queries_df=None, config=None):
        super().__init__(collection_df, config)
        self.model_name = "query_expansion_retriever"
        
        # Initialize device
        self.device = self._get_device()
        
        # Initialize vector store
        self.vector_store_manager = VectorStoreManager(self.config)
        self.vector_store, _ = self._initialize_vector_store(self.vector_store_manager)
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.candidate_count}
        )
        
        # Model configuration
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
        """Get optimal device"""
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
        """Expand all queries and cache results"""
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
        """Expand single query into scientific abstracts"""
        try:
            prompt = self._get_expansion_prompt(query_text)
            
            # Generate multiple expansions
            expansions = []
            for i in range(1):  # Generate 1 variation
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
                    
                    # Use abstract as expansion
                    if expansion.abstract and expansion.abstract not in expansions:
                        expansions.append(expansion.abstract)
                        
                except Exception as e:
                    print(f"Expansion attempt {i} failed: {e}")
                    continue
            
            # Include original query first
            return [query_text] + expansions if expansions else [query_text]
            
        except Exception as e:
            print(f"Error in query expansion: {e}")
            return [query_text]
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve using expanded queries"""
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
            
            docs = self.base_retriever.invoke(combined_query)
            
            # Extract unique document IDs
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