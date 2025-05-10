from typing import List, Optional
import pandas as pd
from tqdm import tqdm
import os

from cache_manager import CacheManager
from preprocessing import TextPreprocessor


class RetrievalConfig:
    """Central configuration class for all retrieval models"""
    
    def __init__(self, **kwargs):
        # General settings
        self.cache_dir = kwargs.get('cache_dir', 'cache')
        self.top_k = kwargs.get('top_k', 5)
        
        # Dataset sampling settings
        self.collection_sample_size = kwargs.get('collection_sample_size', None)
        self.collection_sample_seed = kwargs.get('collection_sample_seed', 42)

        # Vector database settings
        self.vector_db_path = kwargs.get('vector_db_path', os.path.join(self.cache_dir, "chroma_db"))
        self.vector_db_type = kwargs.get('vector_db_type', 'chroma')
        
        # Model-specific settings
        self.embedding_model = kwargs.get('embedding_model', 'sentence-transformers/allenai-specter')
        self.reranker_model = kwargs.get('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.vectordb_model = kwargs.get('vectordb_model', 'nomic-embed-text')
        self.candidate_count = kwargs.get('candidate_count', 100)
        
        # Advanced settings
        self.batch_size = kwargs.get('batch_size', 32)
        self.reranker_batch_size = kwargs.get('reranker_batch_size', 8)
        self.use_gpu = kwargs.get('use_gpu', True)
        self.sample_for_expansion = kwargs.get('sample_for_expansion', 100)
        
        # Knowledge distillation settings
        self.teacher_model = kwargs.get('teacher_model', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.student_model = kwargs.get('student_model', 'cross-encoder/ms-marco-TinyBERT-L-2-v2')
        
        # Hybrid retrieval settings
        self.rrf_k = kwargs.get('rrf_k', 60)
        self.sparse_weight = kwargs.get('sparse_weight', 0.5)
        
        # Contrastive learning settings
        self.contrastive_base_model = kwargs.get('contrastive_base_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # For backward compatibility, add get method
        self.get = lambda key, default=None: getattr(self, key, default)


class BaseRetriever:
    """Base class for all retriever models with improved functionality"""
    
    def __init__(self, collection_df: pd.DataFrame, config: Optional[RetrievalConfig] = None):
        self.collection_df = collection_df
        self.cord_uids = collection_df['cord_uid'].tolist()
        
        # Use provided config or create default one
        self.config = config if config is not None else RetrievalConfig()
        
        # Initialize shared utilities
        self.cache_manager = CacheManager(self.config.cache_dir)
        self.preprocessor = TextPreprocessor()
        
        # Model name for cache keys (should be overridden by subclasses)
        self.model_name = self.__class__.__name__.lower()
    
    def retrieve(self, query_text: str, top_k: Optional[int] = None) -> List[str]:
        """Retrieve top-k documents for a given query"""
        raise NotImplementedError("Subclasses must implement retrieve()")
    
    def batch_retrieve(self, queries: List[str], top_k: Optional[int] = None, 
                      batch_size: Optional[int] = None) -> List[List[str]]:
        """Retrieve documents for a batch of queries"""
        if top_k is None:
            top_k = self.config.top_k
            
        if batch_size is None:
            batch_size = self.config.batch_size
        
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(queries), batch_size), desc=f"Processing {self.model_name} batches"):
            batch = queries[i:i+batch_size]
            batch_results = []
            for query in batch:
                batch_results.append(self.retrieve(query, top_k))
            results.extend(batch_results)
        
        return results
    
    def get_cache_key(self, suffix: str = "") -> str:
        """Get cache key for this model and collection"""
        return self.cache_manager.get_cache_key(self.collection_df, self.model_name, suffix)


class CachedModelMixin:
    """Mixin for models that cache their core components"""
    
    def _load_or_create_model(self, create_fn, suffix: str = "model"):
        """Load model from cache or create new one"""
        cache_key = self.get_cache_key(suffix)
        return self.cache_manager.get_or_create(cache_key, create_fn)


class EmbeddingMixin:
    """Mixin for models that use embeddings"""
    
    def _create_embeddings(self, texts: List[str], model_fn, batch_size: int = 32):
        """Create embeddings in batches"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch = texts[i:i+batch_size]
            batch_embeddings = model_fn(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings


class VectorStoreMixin:
    """Mixin for models using vector stores"""
    
    def _initialize_vector_store(self, vector_store_manager, documents=None):
        """Initialize vector store with proper error handling"""
        try:
            vector_store, was_loaded = vector_store_manager.get_or_create_db(
                self.collection_df, documents)
            return vector_store, was_loaded
        except Exception as e:
            # Clean up and retry
            collection_name = vector_store_manager.get_collection_name(self.collection_df)
            persist_directory = os.path.join(self.config.vector_db_path, collection_name)
            
            if os.path.exists(persist_directory):
                import shutil
                shutil.rmtree(persist_directory)
                os.makedirs(persist_directory, exist_ok=True)
            
            # Try again
            vector_store, _ = vector_store_manager.get_or_create_db(self.collection_df)
            return vector_store, False