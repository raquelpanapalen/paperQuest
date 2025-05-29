import pandas as pd
import os

from cache_manager import CacheManager
from preprocessing import TextPreprocessor


def prepare_text_simple(title, abstract):
    """
    Simple text preparation
    """
    title = str(title) if title and not pd.isna(title) else ""
    abstract = str(abstract) if abstract and not pd.isna(abstract) else ""
    return f"{title} {abstract}".strip()


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
    
        # Hybrid retrieval settings
        self.rrf_k = kwargs.get('rrf_k', 60)
        self.sparse_weight = kwargs.get('sparse_weight', 0.5)
        
        # Progress bar settings
        self.show_progress = kwargs.get('show_progress', True)
        self.progress_desc = kwargs.get('progress_desc', None)
        
        # For backward compatibility, add get method
        self.get = lambda key, default=None: getattr(self, key, default)


class BaseRetriever:
    """Base class for all retriever models"""
    
    def __init__(self, collection_df, config=None):
        self.collection_df = collection_df
        self.cord_uids = collection_df['cord_uid'].tolist()
        
        # Use provided config or create default one
        self.config = config if config is not None else RetrievalConfig()
        
        # Initialize shared utilities
        self.cache_manager = CacheManager(self.config.cache_dir)
        self.preprocessor = TextPreprocessor()
        
        # Model name for cache keys (should be overridden by subclasses)
        self.model_name = self.__class__.__name__.lower()
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query"""
        raise NotImplementedError("Subclasses must implement retrieve()")
    
    def get_cache_key(self, suffix=""):
        """Get cache key for this model and collection"""
        return self.cache_manager.get_cache_key(self.collection_df, self.model_name, suffix)


class CachedModelMixin:
    """Mixin for models that cache their core components"""
    
    def _load_or_create_model(self, create_fn, suffix="model"):
        """Load model from cache or create new one"""
        cache_key = self.get_cache_key(suffix)
        return self.cache_manager.get_or_create(cache_key, create_fn)


class EmbeddingMixin:
    """Mixin for models that use embeddings"""
    
    def _create_embeddings(self, texts, model_fn, batch_size=32, 
                          desc="Creating embeddings", show_progress=None):
        """Create embeddings in batches"""
        if show_progress is None:
            show_progress = getattr(self.config, 'show_progress', True)
            
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = model_fn(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings


class VectorStoreMixin:
    """Mixin for models using vector stores"""
    
    def _initialize_vector_store(self, vector_store_manager, documents=None, show_progress=None):
        """Initialize vector store with proper error handling"""
        if show_progress is None:
            show_progress = getattr(self.config, 'show_progress', True)
            
        try:
            if show_progress:
                print(f"Initializing vector store for {self.model_name}...")
                
            vector_store, was_loaded = vector_store_manager.get_or_create_db(
                self.collection_df, documents)
            
            if show_progress:
                status = "loaded from cache" if was_loaded else "created"
                print(f"Vector store {status} successfully.")
                
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
            if show_progress:
                print(f"Retrying vector store initialization...")
                
            vector_store, _ = vector_store_manager.get_or_create_db(self.collection_df)
            return vector_store, False