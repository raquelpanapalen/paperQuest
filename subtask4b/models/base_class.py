from tqdm import tqdm

import os



#################################
# Configuration Management
#################################

class RetrievalConfig:
    """Central configuration class for all retrieval models"""
    
    def __init__(self, **kwargs):
        # General settings
        self.cache_dir = kwargs.get('cache_dir', 'cache')
        self.top_k = kwargs.get('top_k', 5)
        
        # Dataset sampling settings
        self.collection_sample_size = kwargs.get('collection_sample_size', None)  # Number of papers to sample from collection
        self.collection_sample_seed = kwargs.get('collection_sample_seed', 42)    # Seed for reproducible sampling

        # Vector database settings
        self.vector_db_path = kwargs.get('vector_db_path', os.path.join(self.cache_dir, "chroma_db"))
        self.vector_db_type = kwargs.get('vector_db_type', 'chroma')
        
        # Create paths if they don't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Model-specific settings
        self.embedding_model = kwargs.get('embedding_model', 'sentence-transformers/allenai-specter')
        self.reranker_model = kwargs.get('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.langchain_embedding = kwargs.get('langchain_embedding', 'nomic-embed-text')
        self.candidate_count = kwargs.get('candidate_count', 100)
        
        # Advanced settings
        self.batch_size = kwargs.get('batch_size', 32)
        self.reranker_batch_size = kwargs.get('reranker_batch_size', 8)
        self.use_gpu = kwargs.get('use_gpu', True)
        self.sample_for_expansion = kwargs.get('sample_for_expansion', 100)



#################################
# Base Retriever Class
#################################

class BaseRetriever:
    """Base class for all retriever models"""
    
    def __init__(self, collection_df, config=None):
        self.collection_df = collection_df
        self.cord_uids = collection_df['cord_uid'].tolist()
        
        # Use provided config or create default one
        self.config = config if config is not None else RetrievalConfig()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query"""
        raise NotImplementedError("Subclasses must implement retrieve()")
    
    def batch_retrieve(self, queries, top_k=None, batch_size=None):
        """Retrieve documents for a batch of queries"""
        if top_k is None:
            top_k = self.config.top_k
            
        if batch_size is None:
            batch_size = self.config.batch_size
        
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(queries), batch_size), desc="Processing batches"):
            batch = queries[i:i+batch_size]
            batch_results = []
            for query in batch:
                batch_results.append(self.retrieve(query, top_k))
            results.extend(batch_results)
        
        return results
    