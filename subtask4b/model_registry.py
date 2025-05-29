from models.traditional_methods.bm25 import BM25Retriever
from models.traditional_methods.tfidf import TfidfRetriever

from models.representation_learning.custom_retriever import CustomRetriever
from models.representation_learning.vector_store_retriever import VectorStoreRetriever

from models.reranking_methods.bm25_reranker import BM25Reranker
from models.reranking_methods.tfidf_reranker import TfidfReranker
from models.reranking_methods.custom_retriever_reranker import CustomRetrieverReranker
from models.reranking_methods.vector_store_reranker import VectorStoreReranker

from models.hybrid_methods.multi_stage_hybrid import MultiStageHybrid

from models.query_expansion import QueryExpansionRetriever


class ModelRegistry:
    """Central registry for all retrieval models"""
    
    _models = {
        # Traditional methods
        'bm25': BM25Retriever,
        'tfidf': TfidfRetriever,

        # Representation learning
        'custom_retriever': CustomRetriever,
        'vector_store': VectorStoreRetriever,

        # Reranking methods
        'bm25_reranker': BM25Reranker,
        'tfidf_reranker': TfidfReranker,
        'custom_retriever_reranker': CustomRetrieverReranker,
        'vector_store_reranker': VectorStoreReranker,

        # Hybrid methods
        'multi_stage_hybrid': MultiStageHybrid,

        # Query expansion
        'query_expansion': QueryExpansionRetriever
    }
    
    @classmethod
    def get_model_class(cls, model_name):
        """Get model class by name"""
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(cls._models.keys())}")
        return cls._models[model_name]
    
    @classmethod
    def get_all_models(cls):
        """Get all available model names"""
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name, model_class):
        """Register a new model"""
        cls._models[name] = model_class
    
    @classmethod
    def get_models_by_category(cls):
        """Get models organized by category"""
        return {
            'traditional': ['bm25', 'tfidf'],
            'representation_learning': ['semantic', 'vector_store'],
            'reranking': ['bm25_reranker', 'tfidf_reranker', 'semantic_reranker', 'vector_store_reranker'],
            'hybrid': ['sparse_dense_hybrid', 'neural_hybrid_reranker'],
            'query_expansion': ['query_expansion']
        }