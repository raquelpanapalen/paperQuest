# subtask4b/model_registry.py (Updated)
from models.bm25 import BM25Retriever, EnhancedBM25Retriever
from models.neural_models import DenseRetriever, NeuralReranker, HybridNeuralReranker
from models.langchain_models import (
    LangChainRAGRetriever, 
    LangChainRerankerRetriever,
    LangChainQueryExpansionRetriever
)
from models.tfidf import TfidfRetriever
from models.hybrid_retriever import HybridRetriever

class ModelRegistry:
    """Central registry for all retrieval models"""
    
    _models = {
        'bm25': BM25Retriever,
        'enhanced_bm25': EnhancedBM25Retriever,
        'tfidf': TfidfRetriever,

        'dense': DenseRetriever,
        'neural_rerank': NeuralReranker,
        'hybrid_rerank': HybridNeuralReranker,

        'langchain_rag': LangChainRAGRetriever,
        'langchain_reranker': LangChainRerankerRetriever,
        'langchain_query_expansion': LangChainQueryExpansionRetriever,

        'hybrid_retriever': HybridRetriever
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