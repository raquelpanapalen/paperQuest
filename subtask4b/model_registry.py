from models.bm25 import BM25Retriever, EnhancedBM25Retriever
from models.neural_models import DenseRetriever, NeuralReranker
from models.langchain_models import (
    LangChainRAGRetriever, 
    LangChainRerankerRetriever,
    LangChainQueryExpansionRetriever
)
from models.tfidf import (
    TfidfRetriever,
    EnhancedTfidfRetriever,
    HybridTfidfBM25Retriever
)

from models.advanced_neural_models import (
    DistilledNeuralReranker,
    HybridNeuralReranker,
    ContrastiveReranker
)


class ModelRegistry:
    """Central registry for all retrieval models"""
    
    _models = {
        'bm25': BM25Retriever,
        'enhanced_bm25': EnhancedBM25Retriever,
        'tfidf': TfidfRetriever,
        'enhanced_tfidf': EnhancedTfidfRetriever,
        'hybrid_tfidf_bm25': HybridTfidfBM25Retriever,

        'dense': DenseRetriever,
        'neural_rerank': NeuralReranker,
        'distilled_rerank': DistilledNeuralReranker,
        'hybrid_rerank': HybridNeuralReranker,
        'contrastive_rerank': ContrastiveReranker,

        'langchain_rag': LangChainRAGRetriever,
        'langchain_reranker': LangChainRerankerRetriever,
        'langchain_query_expansion': LangChainQueryExpansionRetriever,
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