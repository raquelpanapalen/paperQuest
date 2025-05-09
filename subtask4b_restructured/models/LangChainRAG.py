from models.LangChainBaseRetriever import LangChainBaseRetriever

class LangChainRAGRetriever(LangChainBaseRetriever):

    """Simple RAG retriever using LangChain with embeddings only (no reranking)"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
