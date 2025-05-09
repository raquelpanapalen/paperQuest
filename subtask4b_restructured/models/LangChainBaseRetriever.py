import os
import shutil
from vector_db.vector_manager import VectorStoreManager
from models.base_class import BaseRetriever


class LangChainBaseRetriever(BaseRetriever):
    """Base class for LangChain retrievers with embeddings"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        
        # Initialize vector store manager
        self.vector_store_manager = VectorStoreManager(self.config)
        
        try:
            # Get or create vector store
            self.vector_store, was_loaded = self.vector_store_manager.get_or_create_db(collection_df)
            
            # Set up the retriever
            self.base_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.candidate_count}
            )
            
        except Exception as e:
            
            # Clean up any potentially problematic directories
            collection_name = self.vector_store_manager.get_collection_name(collection_df)
            persist_directory = os.path.join(self.config.vector_db_path, collection_name)
            
            if os.path.exists(persist_directory):

                shutil.rmtree(persist_directory)
                os.makedirs(persist_directory, exist_ok=True)
            
            # Try again with a fresh database
            self.vector_store, _ = self.vector_store_manager.get_or_create_db(collection_df)
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