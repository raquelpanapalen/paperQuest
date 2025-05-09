from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

import hashlib
import shutil
import os
from tqdm import tqdm

#################################
# Vector Store Management
#################################

class VectorStoreManager:
    """Manager for handling persistent vector database storage"""
    
    def __init__(self, config):
        """Initialize the vector store manager"""
        self.config = config
        self.db_path = config.vector_db_path
        
        # Create the directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)        
        
        # Initialize embedding model lazily
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Lazy-loaded embedding model"""

        if self._embedding_model is None:
            
            self._embedding_model = OllamaEmbeddings(model=self.config.langchain_embedding)

        return self._embedding_model
    
    def get_collection_name(self, collection_df):
        """Generate a unique collection name based on dataset and model"""
        
        # Use first 100 IDs (or all if fewer) as a fingerprint of the dataset
        max_sample = min(100, len(collection_df))
        ids_sample = "_".join(collection_df['cord_uid'].iloc[:max_sample])
        data_hash = hashlib.md5(ids_sample.encode()).hexdigest()[:8]
        
        # Include collection size in the name
        collection_size = len(collection_df)
        embedder_name = self.config.langchain_embedding.replace('/', '_').replace('-', '_')
        
        return f"cord19_{embedder_name}_{collection_size}_{data_hash}"
    
    def get_or_create_db(self, collection_df, documents=None):
        """Get an existing vector DB or create a new one if it doesn't exist"""
            
        collection_name = self.get_collection_name(collection_df)
        persist_directory = os.path.join(self.db_path, collection_name)
        
        # Check if the vector store already exists as a directory
        if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
            # Check for Chroma's specific files
            chroma_files = [f for f in os.listdir(persist_directory) 
                           if f.endswith('.sqlite3') or f == 'chroma.sqlite3']
            
            if chroma_files:

                try:
                    # Load existing database using Chroma's native persistence
                    vector_store = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embedding_model,
                        persist_directory=persist_directory
                    )
                    doc_count = vector_store._collection.count()

                    return vector_store, True
                except Exception as e:

                    shutil.rmtree(persist_directory)
                    os.makedirs(persist_directory, exist_ok=True)
        else:
            os.makedirs(persist_directory, exist_ok=True)
        
        
        # Prepare documents if not provided
        if documents is None:
            documents = self._prepare_documents(collection_df)
        
        # Create the vector store with Chroma's native persistence
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # Ensure it's persisted
        vector_store.persist()
        
        return vector_store, False
    
    def _prepare_documents(self, collection_df):
        """Prepare documents from collection dataframe"""

            
        documents = []
        
        for _, row in tqdm(collection_df.iterrows(), total=len(collection_df), desc="Creating documents"):
            # Format content with title and abstract
            content = f"""Title: {row['title']}\nAbstract: {row['abstract']}"""
            
            # Create Document object with metadata
            doc = Document(
                page_content=content, 
                metadata={
                    "cord_uid": row['cord_uid'],
                    "title": row['title'],
                    "journal": row.get('journal', ""),
                    "publish_time": row.get('publish_time', "")
                }
            )
            documents.append(doc)
        
        return documents