import os
import hashlib
import shutil
from tqdm import tqdm

import torch
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from langchain.schema import Document

# Detect device: prefer MPS on Mac M-series, else CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# Disable Chroma telemetry
client_settings = Settings(anonymized_telemetry=False)


#################################
# Vector Store Management
#################################
class VectorStoreManager:
    """Manager for handling persistent vector database storage with cosine similarity"""

    def __init__(self, config):
        self.config = config
        self.db_path = config.vector_db_path
        os.makedirs(self.db_path, exist_ok=True)
        self._embedding_model = None

    @property
    def embedding_model(self):
        """Lazy-loaded embedding model"""

        if self._embedding_model is None:
            
            self._embedding_model = OllamaEmbeddings(model=self.config.vectordb_model)

        return self._embedding_model

    def get_collection_name(self, collection_df):
        """Generate a unique collection name based on dataset and model"""
        max_sample = min(100, len(collection_df))
        ids_sample = "_".join(collection_df['cord_uid'].iloc[:max_sample])
        data_hash = hashlib.md5(ids_sample.encode()).hexdigest()[:8]
        collection_size = len(collection_df)
        embedder_name = self.config.vectordb_model.replace('/', '_').replace('-', '_')
        return f"cord19_{embedder_name}_{collection_size}_{data_hash}"

    def get_or_create_db(self, collection_df, documents=None):
        """Get an existing vector DB or create a new one using cosine via HNSW"""
        collection_name = self.get_collection_name(collection_df)
        persist_directory = os.path.join(self.db_path, collection_name)

        # Use HNSW index with cosine distance
        collection_metadata = {"hnsw:space": "cosine"}

        if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
            chroma_files = [f for f in os.listdir(persist_directory)
                           if f.endswith('.sqlite3') or f == 'chroma.sqlite3']
            if chroma_files:
                try:
                    vector_store = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embedding_model,
                        persist_directory=persist_directory,
                        collection_metadata=collection_metadata,
                        client_settings=client_settings
                    )
                    # Verify existence
                    _ = vector_store._collection.count()
                    return vector_store, True
                except Exception:
                    # Corrupted store: rebuild
                    shutil.rmtree(persist_directory)
                    os.makedirs(persist_directory, exist_ok=True)
        else:
            os.makedirs(persist_directory, exist_ok=True)

        # Prepare documents if not provided
        if documents is None:
            documents = self._prepare_documents(collection_df)

        # Create store with cosine metric
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
            collection_metadata=collection_metadata,
            client_settings=client_settings
        )
        vector_store.persist()
        return vector_store, False

    def _prepare_documents(self, collection_df):
        """Prepare documents from collection dataframe"""
        documents = []
        for _, row in tqdm(collection_df.iterrows(), total=len(collection_df), desc="Creating documents"):
            content = f"Title: {row['title']}\nAbstract: {row['abstract']}"
            doc = Document(
                page_content=content,
                metadata={
                    "cord_uid": row['cord_uid'],
                    "title": row['title']
                }
            )
            documents.append(doc)
        return documents
