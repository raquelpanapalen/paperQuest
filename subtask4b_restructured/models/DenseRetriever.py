import torch
from models.base_class import BaseRetriever

import pickle
import os
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers import util



class DenseRetriever(BaseRetriever):
    """Dense retriever using transformer embeddings"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        
        # Initialize model and embeddings
        self._init_model()
        self.doc_embeddings = self._get_document_embeddings()
    
    def _init_model(self):
        """Initialize the embedding model"""
        model_name = self.config.embedding_model
      
            
        self.model = SentenceTransformer(model_name)
        
        if self.config.use_gpu and torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda"))
    
    def _get_document_embeddings(self):
        """Create or load document embeddings"""
        model_name = self.config.embedding_model.replace("/", "_")
        cache_file = os.path.join(self.config.cache_dir, f'doc_embeddings_{model_name}.pkl')
        
        if os.path.exists(cache_file):
          
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        

        # Prepare document texts
        docs = self.collection_df['text'].tolist()
        
        # Create embeddings in batches
        batch_size = self.config.batch_size
        embeddings = []
        
        for i in tqdm(range(0, len(docs), batch_size), desc="Creating embeddings"):
            batch = docs[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        
        # Cache embeddings to disk
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents using dense embeddings"""
        if top_k is None:
            top_k = self.config.top_k
                    
        # Encode query
        query_embedding = self.model.encode(query_text)
        
        # Calculate similarity scores
        cos_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        # Get top-k document indices
        top_indices = torch.topk(cos_scores, k=top_k).indices.cpu().numpy()
        
        # Return top document IDs
        return [self.cord_uids[idx] for idx in top_indices]