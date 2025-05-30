from rank_bm25 import BM25Okapi
import numpy as np
from preprocessing import TextPreprocessor
from models.base import BaseRetriever, CachedModelMixin


class BM25Retriever(BaseRetriever, CachedModelMixin):
    """Basic BM25 retriever using keyword matching"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "bm25_baseline"
        
        self.preprocessor = TextPreprocessor()
        self.processed_df = self.preprocessor.prepare_collection_cleaned(collection_df)
        
        self.bm25 = self._load_or_create_model(self._create_bm25)
    
    def _create_bm25(self):
        """Create BM25 index from processed documents"""
        corpus = self.processed_df['text'].tolist()
        tokenized_corpus = [doc.split() for doc in corpus]
        return BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve documents using BM25 scoring"""
        if top_k is None:
            top_k = self.config.top_k
        
        processed_query = self.preprocessor.social_media_preprocessing(query_text)
        tokenized_query = processed_query.split()
        
        # Score documents
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-doc_scores)[:top_k]
        
        # Return valid results
        valid_indices = [idx for idx in top_indices if idx < len(self.cord_uids)]
        return [self.cord_uids[idx] for idx in valid_indices]