from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from models.base import BaseRetriever, CachedModelMixin


class BM25Retriever(BaseRetriever, CachedModelMixin):
    """Baseline BM25 retriever"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "bm25_baseline"
        self.bm25 = self._load_or_create_model(self._create_bm25)
    
    def _create_bm25(self):
        """Create BM25 model"""
        corpus = self.collection_df['text'].tolist()
        tokenized_corpus = [doc.split() for doc in corpus]
        return BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query"""
        if top_k is None:
            top_k = self.config.top_k
            
        tokenized_query = query_text.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-doc_scores)[:top_k]
        
        # Ensure indices are within bounds
        valid_indices = [idx for idx in top_indices if idx < len(self.cord_uids)]
        return [self.cord_uids[idx] for idx in valid_indices]


class EnhancedBM25Retriever(BaseRetriever, CachedModelMixin):
    """Enhanced BM25 retriever with query expansion"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "enhanced_bm25"
        
        # Load cached models
        models = self._load_or_create_model(self._create_models)
        self.bm25 = models['bm25']
        self.tfidf_vectorizer = models['tfidf_vectorizer']
        self.tfidf_matrix = models['tfidf_matrix']
        self.feature_names = models['feature_names']
    
    def _create_models(self):
        """Create all required models"""
        processed_docs = []
        for text in tqdm(self.collection_df['text'], desc="Preprocessing documents"):
            processed = self.preprocessor.preprocess_for_bm25(text)
            processed_docs.append(processed)
        
        # Create BM25 model
        tokenized_corpus = [doc.split() for doc in processed_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Create TF-IDF model for query expansion
        tfidf_vectorizer = TfidfVectorizer(max_features=20000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        return {
            'bm25': bm25,
            'tfidf_vectorizer': tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'feature_names': feature_names
        }
    
    def _pseudo_relevance_feedback(self, query, top_k=3, terms_to_add=5):
        """Implement pseudo-relevance feedback for query expansion"""
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-doc_scores)[:top_k]
        
        # Ensure indices are within bounds
        valid_indices = [idx for idx in top_indices if idx < self.tfidf_matrix.shape[0]]
        
        # Extract important terms from top documents
        expansion_terms = []
        
        for idx in valid_indices:
            doc_vector = self.tfidf_matrix[idx]
            feature_index = doc_vector.nonzero()[1]
            tfidf_scores = zip(feature_index, [doc_vector[0, x] for x in feature_index])
            sorted_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
            
            for term_idx, score in sorted_terms[:terms_to_add]:
                term = self.feature_names[term_idx]
                if term not in tokenized_query and term not in expansion_terms:
                    expansion_terms.append(term)
        
        # Add expansion terms to query
        expanded_query = query + ' ' + ' '.join(expansion_terms[:terms_to_add])
        return expanded_query
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents using enhanced BM25"""
        if top_k is None:
            top_k = self.config.top_k
            
        # Preprocess query
        preprocessed_query = self.preprocessor.preprocess_for_bm25(query_text)
        
        # Apply query expansion
        expanded_query = self._pseudo_relevance_feedback(preprocessed_query)
        
        # Final retrieval with expanded query
        tokenized_query = expanded_query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-doc_scores)[:top_k]
        
        # Ensure indices are within bounds
        valid_indices = [idx for idx in top_indices if idx < len(self.cord_uids)]
        return [self.cord_uids[idx] for idx in valid_indices]