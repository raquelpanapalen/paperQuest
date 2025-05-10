import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from tqdm import tqdm

from models.base import BaseRetriever, CachedModelMixin


class TfidfRetriever(BaseRetriever, CachedModelMixin):
    """TF-IDF retriever using cosine similarity"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "tfidf_baseline"
        
        # Initialize TF-IDF components
        models = self._load_or_create_model(self._create_tfidf_models)
        self.tfidf_vectorizer = models['vectorizer']
        self.tfidf_matrix = models['matrix']
    
    def _create_tfidf_models(self):
        """Create TF-IDF vectorizer and document matrix"""
        # Get preprocessed documents
        docs = self.collection_df['text'].tolist()
        
        # Initialize TF-IDF vectorizer with good settings for scientific text
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=20000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            stop_words='english',
            norm='l2'  # L2 normalization for cosine similarity
        )
        
        # Create TF-IDF matrix for all documents
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            tqdm(docs, desc="Vectorizing documents")
        )
        
        return {
            'vectorizer': self.tfidf_vectorizer,
            'matrix': tfidf_matrix
        }
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Transform query using the fitted vectorizer
        query_vector = self.tfidf_vectorizer.transform([query_text])
        
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k document indices
        top_indices = np.argsort(-similarities)[:top_k]
        
        # Return cord_uids for top documents
        return [self.cord_uids[idx] for idx in top_indices]


class EnhancedTfidfRetriever(BaseRetriever, CachedModelMixin):
    """Enhanced TF-IDF retriever with preprocessing and query expansion"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "enhanced_tfidf"
        
        # Initialize TF-IDF components
        models = self._load_or_create_model(self._create_tfidf_models)
        self.tfidf_vectorizer = models['vectorizer']
        self.tfidf_matrix = models['matrix']
        self.feature_names = models['feature_names']
    
    def _create_tfidf_models(self):
        """Create TF-IDF vectorizer with preprocessing"""
        # Preprocess documents for better TF-IDF performance
        processed_docs = []
        for text in tqdm(self.collection_df['text'], desc="Preprocessing documents"):
            processed = self.preprocessor.preprocess_for_bm25(text)
            processed_docs.append(processed)
        
        # Initialize TF-IDF with enhanced settings
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=30000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 3),  # Include trigrams for better coverage
            sublinear_tf=True,  # Apply sublinear tf scaling
            norm='l2',
            strip_accents='unicode',
            token_pattern=r'\b\w+\b'  # Better tokenization pattern
        )
        
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            tqdm(processed_docs, desc="Creating TF-IDF matrix")
        )
        
        # Get feature names for query expansion
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        return {
            'vectorizer': self.tfidf_vectorizer,
            'matrix': tfidf_matrix,
            'feature_names': feature_names
        }
    
    def _extract_important_terms(self, vector, top_n=10):
        """Extract important terms from a TF-IDF vector"""
        # Get non-zero elements
        if isinstance(vector, sp.spmatrix):
            vector = vector.toarray().flatten()
        
        # Get top terms by TF-IDF score
        top_indices = np.argsort(-vector)[:top_n]
        top_terms = [(self.feature_names[idx], vector[idx]) 
                     for idx in top_indices if vector[idx] > 0]
        
        return top_terms
    
    def _expand_query(self, query_text, expansion_factor=0.3):
        """Expand query using pseudo-relevance feedback"""
        # First, get initial results
        query_vector = self.tfidf_vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top 3 documents for feedback
        top_docs_idx = np.argsort(-similarities)[:3]
        
        # Extract important terms from top documents
        expansion_terms = []
        for idx in top_docs_idx:
            doc_vector = self.tfidf_matrix[idx]
            terms = self._extract_important_terms(doc_vector, top_n=5)
            expansion_terms.extend([term for term, score in terms])
        
        # Remove duplicates and terms already in query
        query_terms = set(query_text.lower().split())
        expansion_terms = [term for term in set(expansion_terms) 
                          if term.lower() not in query_terms]
        
        # Add top expansion terms to query
        num_terms = max(1, int(len(query_terms) * expansion_factor))
        expanded_query = query_text + " " + " ".join(expansion_terms[:num_terms])
        
        return expanded_query
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve with query expansion and preprocessing"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Preprocess query
        preprocessed_query = self.preprocessor.preprocess_for_bm25(query_text)
        
        # Expand query
        expanded_query = self._expand_query(preprocessed_query)
        
        # Transform expanded query
        query_vector = self.tfidf_vectorizer.transform([expanded_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k documents
        top_indices = np.argsort(-similarities)[:top_k]
        
        return [self.cord_uids[idx] for idx in top_indices]


class HybridTfidfBM25Retriever(BaseRetriever, CachedModelMixin):
    """Hybrid retriever combining TF-IDF and BM25 scores"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "hybrid_tfidf_bm25"
        
        # Initialize both models
        from models.bm25 import BM25Retriever
        
        self.tfidf_retriever = TfidfRetriever(collection_df, config)
        self.bm25_retriever = BM25Retriever(collection_df, config)
        
        # Weight for combining scores (can be tuned)
        self.tfidf_weight = 0.6
        self.bm25_weight = 0.4
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve using combined TF-IDF and BM25 scores"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Get more candidates from each model
        candidate_count = min(self.config.candidate_count, len(self.cord_uids))
        
        # Get TF-IDF results
        tfidf_results = self.tfidf_retriever.retrieve(query_text, top_k=candidate_count)
        
        # Get BM25 results  
        bm25_results = self.bm25_retriever.retrieve(query_text, top_k=candidate_count)
        
        # Create score dictionaries
        tfidf_scores = {uid: 1.0 - (i / candidate_count) 
                       for i, uid in enumerate(tfidf_results)}
        bm25_scores = {uid: 1.0 - (i / candidate_count) 
                      for i, uid in enumerate(bm25_results)}
        
        # Combine scores
        all_uids = set(tfidf_results + bm25_results)
        combined_scores = {}
        
        for uid in all_uids:
            tfidf_score = tfidf_scores.get(uid, 0.0)
            bm25_score = bm25_scores.get(uid, 0.0)
            
            # Weighted combination
            combined_scores[uid] = (self.tfidf_weight * tfidf_score + 
                                   self.bm25_weight * bm25_score)
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return [uid for uid, score in sorted_results[:top_k]]