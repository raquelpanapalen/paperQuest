from models.base_class import BaseRetriever
from rank_bm25 import BM25Okapi
import os
import numpy as np
import pickle
import hashlib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import regex as re
import nltk

class EnhancedBM25Retriever(BaseRetriever):
    """Enhanced BM25 retriever with query expansion"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        
        # Initialize models
        self._init_models()
    
    def _get_cache_filename(self):
        """Generate a unique cache filename based on collection size and content"""
        # Create a unique identifier for this specific collection
        collection_size = len(self.collection_df)
        
        # Use first few cord_uids as a fingerprint of the dataset
        max_sample = min(100, len(self.collection_df))
        ids_sample = "_".join(self.collection_df['cord_uid'].iloc[:max_sample])
        data_hash = hashlib.md5(ids_sample.encode()).hexdigest()[:8]
        
        # Include collection size in the cache filename
        cache_file = os.path.join(self.config.cache_dir, f'enhanced_bm25_{collection_size}_{data_hash}.pkl')
        return cache_file
    
    def _init_models(self):
        """Initialize all required models"""
        cache_file = self._get_cache_filename()
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                models = pickle.load(f)
                self.bm25 = models['bm25']
                self.tfidf_vectorizer = models['tfidf_vectorizer']
                self.tfidf_matrix = models['tfidf_matrix']
                self.feature_names = models['feature_names']
        else:
            # Download NLTK resources if needed
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            # Preprocess collection
            self.ps = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            
            processed_docs = []
            for text in tqdm(self.collection_df['text'], desc="Preprocessing documents"):
                processed = self._preprocess_text(text)
                processed_docs.append(processed)
            
            # Create BM25 model
            tokenized_corpus = [doc.split() for doc in processed_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            # Create TF-IDF model for query expansion
            self.tfidf_vectorizer = TfidfVectorizer(max_features=20000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_docs)
            self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Cache models
            models = {
                'bm25': self.bm25,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'feature_names': self.feature_names
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(models, f)
    
    def _preprocess_text(self, text):
        """Preprocess text for BM25"""
        # Import NLTK resources if not already imported
        if not hasattr(self, 'ps'):
            # Download NLTK resources if needed
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            self.ps = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and apply stemming
        tokens = [self.ps.stem(word) for word in tokens if word not in self.stop_words]
        
        # Join tokens back into a string
        return ' '.join(tokens)
    
    def _pseudo_relevance_feedback(self, query, top_k=3, terms_to_add=5):
        """Implement pseudo-relevance feedback for query expansion"""
        # Get initial top documents
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
        preprocessed_query = self._preprocess_text(query_text)
        
        # Apply query expansion with pseudo-relevance feedback
        expanded_query = self._pseudo_relevance_feedback(preprocessed_query)
        
        # Final retrieval with expanded query
        tokenized_query = expanded_query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(-doc_scores)[:top_k]
        
        # Ensure indices are within bounds
        valid_indices = [idx for idx in top_indices if idx < len(self.cord_uids)]
        
        return [self.cord_uids[idx] for idx in valid_indices]