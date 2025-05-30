import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import TextPreprocessor
from models.base import BaseRetriever, CachedModelMixin


class TfidfRetriever(BaseRetriever, CachedModelMixin):
    """TF-IDF retriever using cosine similarity"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "tfidf_baseline"

        self.preprocessor = TextPreprocessor()  

        self.processed_df = self.preprocessor.prepare_collection_cleaned(collection_df)
        
        # Initialize TF-IDF components
        models = self._load_or_create_model(self._create_tfidf_models)
        self.tfidf_vectorizer = models['vectorizer']
        self.tfidf_matrix = models['matrix']
    
    def _create_tfidf_models(self):
        """Create TF-IDF vectorizer and document matrix"""
        docs = self.processed_df['text'].tolist()
        
        # Configure TF-IDF for scientific text
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=20000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english',
            norm='l2'
        )
        
        # Create document matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)
        
        return {
            'vectorizer': self.tfidf_vectorizer,
            'matrix': tfidf_matrix
        }
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve documents using TF-IDF cosine similarity"""
        if top_k is None:
            top_k = self.config.top_k
        
        processed_query = self.preprocessor.social_media_preprocessing(query_text)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = np.argsort(-similarities)[:top_k]
        
        return [self.cord_uids[idx] for idx in top_indices]