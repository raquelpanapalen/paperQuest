import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from tqdm import tqdm
from preprocessing import TextPreprocessor

from models.base import BaseRetriever, CachedModelMixin, prepare_text_simple


class TfidfRetriever(BaseRetriever, CachedModelMixin):
    """TF-IDF retriever using cosine similarity"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "tfidf_baseline"

        self.preprocessor = TextPreprocessor()  
        self.processed_df = self.preprocessor.preprocess_collection(
            collection_df, 
            model_type='bm25'  # Use 'bm25' preprocessing for TF-IDF
        )
        
        # Initialize TF-IDF components
        models = self._load_or_create_model(self._create_tfidf_models)
        self.tfidf_vectorizer = models['vectorizer']
        self.tfidf_matrix = models['matrix']
    
    def _create_tfidf_models(self):
        """Create TF-IDF vectorizer and document matrix"""
        # Get preprocessed documents
        docs = self.processed_df['text'].tolist()
        
        # Initialize TF-IDF vectorizer with good settings for scientific text
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=20000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            stop_words='english',
            norm='l2'  # L2 normalization for cosine similarity
        )
        
        # Create TF-IDF matrix for all documents (no progress bar)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)
        
        return {
            'vectorizer': self.tfidf_vectorizer,
            'matrix': tfidf_matrix
        }
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents for a given query"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Use tweet preprocessing for queries
        processed_query = self.preprocessor.preprocess_tweet_query(query_text)
        
        # Transform query using the fitted vectorizer
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k document indices
        top_indices = np.argsort(-similarities)[:top_k]
        
        # Return cord_uids for top documents
        return [self.cord_uids[idx] for idx in top_indices]


class EnhancedTfidfRetriever(BaseRetriever, CachedModelMixin):
    """Enhanced TF-IDF with reranking"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "enhanced_tfidf"
        
        self.preprocessor = TextPreprocessor()
        self.processed_df = self.preprocessor.preprocess_collection(
            collection_df, 
            model_type='bm25'
        )
        
        # Initialize TF-IDF components
        models = self._load_or_create_model(self._create_tfidf_models)
        self.tfidf_vectorizer = models['vectorizer']
        self.tfidf_matrix = models['matrix']
        
        # Initialize reranker if enabled
        self.use_reranking = getattr(config, 'use_reranking', True) 
        if self.use_reranking:
            self.device = self._get_device()
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(
                getattr(config, 'reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                device=self.device
            )
            self.candidate_count = getattr(config, 'candidate_count', 50)
    
    def _get_device(self):
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _create_tfidf_models(self):
        """Create TF-IDF vectorizer and document matrix"""
        docs = self.processed_df['text'].tolist()
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=20000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english',
            norm='l2'
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)
        
        return {
            'vectorizer': self.tfidf_vectorizer,
            'matrix': tfidf_matrix
        }
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve with optional reranking"""
        if top_k is None:
            top_k = self.config.top_k
        
        processed_query = self.preprocessor.preprocess_tweet_query(query_text)
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        if not self.use_reranking:
            top_indices = np.argsort(-similarities)[:top_k]
            return [self.cord_uids[idx] for idx in top_indices]
        
        # Get more candidates for reranking
        retrieval_count = min(self.candidate_count, len(self.cord_uids))
        top_indices = np.argsort(-similarities)[:retrieval_count]
        
        # Prepare for reranking
        candidates = []
        pairs = []
        for idx in top_indices:
            if idx < len(self.cord_uids):
                uid = self.cord_uids[idx]
                candidates.append(uid)
                
                try:
                    row = self.collection_df.iloc[idx]
                    # FIXED: Use prepare_text_simple instead of arbitrary truncation
                    doc_text = prepare_text_simple(row.get('title', ''), row.get('abstract', ''))
                    pairs.append([query_text, doc_text])
                except:
                    pairs.append([query_text, ""])
        
        if not pairs:
            return candidates[:top_k]
        
        # Rerank
        try:
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return [uid for uid, _ in scored_candidates[:top_k]]
        except:
            return candidates[:top_k]