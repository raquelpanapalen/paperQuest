import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from preprocessing import TextPreprocessor
from models.base import BaseRetriever, CachedModelMixin, prepare_text_simple


class TfidfReranker(BaseRetriever, CachedModelMixin):
    """TF-IDF retrieval with neural reranking"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "tfidf_reranker"
        
        # Preprocess collection 
        self.preprocessor = TextPreprocessor()
        self.processed_df = self.preprocessor.prepare_collection_cleaned(collection_df)
        
        # Create TF-IDF models
        models = self._load_or_create_model(self._create_tfidf_models)
        self.tfidf_vectorizer = models['vectorizer']
        self.tfidf_matrix = models['matrix']
        
        # Initialize reranker
        self.use_reranking = getattr(config, 'use_reranking', True) 
        if self.use_reranking:
            self.device = self._get_device()
            self.reranker = CrossEncoder(
                getattr(config, 'reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                device=self.device
            )
            self.candidate_count = getattr(config, 'candidate_count', 50)
    
    def _get_device(self):
        """Get optimal device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _create_tfidf_models(self):
        """Create TF-IDF vectorizer and matrix"""
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
        """Retrieve with TF-IDF then rerank"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Get TF-IDF similarities 
        processed_query = self.preprocessor.social_media_preprocessing(query_text)
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Return TF-IDF results if no reranking
        if not self.use_reranking:
            top_indices = np.argsort(-similarities)[:top_k]
            return [self.cord_uids[idx] for idx in top_indices]
        
        # Get candidates for reranking
        retrieval_count = min(self.candidate_count, len(self.cord_uids))
        top_indices = np.argsort(-similarities)[:retrieval_count]
        
        # Prepare candidates and pairs
        candidates = []
        pairs = []
        for idx in top_indices:
            if idx < len(self.cord_uids):
                uid = self.cord_uids[idx]
                candidates.append(uid)
                
                try:
                    row = self.collection_df.iloc[idx]
                    doc_text = prepare_text_simple(row.get('title', ''), row.get('abstract', ''))
                    pairs.append([query_text, doc_text])
                except:
                    pairs.append([query_text, ""])
        
        if not pairs:
            return candidates[:top_k]
        
        # Rerank with neural model
        try:
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return [uid for uid, _ in scored_candidates[:top_k]]
        except Exception as e:
            print(f"TF-IDF reranking failed: {e}")
            return candidates[:top_k]