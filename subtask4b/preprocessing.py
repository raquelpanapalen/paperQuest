# subtask4b/preprocessing.py
import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextPreprocessor:
    """Model-aware text preprocessing with minimal transformations"""
    
    def __init__(self):
        self._ensure_nltk_resources()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def _ensure_nltk_resources(self):
        """Download NLTK resources if needed"""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def preprocess_for_neural(self, text):
        """Minimal preprocessing for neural models - preserve as much as possible"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Only normalize excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing spaces
        return text.strip()
    
    def preprocess_for_bm25(self, text):
        """Lighter preprocessing for BM25 - preserve scientific notation"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs but keep scientific URLs patterns
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Lighter punctuation removal - keep hyphens and dots in numbers
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)  # Remove dots not between numbers
        
        # Tokenize
        tokens = text.split()
        
        # Skip stemming for now - it often hurts performance
        # Just remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words or len(word) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_tweet_query(self, text):
        """Essential Twitter-specific preprocessing for queries"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # Remove URLs (Twitter specifics)
        text = re.sub(r'http\S+|www\.\S+|t\.co/\S+', '', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Handle hashtags - keep content but remove #
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess_collection(self, collection_df, use_columns=None, model_type='neural'):
        """Preprocess collection based on model type"""
        df = collection_df.copy()
        
        # Essential columns
        essential_columns = ['cord_uid', 'title', 'abstract']
        
        # Determine columns to keep
        if use_columns is None:
            keep_columns = essential_columns + ['authors', 'journal', 'publish_time', 'doi']
        else:
            keep_columns = list(set(essential_columns + use_columns))
        
        # Keep only existing columns
        keep_columns = [col for col in keep_columns if col in df.columns]
        df = df[keep_columns]
        
        # Fill missing values
        df['title'] = df['title'].fillna("")
        df['abstract'] = df['abstract'].fillna("")
        
        # Apply preprocessing based on model type
        if model_type == 'neural':
            # Minimal preprocessing for neural models
            df['text'] = df.apply(lambda x: f"{x['title']} {x['abstract']}", axis=1)
            df['text'] = df['text'].apply(self.preprocess_for_neural)
        else:
            # BM25 preprocessing
            df['title'] = df['title'].apply(self.preprocess_for_bm25)
            df['abstract'] = df['abstract'].apply(self.preprocess_for_bm25)
            df['text'] = df.apply(lambda x: f"{x['title']} {x['abstract']}", axis=1)
        
        return df
    
    def preprocess_collection_no_limits(self, collection_df, use_columns=None):
        """
        Preprocess collection without any token limits - using prepare_text_simple approach
        This is the new recommended method for better performance
        """
        from models.base import prepare_text_simple
        
        df = collection_df.copy()
        
        # Essential columns
        essential_columns = ['cord_uid', 'title', 'abstract']
        
        # Determine columns to keep
        if use_columns is None:
            keep_columns = essential_columns + ['authors', 'journal', 'publish_time', 'doi']
        else:
            keep_columns = list(set(essential_columns + use_columns))
        
        # Keep only existing columns
        keep_columns = [col for col in keep_columns if col in df.columns]
        df = df[keep_columns]
        
        # Fill missing values
        df['title'] = df['title'].fillna("")
        df['abstract'] = df['abstract'].fillna("")
        
        # Use prepare_text_simple for combining - no arbitrary limits
        df['text'] = df.apply(
            lambda x: prepare_text_simple(x['title'], x['abstract']), 
            axis=1
        )
        
        return df