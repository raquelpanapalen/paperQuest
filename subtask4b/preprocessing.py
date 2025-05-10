import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import List, Optional


class TextPreprocessor:
    """Centralized text preprocessing for all models"""
    
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
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing extra spaces, special characters, etc."""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading and trailing spaces
        return text.strip()
    
    def preprocess_for_bm25(self, text: str) -> str:
        """Preprocess text specifically for BM25"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and apply stemming
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def preprocess_collection(self, collection_df: pd.DataFrame, 
                            use_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Preprocess the collection dataframe"""
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
        
        # Clean text fields
        df['title'] = df['title'].apply(self.clean_text)
        df['abstract'] = df['abstract'].apply(self.clean_text)
        
        # Create combined text field
        df['text'] = df.apply(lambda x: f"{x['title']} {x['abstract']}", axis=1)
        
        return df