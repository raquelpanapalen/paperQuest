import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextPreprocessor:
    """Text preprocessing with clear processing levels"""
    
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
    
    def minimal_preprocessing(self, text):
        """Preserve text with only whitespace normalization - for neural models"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Only normalize excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing spaces
        return text.strip()
    
    def heavy_preprocessing(self, text):
        """Full cleaning: lowercase, remove URLs, stopwords, etc. - for keyword matching"""
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
        
        # Remove stopwords but keep longer words
        tokens = [word for word in tokens if word not in self.stop_words or len(word) > 2]
        
        return ' '.join(tokens)
    
    def social_media_preprocessing(self, text):
        """Handle Twitter/social media elements: mentions, hashtags, URLs"""
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
    
    def prepare_collection_minimal(self, collection_df, use_columns=None):
        """Prepare collection with minimal processing"""
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
        
        # Apply minimal preprocessing
        df['text'] = df.apply(lambda x: f"{x['title']} {x['abstract']}", axis=1)
        df['text'] = df['text'].apply(self.minimal_preprocessing)
        
        return df
    
    def prepare_collection_cleaned(self, collection_df, use_columns=None):
        """Prepare collection with heavy cleaning"""
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
        
        # Apply preprocessing to individual fields
        df['title'] = df['title'].apply(self.heavy_preprocessing)
        df['abstract'] = df['abstract'].apply(self.heavy_preprocessing)
        df['text'] = df.apply(lambda x: f"{x['title']} {x['abstract']}", axis=1)
        
        return df
    
    def prepare_collection_simple(self, collection_df, use_columns=None):
        """
        Prepare collection with simple text combination
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
        
        # Use simple text combination
        df['text'] = df.apply(
            lambda x: prepare_text_simple(x['title'], x['abstract']), 
            axis=1
        )
        
        return df