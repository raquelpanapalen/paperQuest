import os
import pickle
import hashlib
from typing import Any, Optional, Callable


class CacheManager:
    """Centralized cache management for all models"""
    
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, collection_df, model_name: str, suffix: str = "") -> str:
        """Generate a unique cache key based on collection and model"""
        collection_size = len(collection_df)
        
        # Use first few cord_uids as a fingerprint
        max_sample = min(100, len(collection_df))
        ids_sample = "_".join(collection_df['cord_uid'].iloc[:max_sample])
        data_hash = hashlib.md5(ids_sample.encode()).hexdigest()[:8]
        
        # Build cache key
        key_parts = [model_name, str(collection_size), data_hash]
        if suffix:
            key_parts.append(suffix)
            
        return "_".join(key_parts)
    
    def get_cache_path(self, cache_key: str) -> str:
        """Get full path for cache file"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def load_cached(self, cache_key: str) -> Optional[Any]:
        """Load data from cache if exists"""
        cache_path = self.get_cache_path(cache_key)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache"""
        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def get_or_create(self, cache_key: str, create_fn: Callable, *args, **kwargs) -> Any:
        """Get from cache or create with function"""
        cached_data = self.load_cached(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Create data using provided function
        data = create_fn(*args, **kwargs)
        self.save_to_cache(cache_key, data)
        return data