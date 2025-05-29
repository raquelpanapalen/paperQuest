import os
import pickle
import hashlib


class CacheManager:
    """Centralized cache management for all models"""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, collection_df, model_name, suffix=""):
        """Generate a unique cache key based on collection, model, and preprocessing"""
        collection_size = len(collection_df)
        
        # Use first few cord_uids as a fingerprint
        max_sample = min(100, len(collection_df))
        ids_sample = "_".join(collection_df['cord_uid'].iloc[:max_sample])
        data_hash = hashlib.md5(ids_sample.encode()).hexdigest()[:8]
        
        # FIXED: Include columns in hash for different preprocessing
        column_signature = "_".join(sorted(collection_df.columns))
        column_hash = hashlib.md5(column_signature.encode()).hexdigest()[:4]
        
        # Build cache key with more specificity
        key_parts = [model_name, str(collection_size), data_hash, column_hash]
        if suffix:
            key_parts.append(suffix)
            
        return "_".join(key_parts)
    
    def get_cache_path(self, cache_key):
        """Get full path for cache file"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def load_cached(self, cache_key):
        """Load data from cache if exists"""
        cache_path = self.get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError, OSError) as e:
                # FIXED: Handle corrupted cache files
                print(f"Warning: Corrupted cache file {cache_path}, removing: {e}")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return None
        return None
    
    def save_to_cache(self, cache_key, data):
        """Save data to cache with error handling"""
        cache_path = self.get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, OSError) as e:
            print(f"Warning: Could not save to cache {cache_path}: {e}")
    
    def get_or_create(self, cache_key, create_fn, *args, **kwargs):
        """Get from cache or create with function"""
        cached_data = self.load_cached(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Create data using provided function
        data = create_fn(*args, **kwargs)
        self.save_to_cache(cache_key, data)
        return data
    
    def clear_cache(self, model_name=None):
        """Clear cache files, optionally for specific model"""
        if not os.path.exists(self.cache_dir):
            return
            
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                if model_name is None or filename.startswith(model_name):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except OSError:
                        pass