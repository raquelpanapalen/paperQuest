"""
Simplified Paper Quest - Clean API for retrieval evaluation
"""

import pandas as pd
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm

from preprocessing import TextPreprocessor
from model_registry import ModelRegistry
from models.base import RetrievalConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleEvaluator:
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def evaluate(self, config):
        """
        Evaluate retrieval models with a single config dictionary
        
        Args:
            config: Dictionary containing all configuration parameters
                Required keys:
                    - collection_path: Path to collection data
                    - query_path: Path to query data
                Optional keys:
                    - models: List of models to run (default: ["bm25"])
                    - output_dir: Directory for results (default: "results")
                    - top_k: Number of documents to retrieve (default: 5)
                    - sample_size: Number of queries to sample (default: None)
                    - collection_sample_size: Number of docs to sample (default: None)
                    - ... any other model-specific parameters
        
        Returns:
            Dictionary containing evaluation results
        """
        # Extract paths (required)
        collection_path = config['collection_path']
        query_path = config['query_path']
        
        # Extract optional parameters with defaults
        models = config.get('models', ['bm25'])
        output_dir = config.get('output_dir', 'results')
        top_k = config.get('top_k', 5)
        sample_size = config.get('sample_size', None)
        collection_sample_size = config.get('collection_sample_size', None)
        collection_columns = config.get('collection_columns', None)
        mrr_k = config.get('mrr_k', [1, 5, 10])
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create model configuration
        model_config = RetrievalConfig(**{k: v for k, v in config.items() 
                                         if k not in ['collection_path', 'query_path', 
                                                     'models', 'output_dir', 'collection_columns',
                                                     'mrr_k', 'sample_size']})
        
        # Load data
        collection_df, query_df = self._load_data(
            collection_path, query_path, collection_columns, 
            collection_sample_size, sample_size
        )
        
        # Check if this is a test set
        is_test_set = 'cord_uid' not in query_df.columns
        
        # Handle "all" models
        if models == ['all'] or models == 'all':
            models = ModelRegistry.get_all_models()
        
        # Run evaluations
        results = {}
        predictions = {}
        
        for model_name in models:
            try:
                logger.info(f"Running {model_name}...")
                
                # Get model class and instantiate
                model_class = ModelRegistry.get_model_class(model_name)
                
                # Special handling for query expansion model
                if model_name == 'langchain_query_expansion':
                    model = model_class(collection_df, queries_df=query_df, config=model_config)
                else:
                    model = model_class(collection_df, config=model_config)
                
                # Get predictions using simple loop instead of batch_retrieve
                preds = []
                for query_text in tqdm(query_df['tweet_text'].tolist(), desc=f"Processing {model_name}"):
                    try:
                        pred = model.retrieve(query_text, top_k=top_k)
                        preds.append(pred)
                    except Exception as e:
                        logger.warning(f"Error processing query: {e}")
                        preds.append([])
                
                predictions[model_name] = preds
                
                # Evaluate if we have labels
                if not is_test_set:
                    query_df[f'{model_name}_preds'] = preds
                    model_results = self._calculate_mrr(query_df, f'{model_name}_preds', mrr_k)
                    results[model_name] = model_results
                    logger.info(f"{model_name} MRR@5: {model_results.get(5, 'N/A'):.4f}")
                
                # Export predictions
                self._export_predictions(
                    query_df, 
                    model_name, 
                    preds, 
                    os.path.join(output_dir, f"{model_name}_predictions_{timestamp}.tsv")
                )
                
            except Exception as e:
                logger.error(f"Error with {model_name}: {str(e)}")
                continue
        
        # Return comprehensive results
        return {
            'metrics': results,
            'predictions': predictions,
            'is_test_set': is_test_set,
            'output_dir': output_dir,
            'timestamp': timestamp,
            'config': config
        }
    
    def _load_data(self, collection_path, query_path, collection_columns=None,
               collection_sample_size=None, sample_size=None):
        """Load data without preprocessing - let models handle it"""
        collection_df = pd.read_pickle(collection_path)
        
        # Only filter columns if specified, don't preprocess
        if collection_columns:
            keep_columns = [col for col in collection_columns if col in collection_df.columns]
            if 'cord_uid' not in keep_columns:
                keep_columns.append('cord_uid')
            collection_df = collection_df[keep_columns]
        
        if collection_sample_size and collection_sample_size < len(collection_df):
            collection_df = collection_df.sample(collection_sample_size, random_state=42)
            collection_df = collection_df.reset_index(drop=True)
        
        query_df = pd.read_csv(query_path, sep='\t')
        
        if sample_size and sample_size < len(query_df):
            query_df = query_df.sample(sample_size, random_state=42)
            query_df = query_df.reset_index(drop=True)
        
        return collection_df, query_df
    
    def _calculate_mrr(self, df, pred_column, mrr_k):
        """Calculate MRR scores"""
        results = {}
        for k in mrr_k:
            df["in_topx"] = df.apply(
                lambda x: (1/([i for i in x[pred_column][:k]].index(x['cord_uid']) + 1) 
                          if x['cord_uid'] in [i for i in x[pred_column][:k]] else 0), 
                axis=1
            )
            results[k] = df["in_topx"].mean().item()
        return results
    
    def _export_predictions(self, df, model_name, predictions, output_file):
        """Export predictions to file"""
        export_df = pd.DataFrame({
            'post_id': df['post_id'],
            'preds': predictions
        })
        export_df['preds'] = export_df['preds'].apply(lambda x: x[:5])
        export_df.to_csv(output_file, index=None, sep='\t')


# Simple function interface
def evaluate_models(config):
    """
    Simple function interface for evaluating models
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Results dictionary
    """
    evaluator = SimpleEvaluator()
    return evaluator.evaluate(config)