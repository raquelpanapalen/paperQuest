import pandas as pd
import os

import json
import logging
from datetime import datetime

from utils import preprocess_collection, clean_text
from utils import get_performance_mrr, export_predictions

from models.base_class import RetrievalConfig, BaseRetriever
from models.BM25Retriever import BM25Retriever
from models.EnhancedBM25Retriever import EnhancedBM25Retriever
from models.DenseRetriever import DenseRetriever
from models.NeuralReranker import NeuralReranker
from models.LangChainRAG import LangChainRAGRetriever 
from models.LangChainRagReranker import LangChainRerankerRetriever
from models.LangChainQueryExpansion import LangChainQueryExpansionRetriever
from models.LangChainFullExpansion import LangChainFullExpansionRetriever



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('retrieval_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

#################################
# Model Implementations
#################################

MODEL_REGISTRY = {}

MODEL_REGISTRY['bm25'] = BM25Retriever
MODEL_REGISTRY['enhanced_bm25'] = EnhancedBM25Retriever
MODEL_REGISTRY['dense'] = DenseRetriever
MODEL_REGISTRY['neural_rerank'] = NeuralReranker
MODEL_REGISTRY['langchain_rag'] = LangChainRAGRetriever
MODEL_REGISTRY['langchain_reranker'] = LangChainRerankerRetriever
MODEL_REGISTRY['langchain_query_expansion'] = LangChainQueryExpansionRetriever
MODEL_REGISTRY['langchain_full_expansion'] = LangChainFullExpansionRetriever


#################################
# Evaluation Framework
#################################

def evaluate_models(collection_path, query_path, models_to_run=None, 
                   output_dir='output', collection_columns=None,
                   top_k=5, sample_size=None, collection_sample_size=None,
                   mrr_k=[1, 5, 10], **kwargs):
    """Evaluate selected retrieval models"""
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create config with provided parameters
    config_params = kwargs.copy()
    config_params['top_k'] = top_k
    config_params['cache_dir'] = config_params.get('cache_dir', 'cache')
    config_params['collection_sample_size'] = collection_sample_size
    config = RetrievalConfig(**config_params)
    
    logger.info(f"Loading collection data from: {collection_path}")
    collection_df = pd.read_pickle(collection_path)
    collection_df = preprocess_collection(collection_df, collection_columns)
    
    logger.info(f"Loading query data from: {query_path}")
    query_df = pd.read_csv(query_path, sep='\t')
    #query_df = preprocess_queries(query_df)
    
    # Sample collection if requested (before pre-processing to save time)
    if collection_sample_size is not None and collection_sample_size < len(collection_df):
        logger.info(f"Sampling {collection_sample_size} papers from collection (from {len(collection_df)} total)")
        collection_df = collection_df.sample(collection_sample_size, random_state=config.collection_sample_seed)
        # Make sure to reset index after sampling
        collection_df = collection_df.reset_index(drop=True)
        
        # Create a special tag for this sampled collection to avoid confusion with full dataset
        sample_tag = f"_sample{collection_sample_size}"
    else:
        sample_tag = ""
    
    logger.info(f"Loading query data from: {query_path}")
    query_df = pd.read_csv(query_path, sep='\t')
    
    # Sample queries if requested
    if sample_size is not None and sample_size < len(query_df):
        logger.info(f"Sampling {sample_size} queries")
        query_df = query_df.sample(sample_size, random_state=42)
        query_df = query_df.reset_index(drop=True)
    
    logger.info(f"Collection size: {len(collection_df)}")
    logger.info(f"Query set size: {len(query_df)}")
    
    # Define models to run
    all_available_models = list(MODEL_REGISTRY.keys())
    if models_to_run is None:
        models_to_run = all_available_models
    
    # Validate models
    invalid_models = [m for m in models_to_run if m not in all_available_models]
    if invalid_models:
        raise ValueError(f"Unknown models: {invalid_models}. Available models: {all_available_models}")
    
    # Run models and evaluate
    results = {}
    
    results = {}
    
    for model_name in models_to_run:
        logger.info(f"\n=== Running {model_name} ===")
        
        # Initialize model, passing query_df for query expansion models
        model_class = MODEL_REGISTRY[model_name]
        if model_name in ['langchain_query_expansion', 'langchain_full_expansion']:
            model = model_class(collection_df, queries_df=query_df, config=config)
        else:
            model = model_class(collection_df, config)
        
        # Retrieve documents
        logger.info(f"Retrieving documents for {len(query_df)} queries...")
        pred_column = f"{model_name}_preds"
        
        # Process in batches for better memory management
        batch_size = config.batch_size
        all_predictions = []
        
        # Split queries into smaller batches
        for i in range(0, len(query_df), batch_size):
            batch_queries = query_df['tweet_text'].iloc[i:i+batch_size].tolist()
            batch_predictions = model.batch_retrieve(batch_queries, top_k=top_k)
            all_predictions.extend(batch_predictions)
        
        # Add predictions to dataframe
        query_df[pred_column] = all_predictions
        
        # Evaluate
        model_results = get_performance_mrr(query_df, 'cord_uid', pred_column, list_k=mrr_k)
        results[model_name] = model_results
        
        logger.info(f"{model_name} Results: {model_results}")
        
        # Export predictions
        output_file = os.path.join(output_dir, f"{model_name}_predictions_{timestamp}.tsv")
        export_predictions(query_df, pred_column, output_file)
    

    # Modify the output filename to indicate sampled collection
    results_file = os.path.join(output_dir, f"evaluation_results{sample_tag}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    logger.info("\n=== Final Evaluation Summary ===")
    for model_name, model_results in results.items():
        logger.info(f"{model_name} MRR@5: {model_results.get(5, 'N/A')}")
    
    # Determine best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1].get(5, 0))[0]
        logger.info(f"\nBest model: {best_model}")
    
    return results

#################################
# Command Line Interface
#################################

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate retrieval models for scientific claim source retrieval')
    
    # Data paths
    parser.add_argument('--collection', required=True, help='Path to collection data')
    parser.add_argument('--queries', required=True, help='Path to query data')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    parser.add_argument('--cache_dir', default='cache', help='Cache directory')
    
    # Sampling settings
    parser.add_argument('--collection_sample_size', type=int, default=None, 
                       help='Number of papers to sample from collection (default: use all papers)')
    parser.add_argument('--sample_size', type=int, default=None, 
                       help='Number of queries to sample (default: use all queries)')
    
    # Model selection
    parser.add_argument('--models', nargs='+', default=['bm25'], 
                      help=f'Models to evaluate: {", ".join(MODEL_REGISTRY.keys())} or "all"')
    
    # Evaluation parameters
    parser.add_argument('--top_k', type=int, default=5, help='Number of documents to retrieve')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of queries to sample')
    parser.add_argument('--mrr_k', nargs='+', type=int, default=[1, 5, 10], help='K values for MRR evaluation')
    
    # Collection preprocessing
    parser.add_argument('--columns', nargs='+', default=None, help='Columns to use from collection')
    
    # Model-specific parameters
    parser.add_argument('--embedding_model', default='sentence-transformers/allenai-specter', 
                       help='Embedding model for dense retrieval')
    parser.add_argument('--reranker_model', default='cross-encoder/ms-marco-MiniLM-L-6-v2', 
                       help='Reranker model for neural reranking')
    parser.add_argument('--langchain_embedding', default='nomic-embed-text', 
                       help='Embedding model for LangChain retrievers')
    parser.add_argument('--candidate_count', type=int, default=100, 
                       help='Number of candidates for first-stage retrieval')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for processing')
    parser.add_argument('--reranker_batch_size', type=int, default=8, 
                       help='Batch size for reranker')
    parser.add_argument('--no_gpu', action='store_true', 
                       help='Disable GPU acceleration')
    parser.add_argument('--sample_for_expansion', type=int, default=100,
                      help='Number of documents to sample for expansion in full expansion retriever')
    
    args = parser.parse_args()
    
    # Process 'all' model selection
    if 'all' in args.models:
        args.models = list(MODEL_REGISTRY.keys())
        
    return args

#################################
# Main Function
#################################

def main():
    """Main function for command line usage"""
    args = parse_args()
    
    # Convert args to config dict
    config_params = {
        'cache_dir': args.cache_dir,
        'embedding_model': args.embedding_model,
        'reranker_model': args.reranker_model,
        'langchain_embedding': args.langchain_embedding,
        'candidate_count': args.candidate_count,
        'batch_size': args.batch_size,
        'reranker_batch_size': args.reranker_batch_size,
        'use_gpu': not args.no_gpu,
        'sample_for_expansion': args.sample_for_expansion
    }
    
    # Run evaluation
    results = evaluate_models(
        collection_path=args.collection,
        query_path=args.queries,
        models_to_run=args.models,
        output_dir=args.output_dir,
        collection_columns=args.columns,
        top_k=args.top_k,
        sample_size=args.sample_size,
        mrr_k=args.mrr_k,
        **config_params
    )
    
    return results

if __name__ == "__main__":
    main()