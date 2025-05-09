import regex as re
import pandas as pd


#################################
# Data Preprocessing
#################################

def clean_text(text):
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
    text = text.strip()
    
    return text


def preprocess_collection(collection_df, use_columns=None):
    """Preprocess the collection dataframe to standardize format and clean data"""
    
    # Make a copy to avoid modifying the original
    df = collection_df.copy()
    
    # Define essential columns
    essential_columns = ['cord_uid', 'title', 'abstract']
    
    # Define columns to keep
    if use_columns is None:
        # Default set of useful columns
        keep_columns = essential_columns + ['authors', 'journal', 'publish_time', 'doi']
    else:
        # Ensure essential columns are included
        keep_columns = list(set(essential_columns + use_columns))
    
    # Keep only the selected columns that exist in the dataframe
    keep_columns = [col for col in keep_columns if col in df.columns]
    df = df[keep_columns]
    
    # Fill missing values
    df['title'] = df['title'].fillna("")
    df['abstract'] = df['abstract'].fillna("")
    
    # Clean text fields
    df['title'] = df['title'].apply(clean_text)
    df['abstract'] = df['abstract'].apply(clean_text)
    
    # Create a new combined text field
    df['text'] = df.apply(lambda x: f"{x['title']} {x['abstract']}", axis=1)
    
    #df = df.drop_duplicates(subset=['cord_uid'])
    
    return df



#################################
# Evaluation Functions
#################################

def get_performance_mrr(data, col_gold, col_pred, list_k=[1, 5, 10]):
    """Evaluate retrieved candidates using MRR@k"""
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) 
                                               if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
        d_performance[k] = data["in_topx"].mean().item()
    return d_performance


def export_predictions(df, pred_column, output_file):
    """Export predictions to the required format for submission"""
    # Create a copy to avoid modifying the original
    export_df = df[['post_id', pred_column]].copy()
    
    # Rename prediction column
    export_df = export_df.rename(columns={pred_column: 'preds'})
    
    # Ensure predictions are limited to top 5
    export_df['preds'] = export_df['preds'].apply(lambda x: x[:5])
    
    # Save to file
    export_df.to_csv(output_file, index=None, sep='\t')
