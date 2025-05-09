from models.LangChainQueryExpansion import LangChainQueryExpansionRetriever
from models.base_class import RetrievalConfig
import pickle
import os
from tqdm import tqdm
import numpy as np

class LangChainFullExpansionRetriever(LangChainQueryExpansionRetriever):
    """LangChain retriever with both query and document expansion capabilities"""
    
    def __init__(self, collection_df, config=None):
        # Initialize the config first
        if config is None:
            config = RetrievalConfig()
        self.config = config
        
        # Initialize expansion model
        self._init_expansion_model()
        
        # Now expand documents (with sampling for efficiency)
        self.collection_df = collection_df
        self.expanded_collection_df = self._expand_collection_documents(collection_df)
        
        # Initialize parent with expanded documents
        super(LangChainQueryExpansionRetriever, self).__init__(self.expanded_collection_df, config)
    
    def _expand_collection_documents(self, df):
        """Expand abstracts in the collection with synonyms and alternative phrasings"""
        expanded_df = df.copy()
        
        
        # Document expansion prompt
        doc_expansion_prompt = """Your task is to enhance this scientific abstract by adding synonyms and alternative phrasings for key technical terms. 
        Do not change the meaning or add new information. Format the result as a single paragraph.
        
        Original abstract:
        {abstract}
        
        Enhanced abstract:"""
        
        # Process in batches and cache results to avoid repeating work
        cache_file = os.path.join(self.config.cache_dir, "expanded_abstracts_cache.pkl")
        
        # This caching is fine because we're only caching the text expansions, not the vector store
        if os.path.exists(cache_file):

            with open(cache_file, 'rb') as f:
                expanded_abstracts = pickle.load(f)
        else:
            expanded_abstracts = {}
            
            # Sample a subset of documents to expand for efficiency
            sample_size = min(self.config.sample_for_expansion, len(df))
            sampled_indices = np.random.choice(len(df), sample_size, replace=False)
            
            # Process sampled abstracts
            for idx in tqdm(sampled_indices, desc="Expanding abstracts"):
                abstract = df.iloc[idx]['abstract']
                
                # Skip if already in cache
                if abstract in expanded_abstracts:
                    continue
                    
                # Skip very long abstracts or empty ones
                if len(abstract) > 2000 or len(abstract) < 10:
                    expanded_abstracts[abstract] = abstract
                    continue
                
                # Create prompt for this abstract
                prompt = doc_expansion_prompt.format(abstract=abstract)
                
                try:
                    # Get LM response
                    response = self.expansion_model.invoke(prompt)
                    
                    # Extract the expanded abstract
                    expanded_abstract = response.strip()
                    
                    # Store in cache
                    expanded_abstracts[abstract] = expanded_abstract
                    
                    # Save intermediate results periodically
                    if len(expanded_abstracts) % 20 == 0:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(expanded_abstracts, f)
                            
                except Exception as e:

                    expanded_abstracts[abstract] = abstract
            
            # Save final results
            with open(cache_file, 'wb') as f:
                pickle.dump(expanded_abstracts, f)
        
        # Apply expansions to dataframe
        expanded_df['expanded_abstract'] = expanded_df['abstract'].map(
            lambda abstract: expanded_abstracts.get(abstract, abstract)
        )
        
        # Update text field with expanded content
        expanded_df['text'] = expanded_df.apply(
            lambda x: f"{x['title']} {x.get('expanded_abstract', x['abstract'])}", axis=1
        )
        
        return expanded_df