from models.LangChainRAG import LangChainRAGRetriever
from models.base_class import RetrievalConfig

from langchain_ollama.llms import OllamaLLM
import regex as re
import os
import pickle
from tqdm import tqdm


EXPANSION_MODEL = "gemma3:4b-it-qat"


class LangChainQueryExpansionRetriever(LangChainRAGRetriever):
    """LangChain retriever with batch query expansion capabilities that combines expansions into a single query"""
    
    def __init__(self, collection_df, queries_df=None, config=None):
        # Initialize config first if needed
        if config is None:
            config = RetrievalConfig()
        self.config = config
        
        # Initialize expansion model before parent constructors
        self._init_expansion_model()
        
        # Define expansion prompt
        self.expansion_prompt = """You are an expert at converting user twitter tweets into short scientific abstracts. Perform query expansion. If there are multiple common ways of phrasing a user question or common synonyms for key words in the question, make sure to return multiple versions of the query with the different phrasings. If there are acronyms or words you are not familiar with, do not try to rephrase them. Return at most 3 versions of the question. Only return the created Abstracts after Extended Query: [Abstract 1] [Abstract 2] [Abstract 3]. End the generation after the creation of the abstracts. Never generate more than this."""

        # Create expansion cache
        self.query_expansions = {}
        
        # If queries are provided, expand them all at once
        if queries_df is not None:
            self._batch_expand_queries(queries_df)
        
        # Continue with parent initialization
        super().__init__(collection_df, config)
    
    def _init_expansion_model(self):
        """Initialize the Ollama model for query expansion"""
        self.expansion_model = OllamaLLM(model=EXPANSION_MODEL)
    
    def _batch_expand_queries(self, queries_df):
        """Expand all queries at once and cache the results"""
        # Create a cache file path
        cache_file = os.path.join(self.config.cache_dir, "query_expansions_cache.pkl")
        
        # Check if cache exists
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.query_expansions = pickle.load(f)
        else:
            # Process all queries
            unique_queries = queries_df['tweet_text'].unique()
            
            for query in tqdm(unique_queries, desc="Expanding queries"):
                if query not in self.query_expansions:
                    self.query_expansions[query] = self._expand_single_query(query)
                
                # Save cache periodically
                if len(self.query_expansions) % 100 == 0:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(self.query_expansions, f)
            
            # Save final cache
            with open(cache_file, 'wb') as f:
                pickle.dump(self.query_expansions, f)
                
        return self.query_expansions
    
    def _expand_single_query(self, query_text):
        """Expand a single query into multiple variations"""
        # Create a prompt for query expansion
        prompt = f"{self.expansion_prompt}\n\nOriginal query: {query_text}\n\nExpanded queries:"
        
        try:
            # Get model response
            response = self.expansion_model.invoke(prompt)
            expanded_text = response
            
            # Parse the expanded queries
            expanded_queries = []
            for line in expanded_text.split('\n'):
                # Remove numbering if present (e.g., "1. ", "- ", etc.)
                clean_line = re.sub(r'^[\d\.\-\*]+\s*', '', line).strip()
                if clean_line and clean_line != query_text and len(clean_line) > 5:
                    expanded_queries.append(clean_line)
            
            # Always include the original query
            if query_text not in expanded_queries:
                expanded_queries.insert(0, query_text)  # Put original query first
            
            return expanded_queries
            
        except Exception as e:
            # Return just the original query in case of error
            return [query_text]
    
    def _create_combined_query(self, query_text, expanded_queries):
        """Create a combined query from original and expanded queries"""
        # Format: Original Query followed by all expanded queries
        combined_query = query_text + "\n"
        
        # Add expanded queries, skipping the original if it's in there
        for query in expanded_queries:
            if query != query_text:  # Avoid duplication
                combined_query += query + "\n"
        
        return combined_query.strip()
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents using a combined query of original and expansions"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Get expanded queries from cache, or expand on the fly if not found
        if query_text in self.query_expansions:
            expanded_queries = self.query_expansions[query_text]
        else:
            expanded_queries = self._expand_single_query(query_text)
            self.query_expansions[query_text] = expanded_queries
        
        # Create a combined query with format:
        # {Original Query}
        # {Expanded Query 1}
        # {Expanded Query 2}
        # {Expanded Query 3}
        combined_query = self._create_combined_query(query_text, expanded_queries)
        
        # Retrieve documents using the combined query in a single operation
        docs = self.base_retriever.get_relevant_documents(combined_query)
        
        # Extract cord_uids
        unique_results = []
        seen = set()
        
        for doc in docs:
            cord_uid = doc.metadata["cord_uid"]
            if cord_uid not in seen:
                unique_results.append(cord_uid)
                seen.add(cord_uid)
                
                # Stop once we have enough results
                if len(unique_results) >= top_k:
                    break
        
        # Return top-k unique results (should already be limited, but just to be safe)
        return unique_results[:top_k]