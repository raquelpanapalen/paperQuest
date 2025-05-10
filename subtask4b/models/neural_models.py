import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.base import BaseRetriever, CachedModelMixin, EmbeddingMixin
from models.bm25 import BM25Retriever


class DenseRetriever(BaseRetriever, CachedModelMixin, EmbeddingMixin):
    """Dense retriever using transformer embeddings"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "dense_retriever"
        
        # Initialize model
        self.model = self._init_model()
        
        # Get document embeddings
        self.doc_embeddings = self._load_or_create_model(
            self._create_document_embeddings, 
            suffix=f"embeddings_{self.config.embedding_model.replace('/', '_')}"
        )
    
    def _init_model(self):
        """Initialize the embedding model"""
        model = SentenceTransformer(self.config.embedding_model)
        
        if self.config.use_gpu and torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
        
        return model
    
    def _create_document_embeddings(self):
        """Create document embeddings"""
        docs = self.collection_df['text'].tolist()
        
        embeddings = self._create_embeddings(
            docs, 
            lambda batch: self.model.encode(batch, show_progress_bar=False),
            self.config.batch_size
        )
        
        return np.array(embeddings)
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve top-k documents using dense embeddings"""
        if top_k is None:
            top_k = self.config.top_k
                    
        # Encode query
        query_embedding = self.model.encode(query_text)
        
        # Calculate similarity scores
        cos_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        
        # Get top-k document indices
        top_indices = torch.topk(cos_scores, k=top_k).indices.cpu().numpy()
        
        return [self.cord_uids[idx] for idx in top_indices]


class NeuralReranker(BaseRetriever):
    """Neural reranker using a two-stage retrieval approach"""
    
    def __init__(self, collection_df, config=None, first_stage_model=None):
        super().__init__(collection_df, config)
        self.model_name = "neural_reranker"
        
        # Initialize first-stage retriever
        if first_stage_model is None:
            self.first_stage = BM25Retriever(collection_df, config)
        else:
            self.first_stage = first_stage_model
        
        # Initialize reranker model
        self._init_reranker()
    
    def _init_reranker(self):
        """Initialize the reranker model"""
        model_name = self.config.reranker_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Use GPU if available and configured
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        self.device = device
        self.model.to(self.device)
    
    def _batch_inference(self, pairs, batch_size=None):
        """Run batched inference on pairs of query-document texts"""
        if batch_size is None:
            batch_size = self.config.reranker_batch_size
            
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                # Tokenize
                features = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                # Get scores
                outputs = self.model(**features)
                scores = outputs.logits.squeeze(-1).cpu().numpy()
                all_scores.extend(scores)
                
        return all_scores
    
    def retrieve(self, query_text, top_k=None):
        """Two-stage retrieval: BM25 followed by neural reranking"""
        if top_k is None:
            top_k = self.config.top_k
            
        # Get first-stage candidates
        candidates = self.first_stage.retrieve(query_text, top_k=self.config.candidate_count)
        candidate_indices = [self.cord_uids.index(uid) for uid in candidates]
        
        # Create texts for reranking
        candidate_texts = []
        for idx in candidate_indices:
            doc = self.collection_df.iloc[idx]
            text = doc['text'][:512]  # Truncate to fit in context
            candidate_texts.append(text)
        
        # Create pairs of (query, document) for the reranker
        pairs = [[query_text, doc] for doc in candidate_texts]
        
        # Use batched inference
        all_scores = self._batch_inference(pairs)
        
        # Get indices of top-k scores
        reranked_indices = np.argsort(-np.array(all_scores))[:top_k]
        
        # Map back to original document indices
        final_indices = [candidate_indices[idx] for idx in reranked_indices]
        
        return [self.cord_uids[idx] for idx in final_indices]