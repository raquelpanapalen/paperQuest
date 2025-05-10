from typing import List, Optional, Dict, Tuple
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder, util, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
from tqdm import tqdm

from models.base import BaseRetriever, CachedModelMixin, EmbeddingMixin
from models.bm25 import BM25Retriever, EnhancedBM25Retriever
from models.neural_models import DenseRetriever

logger = logging.getLogger(__name__)


class DistilledNeuralReranker(BaseRetriever, CachedModelMixin):
    """Knowledge Distillation Neural Reranker for efficient scientific paper retrieval"""
    
    def __init__(self, collection_df, config=None, queries_df=None):
        super().__init__(collection_df, config)
        self.model_name = "distilled_neural_reranker"
        
        # Configure teacher and student models
        self.teacher_model_name = getattr(config, 'teacher_model', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.student_model_name = getattr(config, 'student_model', 'cross-encoder/ms-marco-TinyBERT-L-2-v2')
        
        # Initialize first-stage retriever
        self.first_stage = BM25Retriever(collection_df, config)
        
        # Initialize or load distilled model
        self.student = self._load_or_create_model(
            lambda: self._create_distilled_model(queries_df),
            suffix=f"distilled_{self.student_model_name.replace('/', '_')}"
        )
    
    def _create_distilled_model(self, queries_df):
        """Create distilled model from teacher"""
        # Check if we already have a trained distilled model
        distilled_path = self._get_distilled_model_path()
        if os.path.exists(distilled_path):
            logger.info(f"Loading existing distilled model from {distilled_path}")
            return CrossEncoder(distilled_path)
        
        # Initialize teacher
        logger.info(f"Initializing teacher model: {self.teacher_model_name}")
        teacher = CrossEncoder(self.teacher_model_name)
        
        # Initialize student
        logger.info(f"Initializing student model: {self.student_model_name}")
        student = CrossEncoder(self.student_model_name)
        
        # Distill knowledge if we have training data
        if queries_df is not None:
            logger.info("Starting knowledge distillation...")
            self._distill_knowledge(teacher, student, queries_df)
            
            # Save the distilled model
            student.save(distilled_path)
            logger.info(f"Saved distilled model to {distilled_path}")
        
        return student
    
    def _get_distilled_model_path(self):
        """Get path for saving distilled model"""
        model_dir = os.path.join(self.config.cache_dir, 'distilled_models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_name = f"distilled_{self.student_model_name.replace('/', '_')}"
        return os.path.join(model_dir, model_name)
    
    def _distill_knowledge(self, teacher, student, queries_df):
        """Distill knowledge from teacher to student using query-paper pairs"""
        from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
        
        # Generate training examples
        train_examples = self._generate_distillation_examples(teacher, queries_df)
        
        if not train_examples:
            logger.warning("No training examples generated for distillation")
            return
        
        # Prepare training data
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Train student to match teacher scores
        logger.info(f"Training student model on {len(train_examples)} examples...")
        student.fit(
            train_dataloader=train_dataloader,
            epochs=1,
            warmup_steps=min(100, len(train_dataloader) // 10),
            use_amp=True
        )
    
    def _generate_distillation_examples(self, teacher, queries_df, num_negatives=5):
        """Generate training examples with teacher scores"""
        examples = []
        
        for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Generating distillation data"):
            query = row['tweet_text']
            positive_uid = row['cord_uid']
            
            # Get positive paper
            positive_idx = self.cord_uids.index(positive_uid)
            positive_paper = self.collection_df.iloc[positive_idx]['text'][:512]
            
            # Get candidates from first stage
            candidates = self.first_stage.retrieve(query, top_k=50)
            
            # Prepare pairs for teacher scoring
            pairs = []
            doc_texts = []
            
            # Add positive example
            pairs.append([query, positive_paper])
            doc_texts.append(positive_paper)
            
            # Add negative examples
            for candidate_uid in candidates:
                if candidate_uid != positive_uid:
                    candidate_idx = self.cord_uids.index(candidate_uid)
                    candidate_text = self.collection_df.iloc[candidate_idx]['text'][:512]
                    pairs.append([query, candidate_text])
                    doc_texts.append(candidate_text)
                    
                    if len(pairs) >= num_negatives + 1:
                        break
            
            # Get teacher scores
            with torch.no_grad():
                teacher_scores = teacher.predict(pairs)
            
            # Create training examples with teacher scores as labels
            for i, (pair, score) in enumerate(zip(pairs, teacher_scores)):
                examples.append(InputExample(texts=pair, label=float(score)))
        
        return examples
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve using efficient distilled model"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Get first-stage candidates
        candidates = self.first_stage.retrieve(query_text, top_k=self.config.candidate_count)
        
        # Prepare texts for reranking
        pairs = []
        candidate_indices = []
        
        for uid in candidates:
            try:
                idx = self.cord_uids.index(uid)
                doc_text = self.collection_df.iloc[idx]['text'][:512]
                pairs.append([query_text, doc_text])
                candidate_indices.append(idx)
            except ValueError:
                continue
        
        if not pairs:
            return []
        
        # Rerank with student model (efficient)
        scores = self.student.predict(pairs)
        
        # Sort by scores
        sorted_indices = np.argsort(-scores)[:top_k]
        
        # Return top-k cord_uids
        return [self.cord_uids[candidate_indices[idx]] for idx in sorted_indices]


class HybridNeuralReranker(BaseRetriever, CachedModelMixin):
    """Hybrid Dense-Sparse Reranker combining BM25 and dense retrieval"""
    
    def __init__(self, collection_df, config=None):
        super().__init__(collection_df, config)
        self.model_name = "hybrid_neural_reranker"
        
        # Initialize sparse and dense retrievers
        self.sparse_retriever = EnhancedBM25Retriever(collection_df, config)
        self.dense_retriever = DenseRetriever(collection_df, config)
        
        # Initialize reranker
        self.reranker = CrossEncoder(config.reranker_model)
        
        # Fusion parameters
        self.rrf_k = getattr(config, 'rrf_k', 60)
        self.sparse_weight = getattr(config, 'sparse_weight', 0.5)
        
        # Cache fusion results
        self._fusion_cache = {}
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve using hybrid sparse-dense approach with neural reranking"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Check cache
        cache_key = f"{query_text}_{self.config.candidate_count}"
        if cache_key in self._fusion_cache:
            fused_candidates = self._fusion_cache[cache_key]
        else:
            # Get candidates from both retrievers
            sparse_candidates = self.sparse_retriever.retrieve(
                query_text, top_k=self.config.candidate_count
            )
            dense_candidates = self.dense_retriever.retrieve(
                query_text, top_k=self.config.candidate_count
            )
            
            # Combine using reciprocal rank fusion
            fused_candidates = self._reciprocal_rank_fusion(
                sparse_candidates, dense_candidates
            )
            
            # Cache the fusion result
            self._fusion_cache[cache_key] = fused_candidates
        
        # Prepare for neural reranking
        rerank_candidates = fused_candidates[:self.config.candidate_count]
        
        if not rerank_candidates:
            return []
        
        # Neural reranking
        return self._neural_rerank(query_text, rerank_candidates, top_k)
    
    def _reciprocal_rank_fusion(self, sparse_results, dense_results):
        """Combine results using Reciprocal Rank Fusion (RRF)"""
        scores = {}
        
        # Score sparse results
        for rank, doc_id in enumerate(sparse_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + self.sparse_weight * rrf_score
        
        # Score dense results
        for rank, doc_id in enumerate(dense_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + (1 - self.sparse_weight) * rrf_score
        
        # Sort by combined score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs]
    
    def _neural_rerank(self, query_text, candidates, top_k):
        """Rerank candidates using neural model"""
        # Prepare query-document pairs
        pairs = []
        valid_indices = []
        
        for i, uid in enumerate(candidates):
            try:
                idx = self.cord_uids.index(uid)
                doc_text = self.collection_df.iloc[idx]['text'][:512]
                pairs.append([query_text, doc_text])
                valid_indices.append(i)
            except ValueError:
                continue
        
        if not pairs:
            return []
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Sort by scores
        sorted_indices = np.argsort(-scores)[:top_k]
        
        # Map back to original candidates
        return [candidates[valid_indices[idx]] for idx in sorted_indices]


class ContrastiveReranker(BaseRetriever, CachedModelMixin, EmbeddingMixin):
    """Contrastive Learning Reranker for improved semantic matching"""
    
    def __init__(self, collection_df, queries_df=None, config=None):
        super().__init__(collection_df, config)
        self.model_name = "contrastive_reranker"
        
        # Initialize first-stage retriever
        self.first_stage = BM25Retriever(collection_df, config)
        
        # Model configuration
        self.base_model_name = getattr(config, 'contrastive_base_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize or load contrastive model
        self.model = self._load_or_create_model(
            lambda: self._create_contrastive_model(queries_df),
            suffix=f"contrastive_{self.base_model_name.replace('/', '_')}"
        )
        
        # Create document embeddings for efficient retrieval
        self.doc_embeddings = self._load_or_create_model(
            self._create_document_embeddings,
            suffix=f"contrastive_embeddings_{self.base_model_name.replace('/', '_')}"
        )
    
    def _create_contrastive_model(self, queries_df):
        """Create or load contrastive model"""
        # Check for existing fine-tuned model
        model_path = self._get_contrastive_model_path()
        if os.path.exists(model_path):
            logger.info(f"Loading existing contrastive model from {model_path}")
            return SentenceTransformer(model_path)
        
        # Initialize base model
        logger.info(f"Initializing base model: {self.base_model_name}")
        model = SentenceTransformer(self.base_model_name)
        
        # Fine-tune with contrastive learning if we have training data
        if queries_df is not None:
            logger.info("Starting contrastive learning...")
            self._train_contrastive(model, queries_df)
            
            # Save the trained model
            model.save(model_path)
            logger.info(f"Saved contrastive model to {model_path}")
        
        return model
    
    def _get_contrastive_model_path(self):
        """Get path for saving contrastive model"""
        model_dir = os.path.join(self.config.cache_dir, 'contrastive_models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_name = f"contrastive_{self.base_model_name.replace('/', '_')}"
        return os.path.join(model_dir, model_name)
    
    def _train_contrastive(self, model, queries_df):
        """Train model using contrastive learning"""
        # Generate training examples
        train_examples = self._create_contrastive_examples(queries_df)
        
        if not train_examples:
            logger.warning("No training examples generated for contrastive learning")
            return
        
        # Create dataloader with no duplicates
        train_dataset = NoDuplicatesDataLoader(train_examples, batch_size=16)
        
        # Use Multiple Negatives Ranking Loss
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # Train the model
        logger.info(f"Training contrastive model on {len(train_examples)} examples...")
        model.fit(
            train_objectives=[(train_dataset, train_loss)],
            epochs=1,
            warmup_steps=100,
            show_progress_bar=True
        )
    
    def _create_contrastive_examples(self, queries_df):
        """Create positive and hard negative examples for contrastive learning"""
        examples = []
        
        for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Creating contrastive examples"):
            query = row['tweet_text']
            positive_uid = row['cord_uid']
            
            try:
                # Get positive paper
                positive_idx = self.cord_uids.index(positive_uid)
                positive_paper = self.collection_df.iloc[positive_idx]['text']
                
                # Create positive example
                examples.append(InputExample(texts=[query, positive_paper]))
                
                # Get hard negatives from BM25
                candidates = self.first_stage.retrieve(query, top_k=50)
                
                # Add hard negative examples
                num_negatives = 0
                for candidate_uid in candidates:
                    if candidate_uid != positive_uid and num_negatives < 5:
                        candidate_idx = self.cord_uids.index(candidate_uid)
                        negative_paper = self.collection_df.iloc[candidate_idx]['text']
                        examples.append(InputExample(texts=[query, negative_paper]))
                        num_negatives += 1
                
            except ValueError:
                continue
        
        return examples
    
    def _create_document_embeddings(self):
        """Create embeddings for all documents"""
        docs = self.collection_df['text'].tolist()
        
        embeddings = self._create_embeddings(
            docs,
            lambda batch: self.model.encode(batch, show_progress_bar=False, convert_to_tensor=True),
            self.config.batch_size
        )
        
        return torch.stack(embeddings) if isinstance(embeddings[0], torch.Tensor) else np.array(embeddings)
    
    def retrieve(self, query_text, top_k=None):
        """Retrieve using contrastive embeddings"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Encode query
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        
        # Calculate similarity scores
        if isinstance(self.doc_embeddings, torch.Tensor):
            cos_scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        else:
            # Convert numpy to tensor for similarity calculation
            doc_tensor = torch.tensor(self.doc_embeddings)
            cos_scores = util.cos_sim(query_embedding, doc_tensor)[0]
        
        # Get top-k indices
        top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))
        top_indices = top_results.indices.cpu().numpy()
        
        return [self.cord_uids[idx] for idx in top_indices]
