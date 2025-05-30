# Scientific Claim Analysis and Retrieval System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A comprehensive system for analyzing scientific claims in social media posts and retrieving corresponding academic publications. This project implements solutions for **CLEF 2025 CheckThat Lab Task 4**, addressing two critical challenges in scientific misinformation detection and academic source verification.

## 🎯 Project Overview

This repository contains implementations for two interconnected subtasks:

- **Subtask 4A**: Scientific Claim Classification in Tweets
- **Subtask 4B**: Scientific Claim Source Retrieval

### Subtask 4A: Tweet Classification
Automatically classify tweets into three binary categories:
1. **Scientific Claim**: Does the tweet contain a scientific assertion?
2. **Scientific Reference**: Does it reference a scientific study or publication?
3. **Scientific Entity**: Does it mention universities, scientists, or research organizations?

### Subtask 4B: Source Retrieval
Given a tweet that implicitly references a scientific paper, retrieve the correct academic publication from a collection of CORD-19 papers using advanced information retrieval techniques.

## 🏆 Key Results

### Subtask 4A Performance
- **Macro F1-Score**: 0.837 on development set
- **Model**: Fine-tuned DeBERTa-v3-base
- **Architecture**: Multi-label classification with 3 binary outputs

### Subtask 4B Performance
- **Best Model**: Multi-Stage Hybrid Retrieval
- **MRR@5**: 0.6209 on development set
- **Architecture**: BM25 + Dense Retrieval + Cross-Encoder Reranking

## 🚀 Quick Start

### Prerequisites
```bash
code tested on python 3.12.4 and 3.9

see requirements.txt

diverse ollama manifests
- install with: ollama run llama3.2
```

### Basic Usage

#### Tweet Classification (Subtask 4A)
```bash
cd subtask4a
python main.py --model debert --eval-test
```

#### Scientific Paper Retrieval (Subtask 4B)
```bash
best usage with the jupyter notebook for testing
```

## 📁 Project Structure

```
├── subtask4a/                    # Tweet Classification
│   ├── data/                     # Training and test datasets
│   ├── main.py                   # Main execution script
│   ├── debert.py                 # DeBERTa implementation
│   ├── llama.py                  # LLaMA-based classifier
│   └── baselines.ipynb           # Experimental baselines
│
├── subtask4b/                    # Scientific Paper Retrieval
│   ├── models/                   # Retrieval model implementations
│   │   ├── traditional_methods/  # BM25, TF-IDF
│   │   ├── representation_learning/ # Dense retrievers
│   │   ├── reranking_methods/    # Neural rerankers
│   │   └── hybrid_methods/       # Multi-stage systems
│   ├── vector_db/               # Vector database management
│   ├── evaluator.py             # Evaluation framework
│   ├── model_registry.py        # Model registry
│   └── paperQuest.ipynb         # Complete evaluation pipeline
│
└── docs/                        # Documentation and reports
```

## 🔧 Detailed Usage

### Subtask 4A: Tweet Classification

#### Training DeBERTa Model
```python
from subtask4a.debert import load_model_and_tokenizer, CT4A_DataLoader
from transformers import Trainer, TrainingArguments

# Load model and data
model, tokenizer = load_model_and_tokenizer("microsoft/deberta-v3-base")
dataloader = CT4A_DataLoader(tokenizer)
train_ds, _ = dataloader.get_dataset("data/ct_train.tsv")

# Configure training
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=2e-5
)

# Train model
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
trainer.train()
```

#### Using LLaMA for Classification
```python
from subtask4a.llama import run_llama_inference

# Run inference with LLaMA
run_llama_inference("data/ct_dev.tsv", output_dir="output")
```

### Subtask 4B: Scientific Paper Retrieval

#### Running Individual Models
```python
from subtask4b.evaluator import evaluate_models

config = {
    'collection_path': 'data/subtask4b_collection_data.pkl',
    'query_path': 'data/subtask4b_query_tweets_dev.tsv',
    'models': ['bm25', 'tfidf', 'multi_stage_hybrid'],
    'top_k': 5
}

results = evaluate_models(config)
```

#### Available Retrieval Models

| Category | Models | Description |
|----------|--------|-------------|
| **Traditional** | `bm25`, `tfidf` | Keyword-based retrieval |
| **Representation** | `custom_retriever`, `vector_store` | Semantic embedding-based |
| **Reranking** | `bm25_reranker`, `tfidf_reranker` | Two-stage with neural reranking |
| **Hybrid** | `multi_stage_hybrid` | Multi-signal fusion + reranking |
| **Extra** | `query_expansion` | LLM-enhanced query expansion |

#### Best Model Configuration
```python
# Multi-Stage Hybrid (Best Performing)
config = {
    'models': ['multi_stage_hybrid'],
    'embedding_model': 'sentence-transformers/allenai-specter',
    'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'rrf_k': 60,
    'sparse_weight': 0.6,
    'candidate_count': 50
}
```

## 📊 Performance Analysis

### Subtask 4A Results
| Model | Cat1 F1 | Cat2 F1 | Cat3 F1 | Macro F1 |
|-------|---------|---------|---------|----------|
| DeBERTa-v3 | 0.821 | 0.792 | 0.899 | **0.837** |
| LLaMA-based | 0.756 | 0.734 | 0.812 | 0.767 |

### Subtask 4B Results
| Method | MRR@1 | MRR@5 | MRR@10 |
|--------|-------|-------|--------|
| BM25 | 0.511 | 0.559 | 0.559 |
| TF-IDF + Reranker | 0.564 | 0.619 | 0.619 |
| **Multi-Stage Hybrid** | **0.572** | **0.621** | **0.621** |
| Vector Store + Reranker | 0.556 | 0.582 | 0.582 |
