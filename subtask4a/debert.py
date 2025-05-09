#partly extracted from provided baselines
import ast
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DebertaV2Tokenizer

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class CT4A_DataLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_dataset(self, dataset_path):
        data = pd.read_csv(dataset_path, sep='\t')
        data['labels'] = data['labels'].apply(lambda x: ast.literal_eval(x))

        def tokenize(examples):
            return self.tokenizer(examples["text"], max_length=128, truncation=True, padding='max_length')

        dataset = Dataset.from_pandas(data[['text', 'labels']])
        dataset = dataset.map(tokenize, batched=True, batch_size=32, remove_columns=['text'])

        return dataset, data

def compute_metrics(data):
    metrics = {}
    for i, cat in enumerate(['cat1', 'cat2', 'cat3']):
        preds = data[f'{cat}_pred'].astype(int)
        labels = data['labels'].apply(lambda x: int(x[i]))

        metrics.update({
            f'{cat}_acc': accuracy_score(labels, preds),
            f'{cat}_prec': precision_score(labels, preds),
            f'{cat}_rec': recall_score(labels, preds),
            f'{cat}_f1': f1_score(labels, preds)
        })

    preds_all = data[[c for c in data.columns if '_pred' in c]].values.tolist()
    labels_all = data['labels'].tolist()
    metrics["macro_f1"] = f1_score(labels_all, preds_all, average="macro")

    return metrics

def annotate_output(data, pred_output):
    logits = pred_output.predictions
    predictions = (logits > 0).astype(int)

    for i, cat in enumerate(['cat1', 'cat2', 'cat3']):
        data[f'{cat}_logits'] = logits[:, i]
        data[f'{cat}_pred'] = predictions[:, i]
        data[f'{cat}_score'] = sigmoid(logits[:, i])

    return data

def load_model_and_tokenizer(model_name="microsoft/deberta-v3-base"):
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        problem_type="multi_label_classification"
    )
    return model, tokenizer
