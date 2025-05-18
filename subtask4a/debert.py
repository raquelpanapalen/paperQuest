#partly extracted from provided baselines and adapted
import ast
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_metrics(data):
    metrics = {}
    for i, _cat in enumerate(['cat1', 'cat2', 'cat3']):
        preds = data[f'{_cat}_pred'].apply(lambda x: int(x) == 1)
        labels = data['labels'].apply(lambda x: int(float(x[i])) == 1)

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        metrics.update({f'{_cat}_avg_acc': acc,
                        f'{_cat}_avg_prec': prec,
                        f'{_cat}_avg_rec': rec,
                        f'{_cat}_avg_f1': f1})

    preds = data[[c for c in data.columns if '_pred' in c]].values.tolist()
    labels = data['labels'].tolist()
    metrics["macro_f1"] = f1_score(labels, preds, average="macro")

    return metrics

def annotate_test_dataframe(data, pred_output):
    data['cat1_logits'] = pred_output.predictions[:, 0]
    data['cat2_logits'] = pred_output.predictions[:, 1]
    data['cat3_logits'] = pred_output.predictions[:, 2]

    predictions = (pred_output.predictions > 0).astype(int)
    data['cat1_pred'] = predictions[:, 0]
    data['cat2_pred'] = predictions[:, 1]
    data['cat3_pred'] = predictions[:, 2]

    data['cat1_score'] = sigmoid(pred_output.predictions[:, 0])
    data['cat2_score'] = sigmoid(pred_output.predictions[:, 1])
    data['cat3_score'] = sigmoid(pred_output.predictions[:, 2])

    return data

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

def run_bert_training():
    model_path = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, problem_type="multi_label_classification")

    dl = CT4A_DataLoader(tokenizer)
    train_ds, _ = dl.get_dataset("data/ct_train.tsv")
    dev_ds, dev_df = dl.get_dataset("data/ct_dev.tsv")

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=256,
        warmup_ratio=0.1,
        weight_decay=0.05,
        learning_rate=3e-5,
        logging_strategy='no',
        save_strategy='no',
        evaluation_strategy='no',
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds
    )

    trainer.train()

    dev_pred_output = trainer.predict(dev_ds)
    dev_df = annotate_test_dataframe(dev_df, dev_pred_output)
    metrics = compute_metrics(dev_df)
    print("Dev Metrics:")
    print(metrics)

    dev_df[["index", "cat1_pred", "cat2_pred", "cat3_pred"]].to_csv("output/bert_dev_predictions.csv", index=False)
    model.save_pretrained("results")
    tokenizer.save_pretrained("results")

if __name__ == "__main__":
    run_bert_training()
