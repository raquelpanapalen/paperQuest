import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report


DATA_DIR = "data/"
OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "ct_train.tsv")
DEV_FILE = os.path.join(DATA_DIR, "ct_dev.tsv")


train_df = pd.read_csv(TRAIN_FILE, sep="\t")
train_df['labels'] = train_df['labels'].apply(eval)  

dev_df = pd.read_csv(DEV_FILE, sep="\t")


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    problem_type="multi_label_classification"
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].tolist(),
    train_df['labels'].tolist(),
    test_size=0.1,
    random_state=42
)

train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
val_dataset = TweetDataset(val_texts, val_labels, tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


trainer.train()

model.save_pretrained("./tweet_bert_classifier")
tokenizer.save_pretrained("./tweet_bert_classifier")

model = BertForSequenceClassification.from_pretrained("./tweet_bert_classifier")
tokenizer = BertTokenizer.from_pretrained("./tweet_bert_classifier")
model.eval()

predictions = []

with torch.no_grad():
    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df)):
        inputs = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).int().squeeze().tolist()

        if isinstance(pred, int):
            pred = [pred]
        if len(pred) < 3:
            pred = pred + [0] * (3 - len(pred))

        predictions.append({
            "index": row["index"],
            "cat1_pred": pred[0],
            "cat2_pred": pred[1],
            "cat3_pred": pred[2],
        })

pred_df = pd.DataFrame(predictions)

pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_classification_bert.csv"), index=False)
print(f"Predictions saved to {os.path.join(OUTPUT_DIR, 'predictions_classification_bert.csv')}")

# Merge predictions with ground truth
merged_df = pd.merge(dev_df, pred_df, on="index")

# Extract true labels and predicted labels
y_true = merged_df[["cat1", "cat2", "cat3"]].values
y_pred = merged_df[["cat1_pred", "cat2_pred", "cat3_pred"]].values

# Macro-Averaged F1 Score
macro_f1 = f1_score(y_true, y_pred, average="macro")
print(f"\nMacro-Averaged F1 Score: {macro_f1:.4f}")

# Detailed Report
report = classification_report(y_true, y_pred, target_names=["scientific_claim", "scientific_reference", "scientific_entity"])
print("\nDetailed Classification Report:")
print(report)
