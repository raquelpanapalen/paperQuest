import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ast

def compute_metrics_from_csv(test_path="data/ct_test.tsv", pred_path="predictions.csv"):
    # Load gold labels
    df_gold = pd.read_csv(test_path, sep="\t")
    df_gold['labels'] = df_gold['labels'].apply(ast.literal_eval)

    # Load predictions
    df_pred = pd.read_csv(pred_path)

    # Merge on index
    df = pd.merge(df_gold, df_pred, on="index", how="inner")

    # Compute per-label metrics
    metrics = {}
    for i, cat in enumerate(['cat1_pred', 'cat2_pred', 'cat3_pred']):
        y_true = df['labels'].apply(lambda x: x[i]).astype(int)
        y_pred = df[cat].astype(int)

        metrics[f'{cat}_acc'] = accuracy_score(y_true, y_pred)
        metrics[f'{cat}_prec'] = precision_score(y_true, y_pred)
        metrics[f'{cat}_rec'] = recall_score(y_true, y_pred)
        metrics[f'{cat}_f1'] = f1_score(y_true, y_pred)

    # Compute macro F1 across all 3 labels
    y_true_all = df['labels'].tolist()
    y_pred_all = df[['cat1_pred', 'cat2_pred', 'cat3_pred']].values.tolist()
    metrics['macro_f1'] = f1_score(y_true_all, y_pred_all, average='macro')

    return metrics


if __name__ == "__main__":
    metrics = compute_metrics_from_csv("data/ct_test.tsv", "predictions.csv")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
