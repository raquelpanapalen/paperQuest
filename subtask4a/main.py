import argparse
import pandas as pd
from transformers import TrainingArguments, Trainer

from debert import (
    load_model_and_tokenizer,
    CT4A_DataLoader,
    annotate_output,
    compute_metrics
)
from llama import run_llama_inference


def run_deberta(eval_test=False):
    MODEL_NAME = "microsoft/deberta-v3-base"
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    print("Loading data...")
    dataloader = CT4A_DataLoader(tokenizer)
    train_ds, train_df = dataloader.get_dataset("data/ct_train.tsv")
    dev_ds, dev_df = dataloader.get_dataset("data/ct_dev.tsv")

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=256,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_strategy='no',
        save_strategy='no',
        no_cuda=False,
        report_to='none'
    )

    print("Training model...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds
    )
    trainer.train()

    print("Evaluating on dev set...")
    pred_output = trainer.predict(dev_ds)
    dev_df = annotate_output(dev_df, pred_output)

    print("Development Set Metrics:")
    dev_metrics = compute_metrics(dev_df)
    for k, v in dev_metrics.items():
        print(f"{k}: {v:.4f}")

    dev_df[["index", "cat1_pred", "cat2_pred", "cat3_pred"]].to_csv("predictions_dev.csv", index=False)
    print("Dev predictions saved to predictions_dev.csv")

    if eval_test:
        print("Evaluating on test set...")
        test_ds, test_df = dataloader.get_dataset("data/ct_test.tsv")
        test_output = trainer.predict(test_ds)
        test_df = annotate_output(test_df, test_output)

        print("Test Set Metrics:")
        test_metrics = compute_metrics(test_df)
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        test_df[["index", "cat1_pred", "cat2_pred", "cat3_pred"]].to_csv("predictions_test.csv", index=False)
        print("Test predictions saved to predictions_test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["debert", "llama"],
        default="debert",
        help="Choose which model to run: 'debert' or 'llama'"
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate on test set (ct_test.tsv) if available"
    )
    args = parser.parse_args()

    if args.model == "debert":
        run_deberta(eval_test=args.eval_test)
    elif args.model == "llama":
        run_llama_inference("data/ct_dev.tsv")

        if args.eval_test:
            run_llama_inference("data/ct_test.tsv")
