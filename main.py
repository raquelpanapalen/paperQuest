import os
from tqdm import tqdm
import pandas as pd
from rank_bm25 import BM25Okapi

from models import TFIDFModel


def get_top_k_indexes(model, query, cord_uids, k=10):
    """
    Get the top k indices for a given query using the specified model.
    """
    scores = model.get_scores(query)
    top_k_indices = scores.argsort()[::-1][:k]  # Get the top k indices
    top_k_papers = [cord_uids[i] for i in top_k_indices]  # Map indices to cord_uids
    return top_k_papers


def evaluate_model(model, df_query, cord_uids, list_k=[1, 5]):
    """
    Evaluate the model on the given query dataset.
    """
    performance = {}

    tqdm.pandas()
    df_query["top_k"] = df_query["tweet_text"].progress_apply(
        lambda x: (
            get_top_k_indexes(model, x.split(" "), cord_uids, k=5)
            if model.__class__.__name__ == "BM25Okapi"
            else get_top_k_indexes(model, x, cord_uids, k=5)
        )
    )

    for k in tqdm(list_k):
        # Calculate the performance for each k
        reciprocal_ranks = []

        for _, row in tqdm(df_query.iterrows(), total=len(df_query)):
            gold = row["cord_uid"]
            predictions = row["top_k"]

            # Check if the gold standard is in the top k results
            if gold in predictions[:k]:
                rank = predictions.index(gold) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

        performance[k] = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return performance, df_query


if __name__ == "__main__":
    # Load your dataset
    PATH_COLLECTION_DATA = "data/subtask4b_collection_data.pkl"
    df = pd.read_pickle(PATH_COLLECTION_DATA)

    # Create corpus
    corpus = (
        df[:][["title", "abstract"]]
        .apply(lambda x: f"{x['title']} {x['abstract']}", axis=1)
        .tolist()
    )
    cord_uids = df[:]["cord_uid"].tolist()
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    # Load query tweets
    PATH_QUERY_TRAIN_DATA = "data/subtask4b_query_tweets_train.tsv"
    PATH_QUERY_DEV_DATA = "data/subtask4b_query_tweets_dev.tsv"

    df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep="\t")
    df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep="\t")

    # Initialize the models
    models = [BM25Okapi(tokenized_corpus), TFIDFModel(corpus)]

    for model in models:
        print(f"Evaluating model: {model.__class__.__name__}")

        # Evaluate on the training set
        print("Evaluating on training set...")
        performance_train, df_query_train = evaluate_model(
            model, df_query_train, cord_uids
        )
        print(f"Performance on training set: {performance_train}")

        # Evaluate on the development set
        print("Evaluating on development set...")
        performance_dev, df_query_dev = evaluate_model(model, df_query_dev, cord_uids)
        print(f"Performance on development set: {performance_dev}")

        output_dir = "output/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the performance metrics
        df_query_dev["preds"] = df_query_dev["top_k"].apply(
            lambda x: x[:5]
        )  # We only need the top 5 predictions
        df_query_dev[["post_id", "preds"]].to_csv(
            f"output/predictions_{model.__class__.__name__}.csv", index=False
        )
