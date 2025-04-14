from tqdm import tqdm
import pandas as pd
from rank_bm25 import BM25Okapi

from models import TFIDFModel


def get_top_k_indexes(model, query, k=10):
    """
    Get the top k indices for a given query using the specified model.
    """
    scores = model.get_scores(query)
    top_k_indices = scores.argsort()[::-1][:k]  # Get the top k indices
    return top_k_indices


def evaluate_model(model, df_query, list_k=[1, 5, 10]):
    """
    Evaluate the model on the given query dataset.
    """
    performance = {}

    df_query["top_k"] = df_query["tweet_text"].apply(
        lambda x: get_top_k_indexes(model, x, k=10)
    )

    for k in tqdm(list_k):
        for _, row in tqdm(df_query.iterrows()):
            gold = row["cord_uid"]
            predictions = row["top_k"]

            # Check if the gold standard is in the top k results
            reciprocal_ranks = []
            if gold in predictions[:k]:
                rank = predictions.index(gold) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

        performance[k] = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return performance


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
        performance_train = evaluate_model(model, df_query_train)
        print(f"Performance on training set: {performance_train}")

        # Evaluate on the development set
        print("Evaluating on development set...")
        performance_dev = evaluate_model(model, df_query_dev)
        print(f"Performance on development set: {performance_dev}")
