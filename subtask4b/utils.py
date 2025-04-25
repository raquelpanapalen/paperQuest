import pandas as pd
from tqdm import tqdm

def get_top_k_indexes(model, query, cord_uids, k=5):
    scores = model.get_scores(query)

    if isinstance(scores, list):
        # Already top cord_uids (LangChain semantic model)
        return scores[:k]
    else:
        # Need to sort by scores (BM25, TFIDF)
        top_k_indices = scores.argsort()[::-1][:k]
        top_k_papers = [cord_uids[i] for i in top_k_indices]
        return top_k_papers

def evaluate_model(model, df_query, cord_uids, list_k=[1, 5]):
    performance = {}

    tqdm.pandas()
    df_query["top_k"] = df_query["tweet_text"].progress_apply(
        lambda x: get_top_k_indexes(model, x, cord_uids, k=5)
    )

    for k in tqdm(list_k):
        reciprocal_ranks = []

        for _, row in tqdm(df_query.iterrows(), total=len(df_query)):
            gold = row["cord_uid"]
            predictions = row["top_k"]

            if gold in predictions[:k]:
                rank = predictions.index(gold) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

        performance[k] = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return performance, df_query