import os
import pandas as pd
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

PATH_COLLECTION_DATA = 'data/subtask4b_collection_data.pkl'
PATH_QUERY_TRAIN_DATA = 'data/subtask4b_query_tweets_train.tsv'
PATH_QUERY_DEV_DATA = 'data/subtask4b_query_tweets_dev.tsv'

OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_FIRST_N_PAPERS = 5  

EMBEDDING_MODEL = "llama3"  

print("Loading papers...")
papers_df = pd.read_pickle(PATH_COLLECTION_DATA)

if USE_FIRST_N_PAPERS:
    papers_df = papers_df.head(USE_FIRST_N_PAPERS)

print(f"Using {len(papers_df)} papers.")

documents = []
cord_uids = []

for _, row in papers_df.iterrows():
    content = f"""Title: {row['title']}
Date: {row['publish_time']}
Journal: {row['journal']}
Authors: {row['authors']}
Abstract: {row['abstract']}"""
    
    doc = Document(page_content=content, metadata={"cord_uid": row['cord_uid']})
    documents.append(doc)
    cord_uids.append(row['cord_uid'])

# Embedding and Vector Store
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = FAISS.from_documents(documents, embedding)

class LangchainSemanticModel:
    def __init__(self, vector_store, k=5):
        self.vector_store = vector_store
        self.k = k

    def get_scores(self, query_text):
        results = self.vector_store.similarity_search(query_text, k=self.k)
        retrieved_ids = [doc.metadata["cord_uid"] for doc in results]
        return retrieved_ids


def get_top_k_indexes(model, query, cord_uids, k=5):
    return model.get_scores(query)

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


if __name__ == "__main__":
    # Load Query Tweets
    df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep="\t")
    df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep="\t")

    # Initialize semantic model
    lc_model = LangchainSemanticModel(vector_store, k=5)

    # Evaluate on train
    print("Evaluating on training set...")
    performance_train, df_query_train = evaluate_model(lc_model, df_query_train, cord_uids)
    print("Performance on training set:", performance_train)

    # Evaluate on dev
    print("Evaluating on dev set...")
    performance_dev, df_query_dev = evaluate_model(lc_model, df_query_dev, cord_uids)
    print("Performance on dev set:", performance_dev)

    # Save predictions
    df_query_dev["preds"] = df_query_dev["top_k"].apply(lambda x: x[:5])
    df_query_dev[["post_id", "preds"]].to_csv(
        os.path.join(OUTPUT_DIR, f"predictions_langchain.csv"),
        index=False
    )

    print(f"Predictions saved to {OUTPUT_DIR}/predictions_langchain.csv")
