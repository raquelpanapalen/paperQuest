import os
import pandas as pd
from tqdm import tqdm
from langchain.schema import Document

from models import BM25Wrapper, TFIDFWrapper, LangchainSemanticModel
from utils import evaluate_model


PATH_COLLECTION_DATA = 'subtask4b/data/subtask4b_collection_data.pkl'
PATH_QUERY_TRAIN_DATA = 'subtask4b/data/subtask4b_query_tweets_train.tsv'
PATH_QUERY_DEV_DATA = 'subtask4b/data/subtask4b_query_tweets_dev.tsv'
OUTPUT_DIR = "output/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_FIRST_N_PAPERS = None
EMBEDDING_MODEL = "llama3"

papers_df = pd.read_pickle(PATH_COLLECTION_DATA)
if USE_FIRST_N_PAPERS:
    papers_df = papers_df.head(USE_FIRST_N_PAPERS)

print(f"Using {len(papers_df)} papers.")

corpus = papers_df[["title", "abstract"]].apply(lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
tokenized_corpus = [doc.split(" ") for doc in corpus]
cord_uids = papers_df["cord_uid"].tolist()


documents = []
for _, row in papers_df.iterrows():
    content = f"""Title: {row['title']}
Date: {row['publish_time']}
Journal: {row['journal']}
Authors: {row['authors']}
Abstract: {row['abstract']}"""
    documents.append(Document(page_content=content, metadata={"cord_uid": row['cord_uid']}))


df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep="\t")
df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep="\t")


models = {
    "BM25": BM25Wrapper(tokenized_corpus),
    "TFIDF": TFIDFWrapper(corpus),
    "LangChain": LangchainSemanticModel(documents, model_name=EMBEDDING_MODEL)
}

for model_name, model in models.items():
    print(f"\nEvaluating model: {model_name}")

    print("Training set:")
    performance_train, df_query_train = evaluate_model(model, df_query_train, cord_uids)
    print(f"Performance: {performance_train}")

    print("Dev set:")
    performance_dev, df_query_dev = evaluate_model(model, df_query_dev, cord_uids)
    print(f"Performance: {performance_dev}")

    df_query_dev["preds"] = df_query_dev["top_k"].apply(lambda x: x[:5])
    df_query_dev[["post_id", "preds"]].to_csv(
        os.path.join(OUTPUT_DIR, f"predictions_{model_name}.csv"), index=False
    )
    print(f"Predictions saved: predictions_{model_name}.csv")