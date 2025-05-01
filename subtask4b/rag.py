import os
import pandas as pd
from tqdm import tqdm

from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import CrossEncoder

PATH_COLLECTION_DATA = 'data/subtask4b_collection_data.pkl'
PATH_QUERY_TRAIN_DATA = 'data/subtask4b_query_tweets_train.tsv'
PATH_QUERY_DEV_DATA = 'data/subtask4b_query_tweets_dev.tsv'
OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_FIRST_N_PAPERS = 1200
EMBEDDING_MODEL = "nomic-embed-text"

print("Loading papers...")
papers_df = pd.read_pickle(PATH_COLLECTION_DATA)
print(f"Using {len(papers_df)} papers.")

documents = []
cord_uids = []

for _, row in papers_df.iterrows():
    content = f"""Title: {row['title']}\nAbstract: {row['abstract']}"""
    doc = Document(page_content=content, metadata={"cord_uid": row['cord_uid']})
    documents.append(doc)
    cord_uids.append(row['cord_uid'])

# === Embedding and Vector Store ===
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = FAISS.from_documents(documents, embedding)

# === Reranker ===
reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank_documents(query, retrieved_docs, top_k=5):
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]

class LangchainSemanticModel:
    def __init__(self, base_retriever, reranker, k=5):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.k = k

    def get_scores(self, query_text):
        retrieved_docs = self.base_retriever.get_relevant_documents(query_text)
        reranked_docs = rerank_documents(query_text, retrieved_docs, top_k=self.k)
        return [doc.metadata["cord_uid"] for doc in reranked_docs]

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
    # Load query tweets
    df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep="\t")
    df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep="\t")

    # Set up base retriever and semantic model
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    lc_model = LangchainSemanticModel(base_retriever, reranker, k=5)

    # Evaluate on training set
    print("Evaluating on training set...")
    performance_train, df_query_train = evaluate_model(lc_model, df_query_train, cord_uids)
    print("Performance on training set:", performance_train)

    # Evaluate on dev set
    print("Evaluating on dev set...")
    performance_dev, df_query_dev = evaluate_model(lc_model, df_query_dev, cord_uids)
    print("Performance on dev set:", performance_dev)

    # Save predictions
    df_query_dev["preds"] = df_query_dev["top_k"].apply(lambda x: x[:5])
    df_query_dev[["post_id", "preds"]].to_csv(
        os.path.join(OUTPUT_DIR, f"predictions_rag.csv"),
        index=False
    )
    print(f"Predictions saved to {OUTPUT_DIR}/predictions_rag.csv")