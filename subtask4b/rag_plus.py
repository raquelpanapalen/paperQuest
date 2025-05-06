import os
import pandas as pd
from tqdm import tqdm

from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import CrossEncoder

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM

PATH_COLLECTION_DATA = "data/subtask4b_collection_data.pkl"
PATH_QUERY_TRAIN_DATA = "data/subtask4b_query_tweets_train.tsv"
PATH_QUERY_DEV_DATA = "data/subtask4b_query_tweets_dev.tsv"
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
    doc = Document(page_content=content)
    documents.append(doc)
    cord_uids.append(row["cord_uid"])

# === Embedding and Vector Store ===
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = FAISS.from_documents(documents, embedding, ids=cord_uids)


# === Reranker ===
reranker = CrossEncoder("BAAI/bge-reranker-large")


def rerank_documents(query, retrieved_docs, top_k=5):
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]


# === Metadata Extractor via Ollama ===


# Define Metadata Schema
class TweetMetadata(BaseModel):
    authors: Optional[list[str]] = Field(
        None, description="The name(s) of the author(s) referenced in the tweet, if any"
    )
    journal: Optional[str] = Field(
        "", description="The name of the journal mentioned, if any"
    )
    year: Optional[str] = Field(
        "", description="The publication year if it appears in the tweet"
    )


# Output parser setup
parser = PydanticOutputParser(pydantic_object=TweetMetadata)
format_instructions = (
    parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    f"""
You are an assistant that extracts bibliographic metadata from tweets that mention scientific papers.

For the tweet below, extract the following metadata if it exists:
- authors: A list of authors (Name and/or Surname of a Person) mentioned in the tweet, usually followed by "et al." or "and". Do not include "et al." or "and" in the list.
- journal: The name of the journal or conference mentioned in the tweet
- year: The year of publication, usually a 4-digit number

If the information is not present, return an empty string or an empty list for that field.

Return your answer in this JSON format:
{format_instructions}

Tweet: {{tweet_text}}
"""
)

# Ollama model
llm = OllamaLLM(model="llama3")

# The metadata extraction chain
metadata_chain = prompt | llm | parser


def extract_metadata(tweet_text):
    try:
        results = metadata_chain.invoke({"tweet_text": tweet_text})
        return TweetMetadata(
            authors=[
                author for author in results.authors if author.lower() != "et al."
            ],
            journal=results.journal,
            year=results.year,
        )

    except Exception as e:
        return TweetMetadata(authors=[], journal="", year="")


def filter_by_metadata(papers_df, metadata):
    filtered_rows_ids = []

    # Filter by authors (for each author)
    for author in metadata.authors:
        names = author.split()
        for name in names:
            try:
                # only keep alpha characters
                name = "".join([c for c in name if c.isalpha()])
                filtered_rows_ids.extend(
                    papers_df[
                        papers_df["authors"].str.contains(name, case=False, na=False)
                    ]["cord_uid"].tolist()
                )
            except Exception as e:
                print(
                    f"Error filtering by author {name} ({names} {author} {metadata.authors})"
                )

    return papers_df[papers_df["cord_uid"].isin(filtered_rows_ids)].drop_duplicates(
        subset=["cord_uid"]
    )


class LangchainSemanticModel:
    def __init__(self, base_retriever, reranker, k=5):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.k = k

    def get_scores(self, query_text):
        metadata = extract_metadata(query_text)
        filtered_rows = filter_by_metadata(papers_df, metadata)
        # get the documents from the store
        if len(filtered_rows) < 5:
            # if there are not enough documents, get the top k from the base retriever
            retrieved_docs = self.base_retriever.get_relevant_documents(query_text)

            if len(filtered_rows) > 0:
                filtered_docs = vector_store.get_by_ids(
                    filtered_rows["cord_uid"].tolist()
                )
                retrieved_docs.extend(filtered_docs)

        else:
            # if there are enough documents, get the top k from the filtered rows
            retrieved_docs = vector_store.get_by_ids(filtered_rows["cord_uid"].tolist())

        # rerank the documents
        reranked_docs = rerank_documents(query_text, retrieved_docs, top_k=self.k)
        return [doc.id for doc in reranked_docs]


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
    performance_train, df_query_train = evaluate_model(
        lc_model, df_query_train, cord_uids
    )
    print("Performance on training set:", performance_train)

    # Evaluate on dev set
    print("Evaluating on dev set...")
    performance_dev, df_query_dev = evaluate_model(lc_model, df_query_dev, cord_uids)
    print("Performance on dev set:", performance_dev)

    # Save predictions
    df_query_dev["preds"] = df_query_dev["top_k"].apply(lambda x: x[:5])
    df_query_dev[["post_id", "preds"]].to_csv(
        os.path.join(OUTPUT_DIR, f"predictions_rag.csv"), index=False
    )
    print(f"Predictions saved to {OUTPUT_DIR}/predictions_rag.csv")
