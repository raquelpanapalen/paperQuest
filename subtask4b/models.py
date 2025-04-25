from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import numpy as np


class TFIDFWrapper:
    def __init__(self, corpus):
        """
        corpus: List of strings (documents)
        """
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def get_scores(self, query_text):
        """
        Returns cosine similarity scores between the query and all documents
        """
        query_vector = self.vectorizer.transform([query_text])
        cosine_similarities = (query_vector @ self.doc_vectors.T).toarray().flatten()
        return cosine_similarities


class BM25Wrapper:
    def __init__(self, tokenized_corpus):
        """
        tokenized_corpus: List of lists of tokens (e.g., [["doc", "one"], ["doc", "two"]])
        """
        self.model = BM25Okapi(tokenized_corpus)

    def get_scores(self, query_text):
        """
        Returns BM25 relevance scores for a tokenized query
        """
        if isinstance(query_text, str):
            query_text = query_text.split(" ")
        return self.model.get_scores(query_text)


class LangchainSemanticModel:
    def __init__(self, documents, model_name="llama3", k=5):
        """
        documents: List of LangChain Document objects
        model_name: Ollama model name (e.g. "llama3", "nomic-embed-text")
        k: Number of documents to retrieve
        """
        self.embedding = OllamaEmbeddings(model=model_name)
        self.vector_store = FAISS.from_documents(documents, self.embedding)
        self.k = k

    def get_scores(self, query_text):
        """
        Returns top-k document IDs (cord_uid) using vector similarity
        """
        results = self.vector_store.similarity_search(query_text, k=self.k)
        retrieved_ids = [doc.metadata["cord_uid"] for doc in results]
        return retrieved_ids
