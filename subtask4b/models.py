from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM


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
    def __init__(self, documents, cord_uids, model_name="llama3", k=5):
        """
        documents: List of LangChain Document objects

        model_name: Ollama model name (e.g. "llama3", "nomic-embed-text")
        k: Number of documents to retrieve
        """
        self.embedding = OllamaEmbeddings(model=model_name)
        self.vector_store = FAISS.from_documents(
            documents, self.embedding, ids=cord_uids
        )
        self.k = k

    def get_scores(self, query_text):
        """
        Returns top-k document IDs (cord_uid) using vector similarity
        """
        results = self.vector_store.similarity_search(query_text, k=self.k)
        retrieved_ids = [doc.id for doc in results]
        return retrieved_ids


###### LLM Tweak ######


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


class LangchainSemanticModelTweak:
    def __init__(self, papers_df, documents, cord_uids, model_name="llama3", k=5):
        """
        documents: List of LangChain Document objects
        cord_uids: List of document IDs (cord_uid)
        model_name: Ollama model name (e.g. "llama3", "nomic-embed-text")
        k: Number of documents to retrieve
        """
        self.papers_df = papers_df
        self.embedding = OllamaEmbeddings(model=model_name)
        self.vector_store = FAISS.from_documents(
            documents, self.embedding, ids=cord_uids
        )
        self.k = k

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
        self.metadata_chain = prompt | llm | parser

    def extract_metadata(self, tweet_text):
        try:
            results = self.metadata_chain.invoke({"tweet_text": tweet_text})
            return TweetMetadata(
                authors=[
                    author for author in results.authors if author.lower() != "et al."
                ],
                journal=results.journal,
                year=results.year,
            )

        except Exception as e:
            return TweetMetadata(authors=[], journal="", year="")

    def filter_by_metadata(self, metadata):
        # make counter and only return the top 5 if the values are different, otherwise all of them
        filtered_papers = {}

        not_names = [
            "et al.",
            "Study",
            "Studies",
            "Research",
            "University",
            "Institute",
            "Unknown",
            "Other",
            "COVID",
            "Recovery",
            "Medicine",
            "Association",
            "Relationship",
            "School",
            "Covid",
            "Coronavirus",
            "Group",
            "Centre",
            "Working",
            "Israel",
            "Author",
            "Authors",
        ]

        # Filter by authors (for each author)
        for author in metadata.authors:
            names = author.split()
            for name in names:
                try:
                    """
                    check that the name starts with an uppercase letter,
                    all other letters are lowercase and no special characters,
                    at least 5 characters long
                    """
                    if (
                        not name.isalpha()
                        or not name[0].isupper()
                        or len(name) < 5
                        or name in not_names
                    ):
                        continue

                    results = self.papers_df[
                        self.papers_df["authors"].str.contains(
                            name, case=False, na=False
                        )
                    ]
                    print(f"Filtering by author: {name}, found {len(results)} results")
                    for _, row in results.iterrows():
                        if row["cord_uid"] not in filtered_papers:
                            filtered_papers[row["cord_uid"]] = 1
                        else:
                            filtered_papers[row["cord_uid"]] += 1

                except Exception as e:
                    print(
                        f"Error filtering by author {name} ({names} {author} {metadata.authors})"
                    )

        # If no authors were found, return empty list
        if not filtered_papers:
            return []

        # If the values are all the same and the number of unique values is less than 5, return all of them
        if len(set(filtered_papers.values())) == 1 and len(filtered_papers) < 5:
            filtered_rows_ids = list(filtered_papers.keys())
        else:
            # Sort by the number of occurrences and take the top 5
            sorted_filtered_papers = sorted(
                filtered_papers.items(), key=lambda x: x[1], reverse=True
            )
            filtered_rows_ids = [x[0] for x in sorted_filtered_papers[:5]]

        print(
            f"Filtered by authors: {len(filtered_rows_ids)} results, {filtered_rows_ids}"
        )
        return self.papers_df[
            self.papers_df["cord_uid"].isin(filtered_rows_ids)
        ].drop_duplicates(subset=["cord_uid"])

    def get_scores(self, query_text):
        metadata = self.extract_metadata(query_text)
        filtered_rows = self.filter_by_metadata(metadata)

        if len(filtered_rows):
            documents = []
            for _, row in filtered_rows.iterrows():
                content = f"""Title: {row['title']}
            Abstract: {row['abstract']}"""
                documents.append(Document(page_content=content))

            # Order them, make new vector store
            vector_store = FAISS.from_documents(
                documents, self.embedding, ids=filtered_rows["cord_uid"].tolist()
            )
            results = vector_store.similarity_search(query_text, k=self.k)
            results = [doc.id for doc in results]

        else:
            results = []

        if len(results) < 5:
            # Fallback to vector similarity search if not enough results
            additional_results = self.vector_store.similarity_search(
                query_text, k=self.k - len(results)
            )
            results.extend([doc.id for doc in additional_results])

        # Remove duplicates
        results = list(set(results))
        return results[: self.k]
