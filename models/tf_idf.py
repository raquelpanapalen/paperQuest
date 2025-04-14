from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFModel:
    def __init__(self, corpus):
        self.corpus = corpus

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tf_idf = self.vectorizer.fit_transform(self.corpus)

    def get_scores(self, query):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tf_idf).flatten()
        return scores
