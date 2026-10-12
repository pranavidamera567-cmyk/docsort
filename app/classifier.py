"""TF-IDF + logistic regression document classifier."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .preprocess import preprocess

HIGH, MEDIUM = 0.7, 0.4


def confidence_level(p):
    return "high" if p >= HIGH else "medium" if p >= MEDIUM else "low"


def build_pipeline():
    return make_pipeline(
        TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=10),
    )


class Classifier:
    def __init__(self):
        self.model = None
        self.classes = []

    def train(self, texts, labels):
        if len(set(labels)) < 2:
            raise ValueError("Need training examples for at least two categories")
        self.model = build_pipeline().fit(texts, labels)
        self.classes = list(self.model.classes_)

    def predict(self, text):
        """Returns (category, confidence, top3) where top3 is [(category, probability), ...]."""
        proba = self.model.predict_proba([text])[0]
        ranked = sorted(zip(self.classes, proba), key=lambda x: -x[1])
        return ranked[0][0], float(ranked[0][1]), [(c, round(float(p), 3)) for c, p in ranked[:3]]
