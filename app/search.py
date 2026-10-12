"""Keyword search with TF-IDF relevance ranking."""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .preprocess import preprocess


def snippet(text, terms, width=160):
    m = re.search("|".join(re.escape(t) for t in terms), text, re.IGNORECASE)
    start = max(0, (m.start() if m else 0) - width // 4)
    out = " ".join(text[start:start + width].split())
    return ("…" if start else "") + out + ("…" if start + width < len(text) else "")


def search(docs, query, category=None):
    """docs: iterable of dict-like rows with id, text, category. Returns [(row, score, snippet)] best first.

    A document must contain at least one query term; ranking is cosine similarity between the
    query and the document in TF-IDF space (idf computed over the whole collection).
    """
    docs = list(docs)
    terms = preprocess(query).split()
    if not docs or not terms:
        return []
    vec = TfidfVectorizer(preprocessor=preprocess, sublinear_tf=True)
    try:
        matrix = vec.fit_transform([d["text"] for d in docs])
    except ValueError:  # empty vocabulary
        return []
    scores = linear_kernel(vec.transform([query]), matrix)[0]
    results = []
    for doc, score in zip(docs, scores):
        if category and doc["category"] != category:
            continue
        tokens = set(preprocess(doc["text"]).split())
        if score > 0 and any(t in tokens for t in terms):
            results.append((doc, round(float(score), 4), snippet(doc["text"], terms)))
    return sorted(results, key=lambda r: -r[1])
