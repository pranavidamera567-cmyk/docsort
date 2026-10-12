"""Text cleaning shared by classification and search."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_NON_WORD = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def preprocess(text):
    """Lowercase, strip digits/punctuation, drop stop words and 1-2 letter tokens."""
    text = _NON_WORD.sub(" ", (text or "").lower())
    tokens = [t for t in _SPACES.split(text) if len(t) > 2 and t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)
