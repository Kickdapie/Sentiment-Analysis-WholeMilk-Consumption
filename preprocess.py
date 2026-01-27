"""
Text preprocessing for sentiment analysis (cleaning, tokenization, stopwords).
Aligns with PeerJ CS 1149 pipeline: cleaning, case folding, tokenization, stopword removal.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import config

# Ensure NLTK data exists (run once); newer NLTK uses punkt_tab for word_tokenize
for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)
try:
    stopwords.words(config.NLTK_STOPWORDS)
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP = set(stopwords.words(config.NLTK_STOPWORDS))


def clean_for_nlp(text):
    """Remove URLs, mentions, extra punctuation; lowercase."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).lower().strip()
    return text


def tokenize(text):
    """Tokenize and remove stopwords; return list of tokens."""
    text = clean_for_nlp(text)
    if not text:
        return []
    tokens = word_tokenize(text)
    return [t for t in tokens if t.isalnum() and t not in STOP and len(t) > 1]


def tokenize_keep_stopwords(text):
    """Tokenize without removing stopwords (for some network/PMI use)."""
    text = clean_for_nlp(text)
    if not text:
        return []
    return word_tokenize(text)
