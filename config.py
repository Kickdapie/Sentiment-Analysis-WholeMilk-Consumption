"""
Configuration for Whole Milk School Policy Sentiment Analysis.
Adapted from methodology in PeerJ CS 1149 (vegan tweets sentiment analysis).
"""

import os

# Topic and search
TOPIC = "Whole milk consumption school policy"
SEARCH_QUERIES = [
    "whole milk school",
    "school milk policy",
    "whole milk school lunch",
    "school lunch milk",
    "milk in schools",
    "flavored milk schools",
    "school nutrition milk",
]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Scraper
SCRAPED_RAW_PATH = os.path.join(DATA_DIR, "scraped_raw.csv")
SCRAPED_CLEAN_PATH = os.path.join(DATA_DIR, "scraped_clean.csv")
MAX_ITEMS_PER_QUERY = 200  # limit per query to keep runs fast
SCRAPE_SOURCE = "reddit"   # "reddit" | "news_rss" | "twitter" (twitter needs snscrape)

# Kaggle dataset (Sentiment140 or similar)
# Download from: https://www.kaggle.com/datasets/kazanova/sentiment140
KAGGLE_RAW_PATH = os.path.join(DATA_DIR, "kaggle_sentiment140", "training.1600000.processed.noemoticon.csv")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

# Sentiment classes (Sentiment140: 0=negative, 4=positive)
SENTIMENT_LABELS = {0: "negative", 4: "positive"}
NEUTRAL_THRESHOLD = 0.1  # for score-based neutral zone

# Network analysis
MIN_KEYWORD_FREQ = 3
TOP_N_KEYWORDS = 80
CO_OCCURRENCE_WINDOW = 5  # words within window co-occur

# NLTK (run once: python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')")
NLTK_STOPWORDS = "english"

for d in (DATA_DIR, OUTPUT_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)
