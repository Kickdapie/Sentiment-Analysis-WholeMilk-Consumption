"""
Configuration for Whole Milk School Policy Sentiment Analysis.
Adapted from methodology in PeerJ CS 1149 (vegan tweets sentiment analysis).
"""

import os

# Load .env so TWITTER_BEARER_TOKEN can be set there (optional; pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
    "whole milk kids",
    "school cafeteria milk",
    "USDA school milk",
    "whole milk ban schools",
    "whole milk healthy kids act",
    "2% milk school lunch",
    "skim milk school policy",
    "dairy school nutrition",
    "children whole milk",
    "school milk fat",
    "milk school cafeteria policy",
]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Scraper
SCRAPED_RAW_PATH = os.path.join(DATA_DIR, "scraped_raw.csv")
SCRAPED_CLEAN_PATH = os.path.join(DATA_DIR, "scraped_clean.csv")
MAX_ITEMS_PER_QUERY = 300  # per-query cap; target ~2500+ total (News + Reddit)
SCRAPE_SOURCE = "reddit"   # fallback when SCRAPE_SOURCES not used
SCRAPE_SOURCES = ["news_rss", "reddit", "bluesky"]  # bluesky: public API, no auth needed; twitter: set TWITTER_BEARER_TOKEN
# Bluesky (free): set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD in .env for authenticated access
# Create app password at: https://bsky.app/settings/app-passwords
BLUESKY_HANDLE = os.environ.get("BLUESKY_HANDLE", "").strip()
BLUESKY_APP_PASSWORD = os.environ.get("BLUESKY_APP_PASSWORD", "").strip()

# Twitter Developer (free tier): set env var TWITTER_BEARER_TOKEN to use official API (no snscrape needed)
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "").strip()

# Kaggle dataset: "sentiment140" or "alternative" (Twitter Sentiment Dataset by saurabhshahane)
# See DATASETS.md. Download: https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset
KAGGLE_DATASET = "alternative"
KAGGLE_RAW_PATH = os.path.join(DATA_DIR, "kaggle_sentiment140", "training.1600000.processed.noemoticon.csv")
# Alternative: saurabhshahane Twitter Sentiment (Twitter_Data.csv: clean_text, category -1/0/1)
KAGGLE_ALT_PATH = os.path.join(DATA_DIR, "Twitter_Data.csv")
KAGGLE_ALT_TEXT_COL = "clean_text"
KAGGLE_ALT_LABEL_COL = "category"  # -1=negative, 0=neutral, 1=positive
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

# Model type: "bert" (recommended for paper) or "tfidf" (fast, traditional)
SENTIMENT_MODEL_TYPE = "bert"
# Pre-trained BERT model for sentiment (RoBERTa fine-tuned on ~124M tweets, 3-class: neg/neu/pos)
BERT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Sentiment classes (Sentiment140: 0=negative, 4=positive)
SENTIMENT_LABELS = {0: "negative", 4: "positive"}
NEUTRAL_THRESHOLD = 0.1  # for score-based neutral zone

# Network analysis
MIN_KEYWORD_FREQ = 3
TOP_N_KEYWORDS = 80
CO_OCCURRENCE_WINDOW = 5  # words within window co-occur

# NLTK (run once: python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')")
NLTK_STOPWORDS = "english"

for d in (DATA_DIR, OUTPUT_DIR, MODEL_DIR, os.path.join(DATA_DIR, "kaggle_alt")):
    os.makedirs(d, exist_ok=True)
