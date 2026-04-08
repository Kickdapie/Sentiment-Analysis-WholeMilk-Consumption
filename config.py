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

# Topic and search (Phase 1: Reddit-only per advisor; whole milk *federal / school-meal policy*)
TOPIC = "Whole milk consumption school policy"
# Narrow queries: tie milk to USDA/federal law/bills—reduces unrelated "school" + "milk" posts.
SEARCH_QUERIES = [
    "whole milk healthy kids act",
    "healthy kids act milk",
    "Trump whole milk school",
    "Trump school milk bill",
    "USDA whole milk school lunch",
    "USDA school milk rule",
    "national school lunch program whole milk",
    "NSLP whole milk",
    "school lunch whole milk bill",
    "federal school milk policy",
    "whole milk ban schools federal",
    "school nutrition standards milk",
    "(whole milk OR \"2% milk\" OR \"school milk\") AND (USDA OR Trump OR bill OR law)",
    "(whole milk OR school milk) AND (Trump OR USDA OR federal OR law OR bill)",
    "(school lunch OR NSLP OR \"national school lunch\") AND (whole milk OR USDA)",
    "(USDA OR federal OR Congress) AND (school milk OR whole milk OR \"school lunch\")",
    "White House whole milk schools",
    "Senate school milk bill",
    "House school milk whole",
    # Extra coverage (still meal/program framing; helps hit ~3k after dedupe)
    "flavored milk school lunch policy",
    "chocolate milk school lunch USDA",
    "plant milk school lunch bill",
    "non dairy milk schools federal",
    "school breakfast milk USDA",
    "(school lunch OR breakfast) AND milk AND (USDA OR federal)",
    "school milk nutrition guideline",
    "Obama school lunch milk policy",
    "2% milk USDA schools",
    # Lexical / year variants to reduce Reddit search overlap
    "whole milk school 2025",
    "whole milk school 2026",
    "USDA school milk 2025",
    "dietary guidelines school milk",
    "school milk rule 2024",
    "RFK school milk",
    "MAHA school milk",
    "school milk lactose policy",
    "(whole OR 2%) AND milk AND (school lunch OR NSLP) AND (USDA OR federal)",
    # Bill/law specific variants to improve recall while staying policy-relevant
    "S.222 whole milk for healthy kids act",
    "H.R.649 whole milk for healthy kids act",
    "public law 119-69 whole milk",
    "whole milk for healthy kids act 2025 senate",
    "whole milk for healthy kids act 2025 house",
    "passed senate whole milk for healthy kids act",
    "passed house whole milk for healthy kids act",
    "trump signs whole milk for healthy kids act",
    "trump signs law returning whole milk to school lunches",
    "national school lunch program 2% milk whole milk law",
    "congress whole milk school lunch",
    "roger marshall whole milk for healthy kids act",
    "glenn thompson whole milk for healthy kids act",
    "school cafeterias whole milk bill",
    "school milk policy obama era limits",
    "nondairy milk alternative parent note schools law",
    "school lunch milk standards federal register",
    "usda final rule school meals milk",
    "school nutrition standards update milk",
    "dietary guidelines and school milk policy",
    # Intermediate-pass recall boosters (still on school meal + policy theme)
    "whole milk in school cafeterias",
    "school milk law",
    "school lunch milk policy",
    "whole milk school lunches policy",
    "whole milk back in schools",
    "USDA school meals milk standards",
    "school meal program milk options",
    "federal school lunch milk",
    "school nutrition policy milk",
    "healthy kids act school lunch milk",
]
# Strict second-pass relevance filter for policy-focused sentiment analysis.
FILTER_REDDIT_POLICY_RELEVANCE = True
REDDIT_EXCLUDED_SUBREDDITS = [
    "bestofredditorupdates",
    "amitheasshole",
    "aitah",
    "relationship_advice",
    "offmychest",
    "trueoffmychest",
    "teenagers",
    "genx",
    "cleaningtips",
    "nosleep",
    "tifu",
    "offmychest",
    "bestof",
    "autonewspaper",
    "autotldr",
    "health2020",
    "conspiracy",
]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Scraper
SCRAPED_RAW_PATH = os.path.join(DATA_DIR, "scraped_raw.csv")
SCRAPED_CLEAN_PATH = os.path.join(DATA_DIR, "scraped_clean.csv")
MAX_ITEMS_PER_QUERY = 120  # lower cap reduces API timeouts and keeps a tighter, cleaner corpus
SCRAPE_REDDIT_QUERY_DELAY = 1.25  # seconds between query strings (reduces Reddit 429 rate limits)
SCRAPE_REDDIT_MAX_RETRIES = 4     # retry attempts for transient Reddit API failures (esp. 429)
SCRAPE_REDDIT_BACKOFF_BASE = 2.0  # exponential backoff base seconds for retries
# Reddit comments: treat comments as separate analysis documents
REDDIT_INCLUDE_COMMENTS = True
REDDIT_MAX_COMMENTS_PER_POST = 10
REDDIT_COMMENT_MIN_CHARS = 20
SCRAPE_SOURCE = "reddit"   # fallback when SCRAPE_SOURCES not used
# Phase 1 (advisor): Reddit only — skip news RSS and Bluesky to reduce platform bias
SCRAPE_SOURCES = ["reddit"]
# SCRAPE_SOURCES = ["news_rss", "reddit", "bluesky"]  # optional multi-source (later phases)
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
