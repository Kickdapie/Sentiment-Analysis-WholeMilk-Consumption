"""
Load trained sentiment model and predict on scraped texts.
Outputs labels (positive/negative/neutral) and scores for final report.
"""

import os
import numpy as np
import pandas as pd
import joblib
from preprocess import clean_for_nlp
import config


def load_model():
    """Load saved vectorizer and classifier."""
    if not os.path.isfile(config.SENTIMENT_MODEL_PATH) or not os.path.isfile(config.VECTORIZER_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run: python train_sentiment_model.py\n"
            "(Requires Kaggle Sentiment140 in data/kaggle_sentiment140/)"
        )
    vectorizer = joblib.load(config.VECTORIZER_PATH)
    model = joblib.load(config.SENTIMENT_MODEL_PATH)
    return vectorizer, model


def predict_texts(texts, vectorizer, model, neutral_threshold=config.NEUTRAL_THRESHOLD):
    """
    Predict sentiment for list of raw texts.
    Returns labels ('positive'/'negative'/'neutral') and scores (prob positive).
    """
    cleaned = [clean_for_nlp(t) for t in texts]
    X = vectorizer.transform(cleaned)
    proba = model.predict_proba(X)[:, 1]  # P(positive)
    labels = []
    for p in proba:
        if p >= 0.5 + neutral_threshold:
            labels.append("positive")
        elif p <= 0.5 - neutral_threshold:
            labels.append("negative")
        else:
            labels.append("neutral")
    return labels, proba


def predict_dataframe(df, text_col="text"):
    """Add columns sentiment_label and sentiment_score to dataframe."""
    vectorizer, model = load_model()
    texts = df[text_col].fillna("").astype(str).tolist()
    labels, scores = predict_texts(texts, vectorizer, model)
    df = df.copy()
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    return df
