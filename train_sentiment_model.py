"""
Train sentiment classifier using Kaggle Sentiment140 dataset (or similar).
Uses TF-IDF + Logistic Regression; option to filter by domain keywords like PeerJ CS 1149.
Saves model and vectorizer for use in pipeline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

import config
from preprocess import clean_for_nlp


# Sentiment140 columns: target, id, date, flag, user, text
S140_COLUMNS = ["target", "id", "date", "flag", "user", "text"]


def load_alternative_dataset(path=None, text_col=None, label_col=None):
    """Load alternative Kaggle CSV (e.g. saurabhshahane Twitter Sentiment). Must have text + label columns."""
    path = path or config.KAGGLE_ALT_PATH
    text_col = text_col or config.KAGGLE_ALT_TEXT_COL
    label_col = label_col or config.KAGGLE_ALT_LABEL_COL
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Alternative dataset not found at {path}. "
            "Place Twitter_Data.csv in the data/ folder, then run again."
        )
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    # Auto-detect text column (saurabhshahane and others often use "tweet" or "text")
    if text_col not in df.columns:
        text_col = "tweet" if "tweet" in df.columns else ("text" if "text" in df.columns else text_col)
    if label_col not in df.columns:
        label_col = "label" if "label" in df.columns else ("sentiment" if "sentiment" in df.columns else label_col)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV must have a text column and a label column. Found: {list(df.columns)}"
        )
    df = df[[text_col, label_col]].dropna()
    # Map labels to 0/1 (negative/positive); handle numeric -1/0/1 (e.g. 1.0, -1.0) and strings
    raw_num = pd.to_numeric(df[label_col], errors="coerce")
    is_num = raw_num.notna()
    df["label"] = np.nan
    df.loc[is_num, "label"] = np.where(raw_num[is_num] == 1, 1, np.where(raw_num[is_num] == -1, 0, np.nan))
    # Non-numeric: "negative"/"positive", "0"/"4", etc.
    if (~is_num).any():
        raw_str = df.loc[~is_num, label_col].astype(str).str.lower().str.strip()
        neg_vals = {"0", "-1", "negative", "neg"}
        pos_vals = {"1", "4", "positive", "pos"}
        df.loc[~is_num, "label"] = np.where(raw_str.isin(pos_vals), 1, np.where(raw_str.isin(neg_vals), 0, np.nan))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df.rename(columns={text_col: "text"})
    return df[["text", "label"]]


def load_kaggle_sentiment140(path=None):
    """Load Sentiment140 CSV. Looks in data/kaggle_sentiment140/ and project root."""
    path = path or config.KAGGLE_RAW_PATH
    if not os.path.isfile(path):
        # Try project root (e.g. if user put CSV in Sentiment Analysis folder)
        root_path = os.path.join(config.BASE_DIR, "training.1600000.processed.noemoticon.csv")
        if os.path.isfile(root_path):
            path = root_path
    if not os.path.isfile(path):
        # Try alternate path names in same directory
        parent = os.path.dirname(path)
        if os.path.isdir(parent):
            for f in os.listdir(parent):
                if f.endswith(".csv") and "1600000" in f:
                    path = os.path.join(parent, f)
                    break
    if not os.path.isfile(path):
        # Last try: project root with any 1600000 CSV
        for f in os.listdir(config.BASE_DIR):
            if f.endswith(".csv") and "1600000" in f:
                path = os.path.join(config.BASE_DIR, f)
                break
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Kaggle Sentiment140 not found. "
            "Place training.1600000.processed.noemoticon.csv in this project folder "
            "or in data/kaggle_sentiment140/"
        )
    df = pd.read_csv(path, encoding="latin-1", header=None, names=S140_COLUMNS)
    return df


def prepare_training_data(df, text_col="text", target_col="target", subsample=None, filter_keywords=None):
    """Preprocess and optionally filter by domain keywords (peerj-paper style)."""
    import re
    from preprocess import clean_for_nlp
    df = df[[text_col, target_col]].copy().dropna()
    if filter_keywords:
        pat = "|".join(re.escape(k) for k in filter_keywords)
        mask = df[text_col].astype(str).str.contains(pat, case=False, na=False)
        df = df[mask]
    if subsample and len(df) > subsample:
        df = df.sample(n=subsample, random_state=42)
    df["label"] = (df[target_col] == 4).astype(int)
    texts = [clean_for_nlp(t) for t in df[text_col].tolist()]
    labels = df["label"].values
    return texts, labels


def train_and_save(
    data_path=None,
    subsample=100_000,
    filter_keywords=None,
    save_model_path=None,
    save_vectorizer_path=None,
):
    """Train TF-IDF + Logistic Regression and save model and vectorizer."""
    import re
    save_model_path = save_model_path or config.SENTIMENT_MODEL_PATH
    save_vectorizer_path = save_vectorizer_path or config.VECTORIZER_PATH
    use_alt = getattr(config, "KAGGLE_DATASET", "sentiment140") == "alternative"
    if use_alt:
        df = load_alternative_dataset()
        if subsample and len(df) > subsample:
            df = df.sample(n=subsample, random_state=42)
        texts = [clean_for_nlp(t) for t in df["text"].tolist()]
        labels = df["label"].values
    else:
        data_path = data_path or config.KAGGLE_RAW_PATH
        filter_keywords = filter_keywords or [
            "milk", "school", "lunch", "whole milk", "nutrition", "kid", "child",
            "policy", "diet", "healthy", "food", "drink",
        ]
        df = load_kaggle_sentiment140(data_path)
        texts, labels = prepare_training_data(
            df, subsample=subsample, filter_keywords=filter_keywords
        )
        if len(texts) < 2000:
            df = load_kaggle_sentiment140(data_path)
            texts, labels = prepare_training_data(df, subsample=subsample, filter_keywords=None)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print("Sentiment model accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    joblib.dump(model, save_model_path)
    joblib.dump(vectorizer, save_vectorizer_path)
    print(f"Saved model to {save_model_path}, vectorizer to {save_vectorizer_path}")
    return model, vectorizer


if __name__ == "__main__":
    train_and_save(subsample=80_000)
