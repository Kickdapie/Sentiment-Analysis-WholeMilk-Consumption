"""
BERT-based sentiment prediction using a pre-trained RoBERTa model
fine-tuned on ~124M tweets (cardiffnlp/twitter-roberta-base-sentiment-latest).

3 classes: negative (0), neutral (1), positive (2).
No fine-tuning needed; the model is used directly for inference.

For papers, cite:
  Loureiro et al., "TimeLMs: Diachronic Language Models from Twitter", 2022.
  https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import config

# Label mapping for the cardiffnlp model
BERT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def load_bert_pipeline():
    """Load the Hugging Face sentiment pipeline (downloads model on first run)."""
    from transformers import pipeline
    model_name = getattr(config, "BERT_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    print(f"Loading BERT model: {model_name} ...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        top_k=None,       # return all class scores
        truncation=True,
        max_length=512,
    )
    print("BERT model loaded.")
    return sentiment_pipe


def predict_texts_bert(texts, sentiment_pipe, batch_size=32):
    """
    Predict sentiment for a list of raw texts using BERT.
    Returns labels ('positive'/'negative'/'neutral') and scores (P(positive)).
    """
    labels = []
    scores = []
    # Process in batches for speed
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT prediction"):
        batch = [str(t)[:512] if t else "" for t in texts[i:i + batch_size]]
        results = sentiment_pipe(batch)
        for result in results:
            # result is a list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
            score_map = {r["label"].lower(): r["score"] for r in result}
            pos_score = score_map.get("positive", 0.0)
            neg_score = score_map.get("negative", 0.0)
            neu_score = score_map.get("neutral", 0.0)
            # Pick highest
            best = max(score_map, key=score_map.get)
            labels.append(best)
            scores.append(pos_score)  # sentiment_score = P(positive) for consistency
    return labels, scores


def predict_dataframe_bert(df, text_col="text"):
    """Add sentiment_label and sentiment_score columns using BERT."""
    sentiment_pipe = load_bert_pipeline()
    texts = df[text_col].fillna("").astype(str).tolist()
    labels, scores = predict_texts_bert(texts, sentiment_pipe)
    df = df.copy()
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    return df


if __name__ == "__main__":
    # Quick test
    pipe = load_bert_pipeline()
    test_texts = [
        "Schools should offer whole milk again. Kids need the nutrition.",
        "Banning whole milk in school lunches was a terrible mistake.",
        "The new school milk policy is just a political stunt.",
        "I'm glad they brought back whole milk for kids!",
    ]
    labels, scores = predict_texts_bert(test_texts, pipe)
    for text, label, score in zip(test_texts, labels, scores):
        print(f"[{label:>8}] (pos={score:.3f}) {text[:80]}")
