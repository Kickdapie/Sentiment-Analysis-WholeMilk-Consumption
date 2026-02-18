"""
End-to-end pipeline: scrape -> train (if needed) -> predict -> network analysis -> final report.
Whole milk consumption school policy sentiment analysis (methodology from PeerJ CS 1149).
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
from scraper import run_scraper
from network_analysis import run_network_analysis


def run_pipeline(
    skip_scrape=False,
    skip_train=False,
    scrape_sources=None,
):
    """
    Run full pipeline:
    1. Scrape data (news + reddit)
    2. Train sentiment model if missing (requires Kaggle dataset)
    3. Predict sentiment on scraped data
    4. Network analysis (keywords + co-occurrence map)
    5. Generate final sentiment report and charts
    """
    scrape_sources = scrape_sources or getattr(config, "SCRAPE_SOURCES", ["news_rss", "reddit"])

    # ----- 1. Scrape -----
    if not skip_scrape:
        print("Step 1: Scraping content...")
        run_scraper(sources=scrape_sources)
    if not os.path.isfile(config.SCRAPED_RAW_PATH):
        print("No scraped data. Run scraper first or use sample.")
        run_scraper(sources=scrape_sources)
    df = pd.read_csv(config.SCRAPED_RAW_PATH)
    print(f"Loaded {len(df)} scraped items.")

    # ----- 2. Model setup -----
    use_bert = getattr(config, "SENTIMENT_MODEL_TYPE", "tfidf") == "bert"

    if use_bert:
        print("Step 2: Using pre-trained BERT model (no training required).")
    else:
        # Legacy TF-IDF path
        if not skip_train and (not os.path.isfile(config.SENTIMENT_MODEL_PATH) or not os.path.isfile(config.VECTORIZER_PATH)):
            print("Step 2: Training TF-IDF sentiment model...")
            try:
                from train_sentiment_model import train_and_save
                train_and_save(subsample=80_000)
            except FileNotFoundError as e:
                print(e)
                print("Continuing without trained model. Predictions will use fallback.")
        elif os.path.isfile(config.SENTIMENT_MODEL_PATH):
            print("Step 2: Using existing TF-IDF sentiment model.")

    # ----- 3. Predict sentiment -----
    print("Step 3: Predicting sentiment...")
    if use_bert:
        from predict_sentiment_bert import predict_dataframe_bert
        df = predict_dataframe_bert(df, text_col="text")
    else:
        try:
            from predict_sentiment import predict_dataframe
            df = predict_dataframe(df, text_col="text")
        except FileNotFoundError:
            # Fallback: VADER or neutral so report still runs
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                comp = [analyzer.polarity_scores(str(t))["compound"] for t in df["text"].fillna("")]
                df["sentiment_score"] = [(c + 1) / 2 for c in comp]
                df["sentiment_label"] = [
                    "positive" if s > 0.6 else ("negative" if s < 0.4 else "neutral")
                    for s in df["sentiment_score"]
                ]
            except Exception:
                df["sentiment_score"] = 0.5
                df["sentiment_label"] = "neutral"
    df.to_csv(config.SCRAPED_CLEAN_PATH, index=False)
    print(f"Saved scored data to {config.SCRAPED_CLEAN_PATH}")

    # ----- 4. Network analysis -----
    print("Step 4: Keyword & network analysis...")
    run_network_analysis(df, text_col="text", label_col="sentiment_label")

    # ----- 5. Final report -----
    print("Step 5: Generating final sentiment report...")
    report_path = os.path.join(config.OUTPUT_DIR, "sentiment_report.txt")
    summary_path = os.path.join(config.OUTPUT_DIR, "sentiment_summary.png")

    counts = df["sentiment_label"].value_counts()
    n_pos = counts.get("positive", 0)
    n_neg = counts.get("negative", 0)
    n_neu = counts.get("neutral", 0)
    total = len(df)
    pct_pos = 100 * n_pos / total if total else 0
    pct_neg = 100 * n_neg / total if total else 0
    pct_neu = 100 * n_neu / total if total else 0
    mean_score = float(df["sentiment_score"].mean())

    # By-source breakdown (includes Twitter when present)
    source_col = "source" if "source" in df.columns else None
    by_source = []
    if source_col:
        for src in df[source_col].dropna().unique():
            sub = df[df[source_col] == src]
            by_source.append((src, sub))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("SENTIMENT ANALYSIS REPORT\n")
        f.write("Topic: Whole Milk Consumption School Policy\n")
        if use_bert:
            f.write(f"Model: BERT ({getattr(config, 'BERT_MODEL_NAME', 'cardiffnlp/twitter-roberta-base-sentiment-latest')})\n")
        else:
            f.write("Model: TF-IDF + Logistic Regression\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total documents analyzed: {total}\n\n")
        f.write("Overall sentiment distribution:\n")
        f.write(f"  Positive: {n_pos} ({pct_pos:.1f}%)\n")
        f.write(f"  Negative: {n_neg} ({pct_neg:.1f}%)\n")
        f.write(f"  Neutral:  {n_neu} ({pct_neu:.1f}%)\n\n")
        f.write(f"Mean sentiment score (0=neg, 1=pos): {mean_score:.3f}\n\n")
        if by_source:
            f.write("Sentiment by source:\n")
            for src, sub in by_source:
                s_total = len(sub)
                s_pos = (sub["sentiment_label"] == "positive").sum()
                s_neg = (sub["sentiment_label"] == "negative").sum()
                s_neu = (sub["sentiment_label"] == "neutral").sum()
                src_name = {"twitter": "Twitter", "news_rss": "News (RSS)", "reddit": "Reddit", "bluesky": "Bluesky"}.get(src, src)
                f.write(f"  {src_name}: {s_total} items — Positive {s_pos} ({100*s_pos/s_total:.1f}%), Negative {s_neg} ({100*s_neg/s_total:.1f}%), Neutral {s_neu} ({100*s_neu/s_total:.1f}%)\n")
            f.write("\n")
        f.write("Conclusion: ")
        if pct_pos > pct_neg and pct_pos > pct_neu:
            f.write("Overall sentiment toward whole milk school policy in the collected content is predominantly positive.\n")
        elif pct_neg > pct_pos and pct_neg > pct_neu:
            f.write("Overall sentiment toward whole milk school policy in the collected content is predominantly negative.\n")
        else:
            f.write("Overall sentiment is mixed or neutral across the collected content.\n")
        f.write("\nOutputs: sentiment_summary.png, sentiment_by_source.png, keyword_network.png, keywords.csv.\n")
    print(f"Report saved to {report_path}")

    # Overall summary bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Positive", "Negative", "Neutral"], [n_pos, n_neg, n_neu], color=["#2ecc71", "#e74c3c", "#95a5a6"])
    ax.set_ylabel("Count")
    ax.set_title("Sentiment distribution: Whole milk school policy")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary chart saved to {summary_path}")

    # Sentiment by source chart (News, Reddit, Twitter when present)
    if by_source:
        source_names = []
        pos_counts, neg_counts, neu_counts = [], [], []
        for src, sub in by_source:
            src_name = {"twitter": "Twitter", "news_rss": "News (RSS)", "reddit": "Reddit", "bluesky": "Bluesky"}.get(src, src)
            source_names.append(src_name)
            pos_counts.append((sub["sentiment_label"] == "positive").sum())
            neg_counts.append((sub["sentiment_label"] == "negative").sum())
            neu_counts.append((sub["sentiment_label"] == "neutral").sum())
        x = range(len(source_names))
        w = 0.25
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar([i - w for i in x], pos_counts, w, label="Positive", color="#2ecc71")
        ax2.bar(x, neg_counts, w, label="Negative", color="#e74c3c")
        ax2.bar([i + w for i in x], neu_counts, w, label="Neutral", color="#95a5a6")
        ax2.set_xticks(x)
        ax2.set_xticklabels(source_names)
        ax2.set_ylabel("Count")
        ax2.set_title("Sentiment by source: Whole milk school policy")
        ax2.legend()
        plt.tight_layout()
        by_source_path = os.path.join(config.OUTPUT_DIR, "sentiment_by_source.png")
        plt.savefig(by_source_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"By-source chart saved to {by_source_path}")

    print("\nPipeline complete.")
    return df


if __name__ == "__main__":
    skip_scrape = "--skip-scrape" in sys.argv
    skip_train = "--skip-train" in sys.argv
    run_pipeline(skip_scrape=skip_scrape, skip_train=skip_train)
