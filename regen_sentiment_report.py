import os
import pandas as pd
import config


def main():
    df = pd.read_csv(config.SCRAPED_CLEAN_PATH)
    report_path = os.path.join(config.OUTPUT_DIR, "sentiment_report.txt")

    counts = df["sentiment_label"].value_counts()
    n_pos = int(counts.get("positive", 0))
    n_neg = int(counts.get("negative", 0))
    n_neu = int(counts.get("neutral", 0))
    total = len(df)
    pct_pos = 100 * n_pos / total if total else 0
    pct_neg = 100 * n_neg / total if total else 0
    pct_neu = 100 * n_neu / total if total else 0
    mean_score = float(df["sentiment_score"].mean()) if "sentiment_score" in df.columns else 0.0

    source_col = "source" if "source" in df.columns else None
    by_source = []
    if source_col:
        for src in df[source_col].dropna().unique():
            by_source.append((src, df[df[source_col] == src]))

    name_map = {
        "twitter": "Twitter",
        "news_rss": "News (RSS)",
        "reddit": "Reddit Post",
        "reddit_comment": "Reddit Comment",
        "bluesky": "Bluesky",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("SENTIMENT ANALYSIS REPORT\n")
        f.write("Topic: Whole Milk Consumption School Policy\n")
        f.write(f"Model: BERT ({getattr(config, 'BERT_MODEL_NAME', 'cardiffnlp/twitter-roberta-base-sentiment-latest')})\n")
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
                s_pos = int((sub["sentiment_label"] == "positive").sum())
                s_neg = int((sub["sentiment_label"] == "negative").sum())
                s_neu = int((sub["sentiment_label"] == "neutral").sum())
                src_name = name_map.get(src, src)
                f.write(
                    f"  {src_name}: {s_total} items — Positive {s_pos} ({100*s_pos/s_total:.1f}%), "
                    f"Negative {s_neg} ({100*s_neg/s_total:.1f}%), Neutral {s_neu} ({100*s_neu/s_total:.1f}%)\n"
                )
            f.write("\n")

        f.write("Conclusion: ")
        if pct_pos > pct_neg and pct_pos > pct_neu:
            f.write("Overall sentiment toward whole milk school policy in the collected content is predominantly positive.\n")
        elif pct_neg > pct_pos and pct_neg > pct_neu:
            f.write("Overall sentiment toward whole milk school policy in the collected content is predominantly negative.\n")
        else:
            f.write("Overall sentiment is mixed or neutral across the collected content.\n")
        f.write("\nOutputs: sentiment_summary.png, sentiment_by_source.png, keyword_network.png, keywords.csv.\n")

    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
