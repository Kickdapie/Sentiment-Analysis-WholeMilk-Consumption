"""
Compare current BERT sentiment predictions with a second model (VADER).

Inputs:
  - data/scraped_clean.csv (preferred; should contain BERT outputs)
  - data/scraped_raw.csv (fallback if clean file missing)

Outputs:
  - output/model_comparison_predictions.csv
  - output/model_comparison_report.txt
"""

import os
import pandas as pd

import config


OUT_PRED_PATH = os.path.join(config.OUTPUT_DIR, "model_comparison_predictions.csv")
OUT_REPORT_PATH = os.path.join(config.OUTPUT_DIR, "model_comparison_report.txt")


def vader_label_from_compound(compound: float, pos_thr: float = 0.05, neg_thr: float = -0.05) -> str:
    """Map VADER compound score to 3 classes."""
    if compound >= pos_thr:
        return "positive"
    if compound <= neg_thr:
        return "negative"
    return "neutral"


def load_base_dataframe() -> pd.DataFrame:
    """Load clean dataframe if available; fallback to raw and run BERT inference."""
    if os.path.isfile(config.SCRAPED_CLEAN_PATH):
        df = pd.read_csv(config.SCRAPED_CLEAN_PATH)
        if "text" not in df.columns:
            raise ValueError("Expected column `text` in scraped_clean.csv")
        if "sentiment_label" not in df.columns:
            from predict_sentiment_bert import predict_dataframe_bert
            df = predict_dataframe_bert(df, text_col="text")
        return df

    if os.path.isfile(config.SCRAPED_RAW_PATH):
        from predict_sentiment_bert import predict_dataframe_bert
        df = pd.read_csv(config.SCRAPED_RAW_PATH)
        if "text" not in df.columns:
            raise ValueError("Expected column `text` in scraped_raw.csv")
        df = predict_dataframe_bert(df, text_col="text")
        return df

    raise FileNotFoundError("Could not find scraped data. Run scraper/analysis first.")


def add_vader_predictions(df: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    texts = df["text"].fillna("").astype(str)
    compounds = texts.apply(lambda t: analyzer.polarity_scores(t)["compound"])

    out = df.copy()
    out["bert_label"] = out["sentiment_label"].fillna("neutral").astype(str).str.lower().str.strip()
    out["bert_score"] = out.get("sentiment_score", pd.Series([None] * len(out)))
    out["vader_compound"] = compounds
    out["vader_label"] = compounds.apply(vader_label_from_compound)
    out["model_agree"] = out["bert_label"] == out["vader_label"]
    return out


def counts_as_pct(series: pd.Series) -> pd.DataFrame:
    c = series.value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    total = c.sum()
    p = (100.0 * c / total) if total else c
    return pd.DataFrame({"count": c, "pct": p})


def write_report(df_cmp: pd.DataFrame) -> None:
    total = len(df_cmp)
    agree = int(df_cmp["model_agree"].sum())
    agree_pct = (100.0 * agree / total) if total else 0.0

    bert_dist = counts_as_pct(df_cmp["bert_label"])
    vader_dist = counts_as_pct(df_cmp["vader_label"])
    confusion = pd.crosstab(df_cmp["bert_label"], df_cmp["vader_label"], dropna=False)

    disagreements = df_cmp.loc[~df_cmp["model_agree"], ["text", "bert_label", "vader_label", "vader_compound"]].head(15)

    with open(OUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("SENTIMENT MODEL COMPARISON REPORT\n")
        f.write("Models: BERT (cardiffnlp/twitter-roberta-base-sentiment-latest) vs VADER\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Total documents: {total}\n")
        f.write(f"Agreement: {agree}/{total} ({agree_pct:.1f}%)\n\n")

        f.write("BERT distribution:\n")
        for label in ["positive", "neutral", "negative"]:
            row = bert_dist.loc[label]
            f.write(f"  {label:8s} {int(row['count']):4d} ({row['pct']:.1f}%)\n")
        f.write("\n")

        f.write("VADER distribution:\n")
        for label in ["positive", "neutral", "negative"]:
            row = vader_dist.loc[label]
            f.write(f"  {label:8s} {int(row['count']):4d} ({row['pct']:.1f}%)\n")
        f.write("\n")

        f.write("Confusion matrix (rows=BERT, cols=VADER):\n")
        f.write(confusion.to_string())
        f.write("\n\n")

        f.write("Sample disagreements (first 15):\n")
        if disagreements.empty:
            f.write("  None.\n")
        else:
            for i, row in disagreements.iterrows():
                snippet = str(row["text"]).replace("\n", " ").strip()
                if len(snippet) > 180:
                    snippet = snippet[:177] + "..."
                f.write(
                    f"- #{i}: BERT={row['bert_label']}, VADER={row['vader_label']}, "
                    f"compound={row['vader_compound']:.3f} | {snippet}\n"
                )

    print(f"Saved: {OUT_REPORT_PATH}")


def main():
    df = load_base_dataframe()
    if "text" not in df.columns:
        raise ValueError("Expected `text` column in dataframe.")
    if "sentiment_label" not in df.columns:
        raise ValueError("Expected `sentiment_label` column from BERT predictions.")

    df_cmp = add_vader_predictions(df)
    df_cmp.to_csv(OUT_PRED_PATH, index=False)
    print(f"Saved: {OUT_PRED_PATH}")

    write_report(df_cmp)


if __name__ == "__main__":
    main()
