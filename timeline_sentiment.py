"""
Create monthly timeline for:
1) Post volume
2) Average sentiment score over time

Outputs:
- outputs/monthly_timeline.csv
- figures/volume_and_avg_sentiment_timeline.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser as dateparser


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "scraped_clean.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def safe_parse_date(val):
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    if not s:
        return pd.NaT
    try:
        dt = dateparser.parse(s, fuzzy=True)
        if dt is not None and dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:
        return pd.NaT


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "published" not in df.columns:
        raise ValueError("Expected 'published' column in scraped_clean.csv")
    if "sentiment_score" not in df.columns:
        raise ValueError("Expected 'sentiment_score' column in scraped_clean.csv")

    df["date"] = df["published"].apply(safe_parse_date)
    df = df[df["date"].notna()].copy()
    if df.empty:
        raise ValueError("No valid dates found in dataset.")

    df["month"] = df["date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("month", as_index=False)
        .agg(
            post_volume=("text", "size"),
            avg_sentiment_score=("sentiment_score", "mean"),
        )
        .sort_values("month")
    )
    monthly["avg_sentiment_score"] = monthly["avg_sentiment_score"].round(4)

    out_csv = os.path.join(OUT_DIR, "monthly_timeline.csv")
    monthly.to_csv(out_csv, index=False)

    x = range(len(monthly))
    fig, ax1 = plt.subplots(figsize=(13, 5))

    bars = ax1.bar(x, monthly["post_volume"], color="#4C78A8", alpha=0.85, label="Post volume")
    ax1.set_ylabel("Post volume", color="#1f3d5a")
    ax1.tick_params(axis="y", labelcolor="#1f3d5a")

    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        monthly["avg_sentiment_score"],
        color="#E45756",
        linewidth=2.2,
        marker="o",
        markersize=3,
        label="Average sentiment score",
    )
    ax2.set_ylabel("Average sentiment score (0=negative, 1=positive)", color="#8b1e1e")
    ax2.tick_params(axis="y", labelcolor="#8b1e1e")
    ax2.set_ylim(0, 1)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(monthly["month"], rotation=45, ha="right", fontsize=8)
    ax1.set_xlabel("Month")
    ax1.set_title("Monthly Post Volume and Average Sentiment Score")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "volume_and_avg_sentiment_timeline.png")
    plt.savefig(out_fig, dpi=170, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
