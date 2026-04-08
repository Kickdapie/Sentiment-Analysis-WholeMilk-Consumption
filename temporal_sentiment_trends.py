"""
Three-panel temporal sentiment analysis (whole milk / school milk policy corpus).

A) Monthly post volume (bars) + linear trend
B) Monthly sentiment mix — smoothed (rolling) so sparse months do not jump 0%/100%
C) Average sentiment (-1..+1) — smoothed mean + rolling band

Why raw monthly lines look "weird": with 1 post/month, % is always 100/0/0 and the mean is
exactly -1, 0, or +1. Rolling windows pool neighboring months for a readable trend.

Inputs:  data/scraped_clean.csv (published, sentiment_label)
Outputs: outputs/temporal_sentiment_monthly.csv (raw monthly; smoothed columns for plot)
         figures/temporal_sentiment_trend_analysis.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser as dateparser


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "scraped_clean.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(BASE_DIR, "figures")

# Drop very old rows often caused by dates embedded in post text (prefer ISO from Reddit).
MIN_ANALYSIS_DATE = "2015-01-01"
# Smooth B/C over this many months (centered rolling mean).
ROLLING_MONTHS = 3
# X-axis major ticks every N months (6 = twice per year; use 12 for yearly).
XTICK_INTERVAL_MONTHS = 6

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

LABEL_TO_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def safe_parse_date(val):
    """Prefer pandas ISO parse; avoid fuzzy parsing that can pick wrong years from body text."""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    if not s:
        return pd.NaT
    ts = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.notna(ts):
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        return t
    try:
        dt = dateparser.parse(s, fuzzy=False)
        if dt is not None and dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except Exception:
        return pd.NaT


def rolling_smooth_percentages(monthly: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling mean on neg/neu/pos % then renormalize rows to sum to 100."""
    cols = ["pct_negative", "pct_neutral", "pct_positive"]
    rolled = monthly[cols].rolling(window=window, center=True, min_periods=1).mean()
    s = rolled.sum(axis=1).replace(0, np.nan)
    rolled = rolled.div(s, axis=0) * 100.0
    return rolled


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Could not find {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "published" not in df.columns or "sentiment_label" not in df.columns:
        raise ValueError("Expected columns: published, sentiment_label")

    df["date"] = df["published"].apply(safe_parse_date)
    df = df[df["date"].notna()].copy()
    if df.empty:
        raise ValueError("No valid dates found.")

    min_ts = pd.Timestamp(MIN_ANALYSIS_DATE)
    df = df[df["date"] >= min_ts].copy()
    if df.empty:
        raise ValueError(f"No rows on or after {MIN_ANALYSIS_DATE}.")

    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["sentiment_label"] = df["sentiment_label"].fillna("").astype(str).str.lower().str.strip()
    df["score_neg1_1"] = df["sentiment_label"].map(LABEL_TO_SCORE)
    df = df[df["score_neg1_1"].notna()].copy()
    if df.empty:
        raise ValueError("No rows with valid sentiment labels.")

    months = sorted(df["month_start"].unique())

    rows = []
    for m in months:
        sub = df[df["month_start"] == m]
        n = len(sub)
        c_neg = (sub["sentiment_label"] == "negative").sum()
        c_neu = (sub["sentiment_label"] == "neutral").sum()
        c_pos = (sub["sentiment_label"] == "positive").sum()
        mean_s = float(sub["score_neg1_1"].mean())
        std_s = float(sub["score_neg1_1"].std(ddof=0)) if n > 1 else 0.0
        rows.append(
            {
                "month_start": m,
                "post_volume": n,
                "pct_negative": 100.0 * c_neg / n if n else 0.0,
                "pct_neutral": 100.0 * c_neu / n if n else 0.0,
                "pct_positive": 100.0 * c_pos / n if n else 0.0,
                "avg_sentiment_neg1_1": round(mean_s, 4),
                "std_sentiment_neg1_1": round(std_s, 4),
            }
        )

    monthly = pd.DataFrame(rows).sort_values("month_start").reset_index(drop=True)

    w = ROLLING_MONTHS
    pct_smooth = rolling_smooth_percentages(monthly, w)
    monthly["pct_negative_smooth"] = pct_smooth["pct_negative"].round(2)
    monthly["pct_neutral_smooth"] = pct_smooth["pct_neutral"].round(2)
    monthly["pct_positive_smooth"] = pct_smooth["pct_positive"].round(2)
    monthly["avg_sentiment_smooth"] = (
        monthly["avg_sentiment_neg1_1"].rolling(window=w, center=True, min_periods=1).mean().round(4)
    )
    monthly["std_sentiment_smooth"] = (
        monthly["std_sentiment_neg1_1"].rolling(window=w, center=True, min_periods=1).mean().round(4)
    )

    out_csv = os.path.join(OUT_DIR, "temporal_sentiment_monthly.csv")
    monthly.to_csv(out_csv, index=False)

    x_dt = pd.to_datetime(monthly["month_start"])
    vol = monthly["post_volume"].values.astype(float)
    x_idx = np.arange(len(monthly), dtype=float)

    if len(x_idx) >= 2:
        slope, intercept = np.polyfit(x_idx, vol, 1)
        trend = slope * x_idx + intercept
    else:
        trend = vol

    mean_plot = monthly["avg_sentiment_smooth"].values
    std_plot = monthly["std_sentiment_smooth"].values

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "Temporal Sentiment Trend Analysis — Whole Milk / School Milk Policy (Reddit)",
        fontsize=14,
        fontweight="bold",
    )

    ax_a = axes[0]
    ax_a.bar(x_dt, vol, width=22, color="#b0b0b0", label="Posts")
    ax_a.plot(x_dt, trend, color="#c0392b", linestyle="--", linewidth=1.8, label="Trend")
    ax_a.set_ylabel("Number of Posts")
    ax_a.set_title("A) Monthly Post Volume Over Time")
    ax_a.grid(axis="y", alpha=0.35)
    ax_a.legend(loc="upper right", fontsize=9)

    ax_b = axes[1]
    ax_b.plot(
        x_dt,
        monthly["pct_negative_smooth"],
        color="#e74c3c",
        marker="o",
        markersize=4,
        label="Negative",
    )
    ax_b.plot(
        x_dt,
        monthly["pct_neutral_smooth"],
        color="#f1c40f",
        marker="o",
        markersize=4,
        label="Neutral",
    )
    ax_b.plot(
        x_dt,
        monthly["pct_positive_smooth"],
        color="#27ae60",
        marker="o",
        markersize=4,
        label="Positive",
    )
    ax_b.set_ylabel("Percentage (%)")
    ax_b.set_ylim(0, 100)
    ax_b.set_title(f"B) Monthly Sentiment Distribution ({w}-month rolling average)")
    ax_b.grid(axis="y", alpha=0.35)
    ax_b.legend(loc="upper right", fontsize=9)

    ax_c = axes[2]
    ax_c.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1)
    ax_c.fill_between(
        x_dt,
        mean_plot - std_plot,
        mean_plot + std_plot,
        color="#3498db",
        alpha=0.22,
        label="± rolling mean (within-month std)",
    )
    ax_c.plot(
        x_dt,
        mean_plot,
        color="#2980b9",
        marker="o",
        markersize=5,
        linewidth=2,
        label="Mean sentiment (smoothed)",
    )
    ax_c.set_ylabel("Sentiment Score (-1 to +1)")
    ax_c.set_ylim(-1.05, 1.05)
    ax_c.set_title(f"C) Average Sentiment Score Over Time ({w}-month rolling mean)")
    ax_c.grid(axis="y", alpha=0.35)
    ax_c.legend(loc="upper right", fontsize=9)
    ax_c.set_xlabel("Time")

    locator = mdates.MonthLocator(interval=XTICK_INTERVAL_MONTHS)
    fmt = mdates.DateFormatter("%b %Y")
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(fmt)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_fig = os.path.join(FIG_DIR, "temporal_sentiment_trend_analysis.png")
    plt.savefig(out_fig, dpi=170, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_fig}")
    print(
        f"Note: B/C use {w}-month rolling averages (sparse months otherwise jump 0–100%). "
        f"Dates before {MIN_ANALYSIS_DATE} excluded. X ticks every {XTICK_INTERVAL_MONTHS} months."
    )


if __name__ == "__main__":
    main()
