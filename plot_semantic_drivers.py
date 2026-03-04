"""
Create a presentation-style semantic keyword drivers figure.

Input:
  outputs/semantic_drivers.csv

Output:
  figures/semantic_drivers_keywords.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(BASE_DIR, "outputs", "semantic_drivers.csv")
OUT_DIR = os.path.join(BASE_DIR, "figures")
OUT_PATH = os.path.join(OUT_DIR, "semantic_drivers_keywords.png")

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    if not os.path.isfile(IN_PATH):
        raise FileNotFoundError(f"Could not find: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    required = {
        "negative_driver",
        "negative_log_odds",
        "positive_driver",
        "positive_log_odds",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"semantic_drivers.csv missing columns: {missing}")

    # Use top 10 rows as ranked in file
    top = df.head(10).copy()

    pos_words = top["positive_driver"].fillna("").astype(str).tolist()[::-1]
    pos_scores = top["positive_log_odds"].fillna(0).astype(float).tolist()[::-1]
    neg_words = top["negative_driver"].fillna("").astype(str).tolist()[::-1]
    neg_scores = top["negative_log_odds"].fillna(0).astype(float).tolist()[::-1]

    fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Semantic Drivers of Sentiment (Keyword Log-Odds)", fontsize=14, fontweight="bold")

    # Positive panel
    y = list(range(len(pos_words)))
    ax_pos.barh(y, pos_scores, color="#7BC67B", edgecolor="#4B9E4B", alpha=0.95)
    ax_pos.set_yticks(y)
    ax_pos.set_yticklabels(pos_words)
    ax_pos.set_title("Positive Sentiment Drivers", fontweight="bold")
    ax_pos.set_xlabel("Log-Odds Ratio")
    ax_pos.grid(axis="x", alpha=0.2)
    for i, v in enumerate(pos_scores):
        ax_pos.text(v + 0.01, y[i], f"{v:.2f}", va="center", fontsize=8, color="#2e5e2e")

    # Negative panel (keep negative axis)
    y2 = list(range(len(neg_words)))
    ax_neg.barh(y2, neg_scores, color="#F28C8C", edgecolor="#D55B5B", alpha=0.95)
    ax_neg.set_yticks(y2)
    ax_neg.set_yticklabels(neg_words)
    ax_neg.set_title("Negative Sentiment Drivers", fontweight="bold")
    ax_neg.set_xlabel("Log-Odds Ratio")
    ax_neg.grid(axis="x", alpha=0.2)
    for i, v in enumerate(neg_scores):
        ax_neg.text(v - 0.01, y2[i], f"{v:.2f}", va="center", ha="right", fontsize=8, color="#7a2a2a")

    # Symmetric x limits for cleaner comparison
    max_abs = max(max(abs(x) for x in pos_scores), max(abs(x) for x in neg_scores))
    lim = max(0.2, round(max_abs * 1.2, 2))
    ax_pos.set_xlim(0, lim)
    ax_neg.set_xlim(-lim, 0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
