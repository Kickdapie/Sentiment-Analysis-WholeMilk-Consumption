"""
Hypothesis + Bias + Sensitivity analysis for Whole Milk policy sentiment.

H0: No difference in sentiment score before vs after policy announcement.
H1: Significant difference exists.

Outputs:
  outputs/normality_results.csv
  outputs/hypothesis_test_results.csv
  outputs/sensitivity_top5pct.csv
  outputs/bias_diagnostics.csv
  figures/pre_post_sentiment_boxplot.png
  phase3_hypothesis_bias.md
"""

import os
import re
from urllib.parse import urlparse
from dateutil import parser as dateparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "scraped_clean.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(BASE_DIR, "figures")
REPORT_PATH = os.path.join(BASE_DIR, "phase3_hypothesis_bias.md")
MAIN_REPORT_PATH = os.path.join(BASE_DIR, "phase3_results.md")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Adjust if your advisor wants a different event date
POLICY_ANNOUNCEMENT_DATE = pd.Timestamp("2026-01-14")
RNG_SEED = 42


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
        return pd.Timestamp(dt)
    except Exception:
        return pd.NaT


def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((y.mean() - x.mean()) / pooled)  # after - before


def bootstrap_ci_mean_diff(before, after, n_boot=4000, alpha=0.05, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        b = rng.choice(before, size=len(before), replace=True)
        a = rng.choice(after, size=len(after), replace=True)
        diffs[i] = a.mean() - b.mean()
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return lo, hi


def run_hypothesis_test(before, after):
    # Normality: D'Agostino K^2 works well with larger n
    nb = stats.normaltest(before) if len(before) >= 8 else None
    na = stats.normaltest(after) if len(after) >= 8 else None

    normal_before = bool(nb is not None and nb.pvalue > 0.05)
    normal_after = bool(na is not None and na.pvalue > 0.05)
    use_t = normal_before and normal_after

    if use_t:
        test = stats.ttest_ind(before, after, equal_var=False, alternative="two-sided")
        test_name = "Welch t-test"
        stat_name = "t_stat"
        stat_value = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        test = stats.mannwhitneyu(before, after, alternative="two-sided")
        test_name = "Mann-Whitney U"
        stat_name = "u_stat"
        stat_value = float(test.statistic)
        p_value = float(test.pvalue)

    d = cohen_d(before, after)
    ci_lo, ci_hi = bootstrap_ci_mean_diff(before, after)
    mean_diff = float(np.mean(after) - np.mean(before))

    normality = pd.DataFrame(
        [
            {
                "group": "before",
                "n": len(before),
                "normality_test": "D'Agostino K^2",
                "statistic": float(nb.statistic) if nb else np.nan,
                "p_value": float(nb.pvalue) if nb else np.nan,
                "normal_at_0_05": normal_before,
            },
            {
                "group": "after",
                "n": len(after),
                "normality_test": "D'Agostino K^2",
                "statistic": float(na.statistic) if na else np.nan,
                "p_value": float(na.pvalue) if na else np.nan,
                "normal_at_0_05": normal_after,
            },
        ]
    )

    result = {
        "test_used": test_name,
        stat_name: stat_value,
        "p_value": p_value,
        "cohen_d_after_minus_before": d,
        "mean_diff_after_minus_before": mean_diff,
        "ci95_mean_diff_low": ci_lo,
        "ci95_mean_diff_high": ci_hi,
        "before_mean": float(np.mean(before)),
        "after_mean": float(np.mean(after)),
        "before_n": int(len(before)),
        "after_n": int(len(after)),
    }
    return normality, pd.DataFrame([result])


def extract_entity_id(row):
    """Approx user/activity entity for sensitivity analysis."""
    src = str(row.get("source", "")).lower()
    url = str(row.get("url", "")).strip()
    if not url:
        return f"{src}:unknown"

    # Bluesky: https://bsky.app/profile/<handle>/post/<id>
    if "bsky.app/profile/" in url:
        m = re.search(r"bsky\.app/profile/([^/]+)/post/", url)
        if m:
            return f"bluesky:{m.group(1).lower()}"

    # Reddit: /user/<username>/ if available, else subreddit proxy
    if "reddit.com" in url:
        mu = re.search(r"reddit\.com/user/([^/]+)/", url)
        if mu:
            return f"reddit_user:{mu.group(1).lower()}"
        mr = re.search(r"reddit\.com/r/([^/]+)/", url)
        if mr:
            return f"reddit_subreddit:{mr.group(1).lower()}"
        return "reddit:unknown"

    # News: domain proxy
    try:
        domain = (urlparse(url).netloc or "").lower()
        if domain:
            return f"domain:{domain}"
    except Exception:
        pass
    return f"{src}:unknown"


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    needed = {"published", "sentiment_score", "source", "url", "text"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = df["published"].apply(safe_parse_date)
    df = df[df["date"].notna()].copy()
    df["period"] = np.where(df["date"] < POLICY_ANNOUNCEMENT_DATE, "before", "after")

    before = df.loc[df["period"] == "before", "sentiment_score"].astype(float).values
    after = df.loc[df["period"] == "after", "sentiment_score"].astype(float).values
    if len(before) < 10 or len(after) < 10:
        raise ValueError("Insufficient before/after samples for hypothesis testing.")

    normality_df, test_df = run_hypothesis_test(before, after)
    normality_df.to_csv(os.path.join(OUT_DIR, "normality_results.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "hypothesis_test_results.csv"), index=False)

    # Sensitivity analysis: exclude top 5% most active entities (approx proxy)
    df["entity_id"] = df.apply(extract_entity_id, axis=1)
    activity = df["entity_id"].value_counts()
    threshold = float(np.quantile(activity.values, 0.95))
    high_entities = set(activity[activity > threshold].index.tolist())
    df_sens = df[~df["entity_id"].isin(high_entities)].copy()

    b2 = df_sens.loc[df_sens["period"] == "before", "sentiment_score"].astype(float).values
    a2 = df_sens.loc[df_sens["period"] == "after", "sentiment_score"].astype(float).values
    _, sens_test_df = run_hypothesis_test(b2, a2)
    sens_test_df["entities_removed_count"] = len(high_entities)
    sens_test_df["rows_removed_count"] = int(len(df) - len(df_sens))
    sens_test_df["rows_removed_pct"] = round(100 * (len(df) - len(df_sens)) / len(df), 1)
    sens_test_df.to_csv(os.path.join(OUT_DIR, "sensitivity_top5pct.csv"), index=False)

    # Bias diagnostics
    source_share = (
        df["source"].value_counts(normalize=True).mul(100).round(1).rename_axis("source").reset_index(name="pct_of_posts")
    )
    source_share["metric"] = "source_share_pct"
    source_share["detail"] = source_share["source"]

    top10_entity_share = round(100 * activity.head(10).sum() / activity.sum(), 1)
    within_30d = ((df["date"] >= (POLICY_ANNOUNCEMENT_DATE - pd.Timedelta(days=30))) &
                  (df["date"] <= (POLICY_ANNOUNCEMENT_DATE + pd.Timedelta(days=30)))).mean() * 100

    # Geo proxy by top-level domain (very approximate)
    domains = df["url"].astype(str).apply(lambda u: (urlparse(u).netloc or "").lower())
    tld = domains.apply(lambda d: d.split(".")[-1] if "." in d else "unknown")
    tld_share = tld.value_counts(normalize=True).mul(100).round(1).head(8)

    bias_rows = []
    for _, r in source_share.iterrows():
        bias_rows.append(
            {"bias_dimension": "Platform/source imbalance", "metric": r["detail"], "value": f"{r['pct_of_posts']:.1f}%"}
        )
    bias_rows.append(
        {"bias_dimension": "High-activity concentration (proxy)", "metric": "Top-10 entities share", "value": f"{top10_entity_share:.1f}%"}
    )
    bias_rows.append(
        {"bias_dimension": "Time-window bias", "metric": "Posts within ±30 days of announcement", "value": f"{within_30d:.1f}%"}
    )
    for k, v in tld_share.items():
        bias_rows.append(
            {"bias_dimension": "Geographic proxy (domain TLD)", "metric": f"TLD .{k}", "value": f"{v:.1f}%"}
        )
    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(os.path.join(OUT_DIR, "bias_diagnostics.csv"), index=False)

    # Figure: pre vs post sentiment distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([before, after], labels=["Before", "After"], showmeans=True)
    ax.set_title("Sentiment Score Before vs After Policy Announcement")
    ax.set_ylabel("Sentiment score (0=negative, 1=positive)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pre_post_sentiment_boxplot.png"), dpi=170, bbox_inches="tight")
    plt.close()

    # Markdown report
    t = test_df.iloc[0]
    s = sens_test_df.iloc[0]
    stat_col = "t_stat" if "t_stat" in test_df.columns else "u_stat"
    s_stat_col = "t_stat" if "t_stat" in sens_test_df.columns else "u_stat"
    signif = "significant" if float(t["p_value"]) < 0.05 else "not significant"
    sens_signif = "significant" if float(s["p_value"]) < 0.05 else "not significant"

    lines = []
    lines.append("# Hypothesis Testing & Bias Assessment\n\n")
    lines.append("## Hypothesis\n")
    lines.append("- **H0:** No difference in sentiment scores before vs after policy announcement.\n")
    lines.append("- **H1:** A significant difference exists.\n\n")
    lines.append("## Main Test Results\n")
    lines.append(f"- Announcement date used: **{POLICY_ANNOUNCEMENT_DATE.date()}**\n")
    lines.append(f"- Test selected after normality check: **{t['test_used']}**\n")
    lines.append(f"- Test statistic: **{stat_col} = {float(t[stat_col]):.4f}**\n")
    lines.append(f"- p-value: **{float(t['p_value']):.6f}** ({signif})\n")
    lines.append(f"- Cohen's d (after-before): **{float(t['cohen_d_after_minus_before']):.3f}**\n")
    lines.append(
        f"- 95% CI of mean difference (after-before): **[{float(t['ci95_mean_diff_low']):.4f}, {float(t['ci95_mean_diff_high']):.4f}]**\n"
    )
    lines.append(
        f"- Means: before={float(t['before_mean']):.4f}, after={float(t['after_mean']):.4f}\n\n"
    )
    lines.append("## Sensitivity Analysis (Exclude top 5% most active entities, proxy)\n")
    lines.append(f"- Removed entities: **{int(s['entities_removed_count'])}**\n")
    lines.append(f"- Removed rows: **{int(s['rows_removed_count'])}** ({float(s['rows_removed_pct']):.1f}%)\n")
    lines.append(f"- Sensitivity test: **{s['test_used']}**\n")
    lines.append(f"- Statistic: **{s_stat_col} = {float(s[s_stat_col]):.4f}**\n")
    lines.append(f"- p-value: **{float(s['p_value']):.6f}** ({sens_signif})\n")
    lines.append(f"- Cohen's d: **{float(s['cohen_d_after_minus_before']):.3f}**\n")
    lines.append(
        f"- 95% CI mean diff: **[{float(s['ci95_mean_diff_low']):.4f}, {float(s['ci95_mean_diff_high']):.4f}]**\n\n"
    )
    lines.append("## Bias Considerations\n")
    lines.append("- Geographic imbalance estimated using domain-TLD proxy (approximate).\n")
    lines.append("- Platform algorithm bias acknowledged (ranking/recommendation effects).\n")
    lines.append("- Time-window bias quantified via ±30-day concentration.\n")
    lines.append("- High-activity over-representation addressed via top-5% entity exclusion sensitivity test.\n\n")
    lines.append("## Semantic Driver Transparency\n")
    lines.append("- Positive/negative drivers are from **smoothed log-odds** of token frequencies between positive vs negative corpora.\n")
    lines.append("- A positive log-odds means a term is relatively over-represented in positive posts; negative log-odds means over-represented in negative posts.\n")
    lines.append("- See: `outputs/semantic_drivers.csv`.\n\n")
    lines.append("![Before vs After Sentiment](figures/pre_post_sentiment_boxplot.png)\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    # Append section to main report if present
    if os.path.isfile(MAIN_REPORT_PATH):
        with open(MAIN_REPORT_PATH, "r", encoding="utf-8") as f:
            main_md = f.read().rstrip()
        section = "\n\n## 3.4 Hypothesis Testing, Bias, and Sensitivity\n\n"
        section += f"- Test used: **{t['test_used']}**; {stat_col}={float(t[stat_col]):.4f}, p={float(t['p_value']):.6f}\n"
        section += f"- Cohen's d: **{float(t['cohen_d_after_minus_before']):.3f}**; 95% CI mean diff: **[{float(t['ci95_mean_diff_low']):.4f}, {float(t['ci95_mean_diff_high']):.4f}]**\n"
        section += f"- Sensitivity (exclude top 5% active entities): p={float(s['p_value']):.6f}, d={float(s['cohen_d_after_minus_before']):.3f}\n"
        section += "- Bias checks reported in `outputs/bias_diagnostics.csv`.\n"
        with open(MAIN_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(main_md + section)

    print("Saved:")
    print(f"- {os.path.join(OUT_DIR, 'normality_results.csv')}")
    print(f"- {os.path.join(OUT_DIR, 'hypothesis_test_results.csv')}")
    print(f"- {os.path.join(OUT_DIR, 'sensitivity_top5pct.csv')}")
    print(f"- {os.path.join(OUT_DIR, 'bias_diagnostics.csv')}")
    print(f"- {os.path.join(FIG_DIR, 'pre_post_sentiment_boxplot.png')}")
    print(f"- {REPORT_PATH}")


if __name__ == "__main__":
    main()
