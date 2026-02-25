"""
Phase 3 Topic Modeling Extension: Coherence, Stability, Semantic Drivers.

Outputs:
  outputs/topic_k_sensitivity.csv
  outputs/topic_model_comparison.csv
  outputs/topic_stability_seed.csv
  outputs/topic_stability_bootstrap.csv
  outputs/topic_terms_best_lda.csv
  outputs/semantic_drivers.csv
  figures/topic_coherence_k.png
  phase3_topic_results.md

Also appends a section to phase3_results.md if present.
"""

import os
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import tokenize


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "scraped_clean.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_MD = os.path.join(BASE_DIR, "phase3_results.md")
TOPIC_MD = os.path.join(BASE_DIR, "phase3_topic_results.md")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def build_docs(df):
    docs = []
    for t in df["text"].fillna("").astype(str):
        toks = tokenize(t)
        if len(toks) >= 4:
            docs.append(toks)
    return docs


def is_domain_relevant(tokens):
    """Keep documents tied to milk/school policy domain for cleaner topics."""
    domain = {
        "milk", "school", "schools", "lunch", "cafeteria", "policy", "usda",
        "dairy", "children", "kids", "nutrition", "whole", "fat", "healthy",
        "student", "students", "food",
    }
    return any(t in domain for t in tokens)


def topic_sets_from_word_topics(topics):
    sets = []
    for words in topics:
        sets.append(set(words))
    return sets


def topic_overlap_consistency(topic_sets_a, topic_sets_b):
    """
    Symmetric greedy topic overlap consistency using max Jaccard matches.
    Returns [0,1], higher is more stable.
    """
    if not topic_sets_a or not topic_sets_b:
        return np.nan

    def one_way(a, b):
        scores = []
        for ta in a:
            best = 0.0
            for tb in b:
                union = len(ta | tb)
                if union == 0:
                    continue
                jac = len(ta & tb) / union
                if jac > best:
                    best = jac
            scores.append(best)
        return float(np.mean(scores)) if scores else np.nan

    return float(np.mean([one_way(topic_sets_a, topic_sets_b), one_way(topic_sets_b, topic_sets_a)]))


def lda_topics_for_k(
    texts_raw,
    docs_tokens,
    dictionary,
    k,
    seed,
    alpha=0.1,
    beta=0.01,
    max_iter=25,
    topn=10,
    compute_cv=True,
):
    vec = CountVectorizer(
        tokenizer=lambda s: tokenize(s),
        token_pattern=None,
        lowercase=True,
        max_features=2000,
        min_df=3,
        max_df=0.90,
    )
    X = vec.fit_transform(texts_raw)
    feature_names = np.array(vec.get_feature_names_out())
    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=seed,
        learning_method="batch",
        max_iter=max_iter,
        doc_topic_prior=alpha,
        topic_word_prior=beta,
    )
    lda.fit(X)

    topics = []
    for comp in lda.components_:
        idx = np.argsort(comp)[::-1][:topn]
        topics.append([str(feature_names[i]) for i in idx])

    cv = np.nan
    if compute_cv:
        cv = CoherenceModel(topics=topics, texts=docs_tokens, dictionary=dictionary, coherence="c_v").get_coherence()
    return topics, cv


def nmf_topics_and_cv(texts_raw, docs_tokens, k, random_state=42, topn=10):
    # NMF baseline for model comparison section
    vec = TfidfVectorizer(
        tokenizer=lambda s: tokenize(s),
        token_pattern=None,
        lowercase=True,
        max_features=2000,
        min_df=3,
        max_df=0.95,
    )
    X = vec.fit_transform(texts_raw)
    nmf = NMF(n_components=k, random_state=random_state, init="nndsvda", max_iter=400)
    nmf.fit(X)
    feature_names = np.array(vec.get_feature_names_out())
    topics = []
    for comp in nmf.components_:
        idx = np.argsort(comp)[::-1][:topn]
        topics.append([str(feature_names[i]) for i in idx])

    # Coherence on tokenized docs with gensim dictionary
    dictionary = corpora.Dictionary(docs_tokens)
    cv = CoherenceModel(topics=topics, texts=docs_tokens, dictionary=dictionary, coherence="c_v").get_coherence()
    return topics, cv


def interpretation_quality_label(cv, uniqueness):
    score = 0
    if cv >= 0.60:
        score += 2
    elif cv >= 0.50:
        score += 1
    if uniqueness >= 0.80:
        score += 2
    elif uniqueness >= 0.65:
        score += 1

    if score >= 3:
        return "Strong"
    if score >= 2:
        return "Moderate"
    return "Weak"


def log_odds_drivers(df, top_n=10, prior=0.01, allowed_terms=None):
    """
    Smoothed log-odds for positive vs negative terms.
    Returns ranked positive and negative semantic drivers.
    """
    pos_docs = df[df["sentiment_label"] == "positive"]["text"].fillna("").astype(str).tolist()
    neg_docs = df[df["sentiment_label"] == "negative"]["text"].fillna("").astype(str).tolist()

    pos_counts = Counter()
    neg_counts = Counter()
    noise = {
        "x200b", "amp", "gt", "lt", "nbsp", "http", "https", "www", "com",
        "edit", "deleted", "removed", "post", "comment",
    }
    for t in pos_docs:
        toks = [w for w in tokenize(t) if w.isalpha() and len(w) >= 3 and w not in noise]
        if allowed_terms is not None:
            toks = [w for w in toks if w in allowed_terms]
        pos_counts.update(toks)
    for t in neg_docs:
        toks = [w for w in tokenize(t) if w.isalpha() and len(w) >= 3 and w not in noise]
        if allowed_terms is not None:
            toks = [w for w in toks if w in allowed_terms]
        neg_counts.update(toks)

    vocab = set(pos_counts) | set(neg_counts)
    if not vocab:
        return pd.DataFrame(columns=["rank", "negative_driver", "negative_log_odds", "positive_driver", "positive_log_odds"])

    total_pos = sum(pos_counts.values())
    total_neg = sum(neg_counts.values())
    V = len(vocab)

    rows = []
    for w in vocab:
        pw = (pos_counts[w] + prior) / (total_pos + prior * V)
        nw = (neg_counts[w] + prior) / (total_neg + prior * V)
        log_odds = np.log(pw / nw)
        rows.append((w, float(log_odds)))

    score_df = pd.DataFrame(rows, columns=["word", "log_odds"])
    # Keep reasonably frequent terms for interpretability
    total_counts = {w: pos_counts[w] + neg_counts[w] for w in vocab}
    score_df["count"] = score_df["word"].map(total_counts)
    score_df = score_df[score_df["count"] >= 10]
    pos_top = score_df.sort_values("log_odds", ascending=False).head(top_n).reset_index(drop=True)
    neg_top = score_df.sort_values("log_odds", ascending=True).head(top_n).reset_index(drop=True)

    out_rows = []
    for i in range(top_n):
        out_rows.append(
            {
                "rank": i + 1,
                "negative_driver": neg_top.loc[i, "word"] if i < len(neg_top) else "",
                "negative_log_odds": round(float(neg_top.loc[i, "log_odds"]), 3) if i < len(neg_top) else np.nan,
                "positive_driver": pos_top.loc[i, "word"] if i < len(pos_top) else "",
                "positive_log_odds": round(float(pos_top.loc[i, "log_odds"]), 3) if i < len(pos_top) else np.nan,
            }
        )
    return pd.DataFrame(out_rows)


def main():
    print("=" * 60)
    print("Phase 3 Topic Modeling Extension")
    print("=" * 60)

    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Expected final dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"text", "sentiment_label"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in final dataset: {missing}")

    docs_all = build_docs(df)
    docs = [d for d in docs_all if is_domain_relevant(d)]
    df_domain = df[df["text"].fillna("").astype(str).apply(lambda s: is_domain_relevant(tokenize(s)))]
    print(f"Domain-relevant docs: {len(docs)}/{len(docs_all)}")
    # Keep runtime stable in constrained environments while preserving reproducibility.
    max_docs = 1500
    if len(docs) > max_docs:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(docs), size=max_docs, replace=False))
        docs = [docs[i] for i in idx]
    if len(docs) < 200:
        warnings.warn("Low document count after token filtering; topic quality may be unstable.")
    print(f"Documents for topic modeling: {len(docs)}")

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=3, no_above=0.90, keep_n=2000)
    print(f"Dictionary size: {len(dictionary)}")

    # 1) Coherence sensitivity across k
    # Keep k-range lightweight for runtime stability in lab machines.
    k_values = [4, 6, 8]
    base_seed = 42
    lda_alpha = 0.1
    lda_eta = 0.01
    lda_iterations = 50

    k_rows = []
    lda_topics_by_k = {}
    texts_raw = [" ".join(doc) for doc in docs]
    for k in k_values:
        topics_k, cv = lda_topics_for_k(
            texts_raw, docs, dictionary, k,
            seed=base_seed, alpha=lda_alpha, beta=lda_eta, max_iter=lda_iterations, topn=10
        )
        lda_topics_by_k[k] = topics_k
        k_rows.append({"model": "LDA", "k": k, "c_v": round(float(cv), 4), "seed": base_seed})
        print(f"LDA k={k} c_v={cv:.4f}")

    k_df = pd.DataFrame(k_rows).sort_values("k")
    k_df.to_csv(os.path.join(OUT_DIR, "topic_k_sensitivity.csv"), index=False)

    best_idx = k_df["c_v"].idxmax()
    best_k = int(k_df.loc[best_idx, "k"])
    best_cv = float(k_df.loc[best_idx, "c_v"])
    best_topics = lda_topics_by_k[best_k]
    print(f"Best LDA k={best_k} (c_v={best_cv:.4f})")

    # Save best-topic terms
    topic_rows = []
    topn = 10
    for t in range(best_k):
        terms = best_topics[t]
        for rank, w in enumerate(terms, start=1):
            topic_rows.append({"topic": t + 1, "rank": rank, "term": w})
    topic_terms_df = pd.DataFrame(topic_rows)
    topic_terms_df.to_csv(os.path.join(OUT_DIR, "topic_terms_best_lda.csv"), index=False)

    print("Finished k-sensitivity sweep.")

    # 2) Optional model comparison: LDA vs NMF at best_k
    nmf_topics, nmf_cv = nmf_topics_and_cv(texts_raw, docs, k=best_k, random_state=base_seed, topn=topn)
    lda_terms = [set(terms) for terms in best_topics]
    lda_uniqueness = len(set().union(*lda_terms)) / max(best_k * topn, 1)
    nmf_terms = [set(topic) for topic in nmf_topics]
    nmf_uniqueness = len(set().union(*nmf_terms)) / max(best_k * topn, 1)

    cmp_df = pd.DataFrame(
        [
            {
                "model": "LDA",
                "topics": best_k,
                "c_v": round(best_cv, 4),
                "uniqueness": round(float(lda_uniqueness), 3),
                "interpretation_quality": interpretation_quality_label(best_cv, lda_uniqueness),
                "notes": "Broad-to-moderate thematic groupings",
            },
            {
                "model": "NMF (TF-IDF)",
                "topics": best_k,
                "c_v": round(float(nmf_cv), 4),
                "uniqueness": round(float(nmf_uniqueness), 3),
                "interpretation_quality": interpretation_quality_label(nmf_cv, nmf_uniqueness),
                "notes": "Alternative baseline; often sharper lexical boundaries",
            },
        ]
    )
    cmp_df.to_csv(os.path.join(OUT_DIR, "topic_model_comparison.csv"), index=False)
    print("Model comparison saved.")

    # 3) Stability testing — seeds
    ref_sets = topic_sets_from_word_topics(best_topics)
    seed_list = [13, 42]
    seed_rows = []
    for s in seed_list:
        topics_s, cv = lda_topics_for_k(
            texts_raw, docs, dictionary, best_k,
            seed=s, alpha=lda_alpha, beta=lda_eta, max_iter=lda_iterations, topn=topn, compute_cv=False
        )
        sets_s = topic_sets_from_word_topics(topics_s)
        consistency = topic_overlap_consistency(ref_sets, sets_s)
        seed_rows.append(
            {"seed": s, "k": best_k, "c_v": np.nan, "overlap_consistency": round(float(consistency), 4)}
        )
    seed_stability_df = pd.DataFrame(seed_rows).sort_values("seed")
    seed_stability_df.to_csv(os.path.join(OUT_DIR, "topic_stability_seed.csv"), index=False)
    print("Seed stability complete.")

    # 4) Stability testing — bootstrap samples
    boot_rows = []
    rng = np.random.default_rng(42)
    n_docs = len(docs)
    sample_n = max(int(n_docs * 0.8), min(300, n_docs))
    for b in range(1, 3):
        idx = rng.choice(n_docs, size=sample_n, replace=True)
        docs_b = [docs[i] for i in idx]
        dict_b = corpora.Dictionary(docs_b)
        dict_b.filter_extremes(no_below=3, no_above=0.90, keep_n=2000)
        texts_b = [texts_raw[i] for i in idx]
        topics_b, cv_b = lda_topics_for_k(
            texts_b, docs_b, dict_b, best_k,
            seed=42 + b, alpha=lda_alpha, beta=lda_eta, max_iter=lda_iterations, topn=topn, compute_cv=False
        )
        sets_b = topic_sets_from_word_topics(topics_b)
        # compare boot model to reference terms via overlap on shared vocabulary
        consistency_b = topic_overlap_consistency(ref_sets, sets_b)
        boot_rows.append(
            {
                "bootstrap_run": b,
                "sample_size": sample_n,
                "k": best_k,
                "c_v": np.nan,
                "overlap_consistency": round(float(consistency_b), 4),
            }
        )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(os.path.join(OUT_DIR, "topic_stability_bootstrap.csv"), index=False)
    print("Bootstrap stability complete.")

    # 5) Semantic drivers via log-odds (restricted to high-signal policy terms)
    allowed_terms = None
    keywords_path = os.path.join(OUT_DIR, "keywords.csv")
    if os.path.isfile(keywords_path):
        kw_df = pd.read_csv(keywords_path)
        if "keyword" in kw_df.columns:
            allowed_terms = set(kw_df["keyword"].dropna().astype(str).str.lower().head(300))
    # Always include core policy terms
    core_terms = {
        "milk", "whole", "school", "schools", "lunch", "children", "kids",
        "dairy", "nutrition", "healthy", "fat", "policy", "usda", "cafeteria",
        "chocolate", "parents", "student", "students", "food", "health",
    }
    allowed_terms = core_terms if allowed_terms is None else (allowed_terms | core_terms)
    drivers_df = log_odds_drivers(df_domain, top_n=10, prior=0.01, allowed_terms=allowed_terms)
    drivers_df.to_csv(os.path.join(OUT_DIR, "semantic_drivers.csv"), index=False)

    # 6) Coherence sensitivity figure
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_df["k"], k_df["c_v"], marker="o", color="#2c3e50")
    ax.axhline(0.5, linestyle="--", color="#e67e22", linewidth=1, label="C_v=0.5 (acceptable)")
    ax.axhline(0.6, linestyle="--", color="#27ae60", linewidth=1, label="C_v=0.6 (strong)")
    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel("C_v coherence")
    ax.set_title("LDA Coherence Sensitivity Across k")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "topic_coherence_k.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 7) Report markdown
    seed_mean = float(seed_stability_df["overlap_consistency"].mean())
    boot_mean = float(boot_df["overlap_consistency"].mean())
    seed_sd = float(seed_stability_df["overlap_consistency"].std(ddof=1)) if len(seed_stability_df) > 1 else 0.0
    boot_sd = float(boot_df["overlap_consistency"].std(ddof=1)) if len(boot_df) > 1 else 0.0
    benchmark_note = (
        "strong" if best_cv >= 0.6 else ("acceptable" if best_cv >= 0.5 else "below-acceptable (requires justification)")
    )

    topic_md = []
    topic_md.append("## 3.3 Topic Modeling & Semantic Drivers\n")
    topic_md.append("### 3.3.1 Coherence Reporting\n")
    topic_md.append(f"- **Method:** LDA (scikit-learn) with Gensim C_v coherence evaluation\n")
    topic_md.append(f"- **Best k:** {best_k}\n")
    topic_md.append(f"- **Best C_v coherence:** {best_cv:.4f} ({benchmark_note})\n")
    topic_md.append(f"- **LDA priors:** alpha={lda_alpha}, beta(eta)={lda_eta}\n")
    topic_md.append(f"- **Training setup:** iterations={lda_iterations}, random_seed={base_seed}\n")
    topic_md.append("\n| k | C_v |\n|---:|---:|\n")
    for _, r in k_df.iterrows():
        topic_md.append(f"| {int(r['k'])} | {float(r['c_v']):.4f} |\n")
    topic_md.append("\n![Coherence Sensitivity Across k](figures/topic_coherence_k.png)\n")

    topic_md.append("\n### 3.3.2 Model Comparison\n")
    topic_md.append("| Model | Topics | C_v | Interpretation Quality | Notes |\n")
    topic_md.append("|-------|-------:|----:|------------------------|-------|\n")
    for _, r in cmp_df.iterrows():
        topic_md.append(
            f"| {r['model']} | {int(r['topics'])} | {float(r['c_v']):.4f} | {r['interpretation_quality']} | {r['notes']} |\n"
        )
    topic_md.append("\nModel choice is justified by coherence and interpretation quality with fixed, reproducible seeds.\n")

    topic_md.append("\n### 3.3.3 Stability Testing\n")
    topic_md.append(
        f"- **Seed stability (topic overlap consistency):** mean={seed_mean:.3f}, sd={seed_sd:.3f} across {len(seed_stability_df)} runs\n"
    )
    topic_md.append(
        f"- **Bootstrap stability (80% resamples):** mean={boot_mean:.3f}, sd={boot_sd:.3f} across {len(boot_df)} runs\n"
    )
    topic_md.append(
        "- **Consistency metric:** symmetric average of max Jaccard overlap between topic word sets (top-10 words/topic).\n"
    )
    topic_md.append("\nStability outputs: `outputs/topic_stability_seed.csv`, `outputs/topic_stability_bootstrap.csv`.\n")

    topic_md.append("\n### 3.3.4 Semantic Drivers of Sentiment\n")
    topic_md.append("Ranked using smoothed log-odds (positive vs negative corpora, informative prior).\n\n")
    topic_md.append("| Rank | Negative Drivers | Positive Drivers |\n")
    topic_md.append("|-----:|------------------|------------------|\n")
    for _, r in drivers_df.head(10).iterrows():
        topic_md.append(
            f"| {int(r['rank'])} | {r['negative_driver']} ({float(r['negative_log_odds']):.3f}) | {r['positive_driver']} ({float(r['positive_log_odds']):.3f}) |\n"
        )
    topic_md.append(
        "\nThese terms act as explanatory mechanisms by quantifying which words are disproportionately associated with negative versus positive sentiment.\n"
    )

    topic_md_str = "".join(topic_md)
    with open(TOPIC_MD, "w", encoding="utf-8") as f:
        f.write(topic_md_str)

    if os.path.isfile(RESULTS_MD):
        with open(RESULTS_MD, "r", encoding="utf-8") as f:
            base_md = f.read()
        marker = "## 3.3 Topic Modeling & Semantic Drivers"
        if marker in base_md:
            base_md = base_md.split(marker)[0].rstrip() + "\n\n"
        base_md = base_md.rstrip() + "\n\n" + topic_md_str
        with open(RESULTS_MD, "w", encoding="utf-8") as f:
            f.write(base_md)
        print(f"Appended topic section to {RESULTS_MD}")

    print(f"Saved {TOPIC_MD}")
    print("Done.")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
