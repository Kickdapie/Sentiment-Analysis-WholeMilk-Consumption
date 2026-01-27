"""
Keyword classification and network analysis for sentiment corpus.
- Extract and count keywords; build co-occurrence network.
- Optional: PMI (pointwise mutual information) for semantic orientation (PeerJ CS 1149).
- Visualize keyword network and sentiment-linked terms.
"""

import os
import math
import collections
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

import config
from preprocess import tokenize, tokenize_keep_stopwords


def get_word_freq(texts, min_freq=None):
    """Tokenize all texts and return word frequencies."""
    min_freq = min_freq or config.MIN_KEYWORD_FREQ
    counter = collections.Counter()
    for t in texts:
        counter.update(tokenize(t))
    return {w: c for w, c in counter.most_common() if c >= min_freq}


def cooccurrence_pairs(texts, window=config.CO_OCCURRENCE_WINDOW):
    """Build co-occurrence counts for pairs within a sliding window."""
    pairs = defaultdict(int)
    for text in texts:
        tokens = tokenize(text)
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + window + 1, len(tokens))):
                a, b = tokens[i], tokens[j]
                if a != b:
                    pairs[(min(a, b), max(a, b))] += 1
    return dict(pairs)


def build_pmi(texts, word_freq=None, vocab_size=None):
    """
    Compute PMI for word pairs (PeerJ CS 1149 style).
    P(t) = document frequency / D; P(t1,t2) = DF(t1,t2)/D.
    PMI(t1,t2) = log( P(t1,t2) / (P(t1)*P(t2)) ).
    """
    if word_freq is None:
        word_freq = get_word_freq(texts, min_freq=2)
    pairs = cooccurrence_pairs(texts)
    D = len(texts)
    # Document frequencies: count docs containing word
    doc_freq = defaultdict(int)
    for t in texts:
        seen = set()
        for w in tokenize(t):
            if w in word_freq and w not in seen:
                doc_freq[w] += 1
                seen.add(w)
    vocab_size = vocab_size or max(len(doc_freq), 1)
    P_t = {w: (doc_freq[w] / D) for w in doc_freq}
    P_t = {w: max(p, 1e-6) for w, p in P_t.items()}
    pmi = {}
    for (t1, t2), count in pairs.items():
        if t1 not in P_t or t2 not in P_t:
            continue
        p_joint = count / max(D, 1)
        if p_joint <= 0:
            continue
        pmi[(t1, t2)] = math.log2(p_joint / (P_t[t1] * P_t[t2] + 1e-10))
    return pmi, P_t, doc_freq


def semantic_orientation_scores(texts, labels, positive_seed=None, negative_seed=None):
    """
    SO(t) = sum(PMI(t, pos)) - sum(PMI(t, neg)) over seed words (PeerJ CS 1149 Eq 4).
    labels: 0 = negative, 1 = positive (per document).
    """
    positive_seed = positive_seed or {"good", "great", "love", "best", "healthy", "support", "better"}
    negative_seed = negative_seed or {"bad", "wrong", "against", "harm", "ban", "poor", "against"}
    pmi, P_t, _ = build_pmi(texts)
    # Build term -> SO from co-occurrence with seed words
    so = defaultdict(float)
    for (t1, t2), score in pmi.items():
        if t1 in positive_seed:
            so[t2] += score
        elif t1 in negative_seed:
            so[t2] -= score
        if t2 in positive_seed:
            so[t1] += score
        elif t2 in negative_seed:
            so[t1] -= score
    return dict(so)


def build_keyword_network(texts, top_n=config.TOP_N_KEYWORDS, min_freq=config.MIN_KEYWORD_FREQ):
    """Build NetworkX graph: nodes = keywords, edges = co-occurrence (weight = count or PMI)."""
    word_freq = get_word_freq(texts, min_freq=min_freq)
    top_words = set(w for w, _ in list(word_freq.items())[:top_n])
    pairs = cooccurrence_pairs(texts)
    G = nx.Graph()
    for w in top_words:
        G.add_node(w)
    for (a, b), count in pairs.items():
        if a in top_words and b in top_words and count >= 2:
            G.add_edge(a, b, weight=count)
    return G, word_freq


def plot_network(G, word_freq, sentiment_scores=None, output_path=None):
    """Draw keyword co-occurrence network; color nodes by sentiment if provided."""
    output_path = output_path or os.path.join(config.OUTPUT_DIR, "keyword_network.png")
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    node_sizes = [word_freq.get(n, 5) * 15 for n in G.nodes()]
    if sentiment_scores:
        colors = []
        for n in G.nodes():
            s = sentiment_scores.get(n, 0)
            if s > 0.1:
                colors.append("#2ecc71")
            elif s < -0.1:
                colors.append("#e74c3c")
            else:
                colors.append("#95a5a6")
    else:
        colors = "#3498db"
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Keyword co-occurrence network (green=positive, red=negative, gray=neutral)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved network plot to {output_path}")


def run_network_analysis(df, text_col="text", label_col=None):
    """
    Full pipeline: freq -> co-occurrence -> optional SO -> network plot.
    df must have text_col; if label_col present, compute semantic orientation.
    """
    texts = df[text_col].dropna().astype(str).tolist()
    if not texts:
        raise ValueError("No texts in dataframe")
    G, word_freq = build_keyword_network(texts)
    sentiment_scores = None
    if label_col and label_col in df.columns:
        labels = (df[label_col].dropna() == "positive").astype(int).tolist()
        if len(labels) == len(texts):
            sentiment_scores = semantic_orientation_scores(texts, labels)
    plot_network(G, word_freq, sentiment_scores)
    # Save keyword table with optional SO
    keyword_list = list(word_freq.items())[:config.TOP_N_KEYWORDS]
    out_df = pd.DataFrame(keyword_list, columns=["keyword", "freq"])
    if sentiment_scores:
        out_df["semantic_orientation"] = out_df["keyword"].map(sentiment_scores)
    out_df.to_csv(os.path.join(config.OUTPUT_DIR, "keywords.csv"), index=False)
    return G, word_freq, sentiment_scores
