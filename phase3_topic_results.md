## 3.3 Topic Modeling & Semantic Drivers
### 3.3.1 Coherence Reporting
- **Method:** LDA (scikit-learn) with Gensim C_v coherence evaluation
- **Best k:** 8
- **Best C_v coherence:** 0.5013 (acceptable)
- **LDA priors:** alpha=0.1, beta(eta)=0.01
- **Training setup:** iterations=50, random_seed=42

| k | C_v |
|---:|---:|
| 4 | 0.4694 |
| 6 | 0.4927 |
| 8 | 0.5013 |

![Coherence Sensitivity Across k](figures/topic_coherence_k.png)

### 3.3.2 Model Comparison
| Model | Topics | C_v | Interpretation Quality | Notes |
|-------|-------:|----:|------------------------|-------|
| LDA | 8 | 0.5013 | Moderate | Broad-to-moderate thematic groupings |
| NMF (TF-IDF) | 8 | 0.5358 | Strong | Alternative baseline; often sharper lexical boundaries |

Model choice is justified by coherence and interpretation quality with fixed, reproducible seeds.

### 3.3.3 Stability Testing
- **Seed stability (topic overlap consistency):** mean=0.700, sd=0.424 across 2 runs
- **Bootstrap stability (80% resamples):** mean=0.392, sd=0.065 across 2 runs
- **Consistency metric:** symmetric average of max Jaccard overlap between topic word sets (top-10 words/topic).

Stability outputs: `outputs/topic_stability_seed.csv`, `outputs/topic_stability_bootstrap.csv`.

### 3.3.4 Semantic Drivers of Sentiment
Ranked using smoothed log-odds (positive vs negative corpora, informative prior).

| Rank | Negative Drivers | Positive Drivers |
|-----:|------------------|------------------|
| 1 | policy (-1.239) | healthy (0.820) |
| 2 | usda (-0.942) | nutrition (0.646) |
| 3 | parents (-0.938) | chocolate (0.540) |
| 4 | children (-0.802) | dairy (0.418) |
| 5 | food (-0.402) | whole (0.190) |
| 6 | fat (-0.358) | cafeteria (0.115) |
| 7 | student (-0.243) | schools (0.084) |
| 8 | kids (-0.097) | students (0.076) |
| 9 | health (-0.026) | lunch (0.075) |
| 10 | school (-0.006) | milk (0.064) |

These terms act as explanatory mechanisms by quantifying which words are disproportionately associated with negative versus positive sentiment.
