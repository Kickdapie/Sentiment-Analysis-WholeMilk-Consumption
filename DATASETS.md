# Sentiment Training Datasets

Your topic (whole milk school policy) is **recent** (debate peaked around 2024–2026; Whole Milk for Healthy Kids Act signed Jan 2026). Sentiment140 is from **2009**, so language and style are outdated.

## Better Kaggle options

| Dataset | Why it’s a better fit | Link | Notes |
|--------|------------------------|------|------|
| **Twitter Sentiment Dataset** | Newer than Sentiment140, similar task | [saurabhshahane/twitter-sentiment-dataset](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset) | Check CSV columns (e.g. `text`, `label` or `sentiment`). |
| **News Sentiment Dataset** | News text is closer to your scraped content (news + Reddit) | [hoshi7/news-sentiment-dataset](https://www.kaggle.com/datasets/hoshi7/news-sentiment-dataset) | Good match for News RSS + Reddit. |
| **Social Media Sentiments Analysis** | Broader social media, often newer | [kashishparmar02/social-media-sentiments-analysis-dataset](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset) | Verify column names and label encoding. |
| **Tweet Sentiment Extraction** (competition) | More recent tweets, sentiment spans | [tweet-sentiment-extraction](https://www.kaggle.com/c/tweet-sentiment-extraction/data) | May need to use `text` + `sentiment` and map to binary. |

## Recommendation

- **Best single replacement:** **News Sentiment Dataset** – your pipeline already scrapes news (RSS) and Reddit, so a news-sentiment dataset aligns better with your data than 2009 Twitter.
- **If you want to stay with Twitter-style text:** Use **Twitter Sentiment Dataset** (saurabhshahane) as a newer alternative to Sentiment140.

## No perfect dataset

There is no Kaggle dataset specifically for “whole milk school policy” sentiment. Any choice is a trade-off:

- **Sentiment140:** Old (2009) but large and well-known.
- **News Sentiment:** Newer style, matches news/Reddit content better.
- **Newer Twitter datasets:** Newer language, still Twitter-specific.

You can also **label a few hundred of your own scraped texts** and fine-tune or train a small classifier on that for a more domain-specific model.

## Using Twitter Sentiment Dataset (saurabhshahane) — current setup

This project is configured to use the **Twitter Sentiment Dataset** by saurabhshahane (newer than Sentiment140).

1. **Download:** [Twitter Sentiment Dataset](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset) on Kaggle.
2. **Place:** Put the training CSV (often named `train.csv`) in `data/kaggle_alt/train.csv`.
3. **Run:** `python train_sentiment_model.py` or the full pipeline. The loader auto-detects columns `text`/`tweet` and `label`/`sentiment`.

If your CSV uses different column names, set `KAGGLE_ALT_TEXT_COL` and `KAGGLE_ALT_LABEL_COL` in `config.py`.

## Using another alternative dataset

1. Download a dataset and place the CSV in `data/kaggle_alt/` (e.g. `train.csv`).
2. In `config.py` set:
   - `KAGGLE_DATASET = "alternative"`
   - `KAGGLE_ALT_PATH` to the path of your CSV
   - `KAGGLE_ALT_TEXT_COL` and `KAGGLE_ALT_LABEL_COL` to the CSV’s text and label column names
3. Labels are mapped to 0 (negative) / 1 (positive); the loader accepts numeric (0/1 or 0/4) or strings like `"negative"` / `"positive"`.
4. Run `python train_sentiment_model.py` (or the full pipeline).
