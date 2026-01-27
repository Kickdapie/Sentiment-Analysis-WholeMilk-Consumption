"""
Scraper for "Whole milk consumption school policy" content.
Gathers text from News RSS and Reddit (optional Twitter via snscrape) for sentiment analysis.
Methodology inspired by PeerJ CS 1149 (vegan tweets).
"""

import os
import re
import time
import csv
from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
import requests
import feedparser
from tqdm import tqdm

import config


def clean_text(text):
    """Normalize and clean scraped text."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]  # cap length for consistency


def scrape_news_rss(queries=None, max_per_query=config.MAX_ITEMS_PER_QUERY):
    """Scrape news articles via Google News RSS (no API key)."""
    queries = queries or config.SEARCH_QUERIES
    rows = []
    for q in tqdm(queries, desc="News RSS"):
        url = (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            feed = feedparser.parse(
                url,
                request_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"},
            )
            for i, entry in enumerate(feed.entries):
                if i >= max_per_query:
                    break
                title = clean_text(entry.get("title", ""))
                summary = clean_text(entry.get("summary", ""))
                text = f"{title}. {summary}" if summary else title
                if len(text) < 20:
                    continue
                rows.append({
                    "source": "news_rss",
                    "query": q,
                    "text": text,
                    "title": title,
                    "url": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception as e:
            tqdm.write(f"News RSS error for '{q}': {e}")
        time.sleep(0.5)
    return rows


def scrape_reddit(queries=None, max_per_query=config.MAX_ITEMS_PER_QUERY):
    """Scrape Reddit via public search JSON (no auth required for limited use)."""
    queries = queries or config.SEARCH_QUERIES
    rows = []
    base = "https://www.reddit.com/search.json"
    headers = {"User-Agent": "SentimentAnalysisBot/1.0 (SchoolMilkPolicy)"}
    for q in tqdm(queries, desc="Reddit"):
        try:
            r = requests.get(
                base,
                params={"q": q, "limit": min(25, max_per_query), "type": "link"},
                headers=headers,
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            for child in data.get("data", {}).get("children", [])[:max_per_query]:
                post = child.get("data", {})
                title = clean_text(post.get("title", ""))
                selftext = clean_text(post.get("selftext", ""))
                text = f"{title}. {selftext}".strip().rstrip(".")
                if len(text) < 15:
                    continue
                rows.append({
                    "source": "reddit",
                    "query": q,
                    "text": text,
                    "title": title,
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "published": datetime.utcfromtimestamp(post.get("created_utc", 0)).isoformat(),
                })
        except Exception as e:
            tqdm.write(f"Reddit error for '{q}': {e}")
        time.sleep(1)
    return rows


def scrape_twitter_snscrape(queries=None, max_per_query=50):
    """Optional: scrape Twitter/X using snscrape (may have rate limits)."""
    try:
        import snscrape.modules.twitter as sntwitter
    except ImportError:
        tqdm.write("snscrape not installed. Skip Twitter or: pip install snscrape")
        return []
    queries = queries or config.SEARCH_QUERIES
    rows = []
    for q in tqdm(queries, desc="Twitter"):
        try:
            scraper = sntwitter.TwitterSearchScraper(q)
            for i, tweet in enumerate(scraper.get_items()):
                if i >= max_per_query:
                    break
                text = clean_text(tweet.rawContent or tweet.content or "")
                if len(text) < 10:
                    continue
                rows.append({
                    "source": "twitter",
                    "query": q,
                    "text": text,
                    "title": "",
                    "url": tweet.url or "",
                    "published": tweet.date.isoformat() if tweet.date else "",
                })
        except Exception as e:
            tqdm.write(f"Twitter error for '{q}': {e}")
        time.sleep(1)
    return rows


def run_scraper(sources=None):
    """Run enabled scrapers and save to config.SCRAPED_RAW_PATH."""
    sources = sources or [config.SCRAPE_SOURCE]
    if "twitter" in sources:
        sources = [s for s in sources if s != "twitter"] + ["twitter"]
    all_rows = []
    if "news_rss" in sources:
        all_rows.extend(scrape_news_rss())
    if "reddit" in sources:
        all_rows.extend(scrape_reddit())
    if "twitter" in sources:
        all_rows.extend(scrape_twitter_snscrape(max_per_query=30))
    if not all_rows:
        # Fallback: create sample data so pipeline still runs
        sample = (
            "Schools should offer whole milk again. Kids need the nutrition. "
            "Banning whole milk in school lunches was a mistake."
        )
        all_rows = [{"source": "sample", "query": "whole milk school", "text": sample, "title": "", "url": "", "published": ""}]
        tqdm.write("No scraped results; using small sample so pipeline can run.")
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["text"], keep="first")
    os.makedirs(os.path.dirname(config.SCRAPED_RAW_PATH), exist_ok=True)
    df.to_csv(config.SCRAPED_RAW_PATH, index=False, encoding="utf-8")
    tqdm.write(f"Saved {len(df)} items to {config.SCRAPED_RAW_PATH}")
    return df


if __name__ == "__main__":
    run_scraper(sources=["news_rss", "reddit"])
