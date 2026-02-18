"""
Scraper for "Whole milk consumption school policy" content.
Gathers text from News RSS, Reddit, Bluesky, and optional Twitter for sentiment analysis.
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
    """Scrape Reddit via public search JSON with pagination (no auth required for limited use)."""
    queries = queries or config.SEARCH_QUERIES
    rows = []
    base = "https://www.reddit.com/search.json"
    headers = {"User-Agent": "SentimentAnalysisBot/1.0 (SchoolMilkPolicy)"}
    for q in tqdm(queries, desc="Reddit"):
        collected = 0
        after = None
        while collected < max_per_query:
            try:
                params = {"q": q, "limit": 100, "type": "link"}
                if after:
                    params["after"] = after
                r = requests.get(base, params=params, headers=headers, timeout=10)
                r.raise_for_status()
                data = r.json()
                children = data.get("data", {}).get("children", [])
                if not children:
                    break
                for child in children:
                    if collected >= max_per_query:
                        break
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
                    collected += 1
                after = data.get("data", {}).get("after")
                if not after:
                    break
                time.sleep(1)
            except Exception as e:
                tqdm.write(f"Reddit error for '{q}': {e}")
                break
        time.sleep(0.5)
    return rows


def _bluesky_post_to_row(post, query):
    """Convert a Bluesky post (dict or atproto model) to a row dict."""
    # Handle atproto SDK response objects
    if hasattr(post, "record"):
        record = post.record
        text = clean_text(getattr(record, "text", "") or "")
        author = getattr(post, "author", None)
        handle = getattr(author, "handle", "") if author else ""
        uri = getattr(post, "uri", "") or ""
        created_at = getattr(record, "created_at", "") or ""
    else:
        # Raw dict from public API
        record = post.get("record", {})
        text = clean_text(record.get("text", ""))
        author = post.get("author", {})
        handle = author.get("handle", "")
        uri = post.get("uri", "")
        created_at = record.get("createdAt", "")
    if len(text) < 15:
        return None
    post_url = ""
    if uri and handle:
        parts = uri.split("/")
        if len(parts) >= 5:
            rkey = parts[-1]
            post_url = f"https://bsky.app/profile/{handle}/post/{rkey}"
    return {
        "source": "bluesky",
        "query": query,
        "text": text,
        "title": "",
        "url": post_url,
        "published": created_at,
    }


def _scrape_bluesky_sdk(queries, max_per_query):
    """Scrape Bluesky using the atproto SDK with authentication."""
    bsky_handle = getattr(config, "BLUESKY_HANDLE", "") or os.environ.get("BLUESKY_HANDLE", "").strip()
    bsky_password = getattr(config, "BLUESKY_APP_PASSWORD", "") or os.environ.get("BLUESKY_APP_PASSWORD", "").strip()
    if not bsky_handle or not bsky_password:
        return None
    try:
        from atproto import Client
    except ImportError:
        tqdm.write("Bluesky SDK: pip install atproto")
        return None
    rows = []
    try:
        client = Client()
        client.login(bsky_handle, bsky_password)
        tqdm.write(f"Bluesky: logged in as {bsky_handle}")
    except Exception as e:
        tqdm.write(f"Bluesky login failed: {e}")
        return None
    for q in tqdm(queries, desc="Bluesky (SDK)"):
        collected = 0
        cursor = None
        while collected < max_per_query:
            try:
                resp = client.app.bsky.feed.search_posts(
                    params={"q": q, "limit": 100, "cursor": cursor} if cursor
                    else {"q": q, "limit": 100}
                )
                posts = getattr(resp, "posts", []) or []
                if not posts:
                    break
                for post in posts:
                    if collected >= max_per_query:
                        break
                    row = _bluesky_post_to_row(post, q)
                    if row:
                        rows.append(row)
                        collected += 1
                cursor = getattr(resp, "cursor", None)
                if not cursor:
                    break
                time.sleep(0.3)
            except Exception as e:
                tqdm.write(f"Bluesky SDK error for '{q}': {e}")
                break
        time.sleep(0.2)
    return rows if rows else None


def _scrape_bluesky_public(queries, max_per_query):
    """Scrape Bluesky via the public API (no auth required)."""
    rows = []
    base = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
    headers = {
        "User-Agent": "SentimentAnalysisBot/1.0 (SchoolMilkPolicy)",
        "Accept": "application/json",
    }
    for q in tqdm(queries, desc="Bluesky"):
        collected = 0
        cursor = None
        while collected < max_per_query:
            try:
                params = {"q": q, "limit": 100}
                if cursor:
                    params["cursor"] = cursor
                r = requests.get(base, params=params, headers=headers, timeout=15)
                r.raise_for_status()
                data = r.json()
                posts = data.get("posts", [])
                if not posts:
                    break
                for post in posts:
                    if collected >= max_per_query:
                        break
                    row = _bluesky_post_to_row(post, q)
                    if row:
                        rows.append(row)
                        collected += 1
                cursor = data.get("cursor")
                if not cursor:
                    break
                time.sleep(0.5)
            except Exception as e:
                tqdm.write(f"Bluesky public API error for '{q}': {e}")
                break
        time.sleep(0.3)
    return rows if rows else None


def scrape_bluesky(queries=None, max_per_query=config.MAX_ITEMS_PER_QUERY):
    """Scrape Bluesky posts. Tries authenticated SDK first, then public API."""
    queries = queries or config.SEARCH_QUERIES
    # 1) Try authenticated SDK (more reliable)
    rows = _scrape_bluesky_sdk(queries, max_per_query)
    if rows:
        tqdm.write(f"Bluesky: got {len(rows)} posts via SDK.")
        return rows
    # 2) Fall back to public API (no auth)
    tqdm.write("Bluesky: trying public API (no auth)...")
    rows = _scrape_bluesky_public(queries, max_per_query)
    if rows:
        tqdm.write(f"Bluesky: got {len(rows)} posts via public API.")
        return rows
    tqdm.write("Bluesky: no data. Set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD in .env for authenticated access.")
    return []


def scrape_twitter_api(queries=None, max_per_query=30):
    """Scrape Twitter/X using official API (free tier). Requires TWITTER_BEARER_TOKEN in environment."""
    token = getattr(config, "TWITTER_BEARER_TOKEN", "") or os.environ.get("TWITTER_BEARER_TOKEN", "").strip()
    if not token:
        return None  # no credentials, caller will try other methods
    try:
        import tweepy
    except ImportError:
        tqdm.write("Twitter API: pip install tweepy")
        return None
    queries = queries or config.SEARCH_QUERIES
    rows = []
    try:
        client = tweepy.Client(bearer_token=token)
        for q in tqdm(queries[:7], desc="Twitter (API)"):
            try:
                resp = client.search_recent_tweets(
                    query=q,
                    max_results=min(100, max_per_query * 2),
                    tweet_fields=["created_at", "text"],
                    expansions=["author_id"],
                    user_fields=["username"],
                )
                if resp.data:
                    for t in resp.data:
                        text = clean_text(getattr(t, "text", "") or "")
                        if len(text) < 10:
                            continue
                        rows.append({
                            "source": "twitter",
                            "query": q,
                            "text": text,
                            "title": "",
                            "url": f"https://twitter.com/i/status/{t.id}" if getattr(t, "id", None) else "",
                            "published": t.created_at.isoformat() if getattr(t, "created_at", None) else "",
                        })
                        if len(rows) >= max_per_query * len(queries[:7]):
                            break
            except Exception as e:
                tqdm.write(f"Twitter API search error for '{q}': {e}")
            time.sleep(1)
        if rows:
            tqdm.write(f"Twitter: got {len(rows)} tweets via API.")
        return rows if rows else None
    except Exception as e:
        tqdm.write(f"Twitter API error: {e}")
        return None


def scrape_twitter_nitter_rss(queries=None, max_per_query=30):
    """Fallback: try to get Twitter search results via Nitter RSS (works on Python 3.12+)."""
    queries = queries or config.SEARCH_QUERIES
    rows = []
    base_urls = [
        "https://nitter.poast.org/search/rss",
        "https://nitter.privacydev.net/search/rss",
    ]
    for q in tqdm(queries[:5], desc="Twitter (Nitter RSS)"):
        for base in base_urls:
            try:
                url = f"{base}?f=tweets&q={quote_plus(q)}"
                feed = feedparser.parse(
                    url,
                    request_headers={"User-Agent": "Mozilla/5.0 (compatible; SentimentBot/1.0)"},
                )
                for i, entry in enumerate(feed.entries):
                    if i >= max_per_query:
                        break
                    title = clean_text(entry.get("title", ""))
                    summary = clean_text(entry.get("summary", ""))
                    text = f"{title}. {summary}".strip().rstrip(".") or title
                    if len(text) < 15:
                        continue
                    rows.append({
                        "source": "twitter",
                        "query": q,
                        "text": text,
                        "title": title,
                        "url": entry.get("link", ""),
                        "published": entry.get("published", ""),
                    })
                if rows:
                    time.sleep(0.5)
                    break
            except Exception:
                continue
            time.sleep(0.5)
    return rows


def scrape_twitter_snscrape(queries=None, max_per_query=50):
    """Scrape Twitter/X: try API (if Bearer token set), then snscrape, then Nitter RSS."""
    # 1) Twitter Developer API (free tier) – works on any Python
    if getattr(config, "TWITTER_BEARER_TOKEN", "") or os.environ.get("TWITTER_BEARER_TOKEN", "").strip():
        rows = scrape_twitter_api(queries=queries, max_per_query=max_per_query)
        if rows:
            return rows
    # 2) snscrape (needs Python 3.11 or earlier)
    try:
        import snscrape.modules.twitter as sntwitter
    except (ImportError, AttributeError):
        tqdm.write("Twitter: snscrape not available (Python 3.12+). Trying Nitter RSS fallback...")
        rows = scrape_twitter_nitter_rss(queries=queries, max_per_query=max_per_query)
        if rows:
            tqdm.write(f"Twitter: got {len(rows)} items via Nitter RSS.")
        else:
            tqdm.write("Twitter: no data. Set TWITTER_BEARER_TOKEN for API, or use Python 3.11 (see TWITTER_SETUP.md).")
        return rows
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
    sources = sources or getattr(config, "SCRAPE_SOURCES", [config.SCRAPE_SOURCE])
    if "twitter" in sources:
        sources = [s for s in sources if s != "twitter"] + ["twitter"]
    all_rows = []
    if "news_rss" in sources:
        all_rows.extend(scrape_news_rss())
    if "reddit" in sources:
        all_rows.extend(scrape_reddit())
    if "bluesky" in sources:
        all_rows.extend(scrape_bluesky())
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
    run_scraper(sources=getattr(config, "SCRAPE_SOURCES", ["news_rss", "reddit"]))
