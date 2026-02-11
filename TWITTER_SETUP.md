# Including Twitter in Your Sentiment Analysis

Your pipeline can include **Twitter** in the scraped data so the report and charts show "how people on Twitter react" alongside News and Reddit.

---

## Option 1: Twitter Developer account (free tier) – recommended

If you have a **Twitter / X developer account** (free tier), you can use the official API. No Python 3.11 needed; works on your current setup.

### 1. Get your Bearer Token

1. Go to [developer.x.com](https://developer.x.com) and sign in.
2. Open your **Project** → **App** (or create one).
3. In the app, open the **Keys and tokens** tab.
4. Under **Authentication Tokens**, copy the **Bearer Token** (or generate one).  
   The free tier includes **Search Tweets** (recent, last 7 days).

### 2. Set the token when running (don’t commit it)

**PowerShell (current session):**
```powershell
$env:TWITTER_BEARER_TOKEN = "your_bearer_token_here"
python run_analysis.py --skip-train
```

**Or create a `.env` file** in the project folder (same folder as `config.py`):
```
TWITTER_BEARER_TOKEN=your_bearer_token_here
```
The project loads `.env` automatically (python-dotenv is in requirements). `.env` is in `.gitignore` so the token is never committed.

### 3. Run the pipeline

With `TWITTER_BEARER_TOKEN` set:

```powershell
.\venv\Scripts\Activate.ps1
python run_analysis.py --skip-train
```

You should see **"Twitter (API)"** in the logs and Twitter in `output/sentiment_report.txt` and `sentiment_by_source.png`.

**Free tier note:** Recent Search is limited (e.g. 450 requests per 15 min, tweets from last 7 days). The scraper uses a few queries and stays within normal use.

---

## Option 2: Nitter RSS fallback (no account, Python 3.12+)

If you have **twitter** in `config.SCRAPE_SOURCES`, the scraper will:

1. Try **snscrape** (will fail on Python 3.12+).
2. Then try **Nitter RSS** (public Twitter-to-RSS). If a Nitter instance is up, you get some Twitter-style results with no extra setup.

Run the full pipeline with scraping:

```powershell
.\venv\Scripts\Activate.ps1
python run_analysis.py --skip-train
```

If Nitter works, you’ll see "Twitter (Nitter RSS)" in the logs and `source` will include `twitter` in the report.

---

## Option 2: Real Twitter via snscrape (Python 3.11)

To scrape **real Twitter/X search results** (not Nitter), snscrape must run on **Python 3.11 or earlier**.

### Step 1: Install Python 3.11

- Download from [python.org/downloads](https://www.python.org/downloads/) (e.g. 3.11.9).
- Install and ensure **"Add to PATH"** is checked (or note the install path).

### Step 2: Create a separate venv with Python 3.11

In PowerShell, from your project folder (`Sentiment Analysis`):

```powershell
# Use Python 3.11 explicitly (adjust path if needed)
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Step 3: Scrape with Twitter (News + Reddit + Twitter)

Still with `venv311` activated:

```powershell
python scraper.py
```

This writes `data/scraped_raw.csv` including rows with `source=twitter`.

### Step 4: Run the rest of the pipeline

You can either:

- **A)** Keep using the Python 3.11 venv for everything:

  ```powershell
  python run_analysis.py --skip-scrape --skip-train
  ```

- **B)** Switch back to your main venv and run prediction/report only (no new scraping):

  ```powershell
  .\venv\Scripts\Activate.ps1
  python run_analysis.py --skip-scrape --skip-train
  ```

  The existing `scraped_raw.csv` (with Twitter) is used; the script only re-reads it and runs sentiment + report.

### Step 5: Check the report

Open `output/sentiment_report.txt`. You should see:

- **Sentiment by source:** News (RSS), Reddit, and **Twitter** with counts and percentages.
- `output/sentiment_by_source.png` will include a Twitter bar.

---

## Summary

| Goal                         | What to do |
|-----------------------------|------------|
| Try to get Twitter data now | Use Option 1 (Nitter fallback); run `python run_analysis.py --skip-train`. |
| Use real Twitter search     | Use Option 2: Python 3.11 venv → run `python scraper.py` → then run analysis (Option 2 Step 4). |

Ensure `config.SCRAPE_SOURCES` includes `"twitter"` (it does by default).
