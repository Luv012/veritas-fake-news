# Fake News Detector

An AIML project that combines three independent signals to judge whether a news story is likely fake:

1. **ML classifier** — a transformer (RoBERTa fine-tuned on fake-news data) scores the text.
2. **Source credibility** — a curated tier list (`data/source_credibility.json`) rates the publishing domain.
3. **Live fact-check lookup** — Google's Fact Check Tools API returns human verdicts on matching claims.

A daily ingestion job pulls top headlines (NewsAPI or RSS fallback), classifies them, and stores results in SQLite. Streamlit provides the UI.

---

## Quick start

```bash
# 1. Install deps (use a venv)
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2. Copy env template & (optionally) add keys
copy .env.example .env
#   NEWSAPI_KEY         – free at https://newsapi.org (else RSS is used)
#   GOOGLE_FACTCHECK_KEY – free at https://console.cloud.google.com (enable "Fact Check Tools API")

# 3. Launch the UI
streamlit run app.py
```

First analysis call downloads the transformer model (~500 MB) into the HuggingFace cache. Subsequent calls are fast (CPU, ~1–2 s per article).

---

## Daily news updates

Two ways to run the daily job:

**Option A — keep the scheduler running**

```bash
python scripts/scheduler.py
```

Runs once immediately, then daily at 07:00. Leave the terminal open.

**Option B — Windows Task Scheduler (recommended for a student project)**

1. Open Task Scheduler → *Create Basic Task*
2. Trigger: Daily, e.g. 07:00
3. Action: Start a program
   - Program: `C:\Users\Luv\OneDrive\Desktop\news_detector\.venv\Scripts\python.exe`
   - Arguments: `scripts\run_daily_update.py`
   - Start in: `C:\Users\Luv\OneDrive\Desktop\news_detector`

Either way, new articles land in `data/news.db` and show up in the **Today's feed** tab.

---

## Project layout

```
news_detector/
├── app.py                         Streamlit UI
├── requirements.txt
├── .env.example
├── data/
│   ├── source_credibility.json    Tiered domain list
│   └── news.db                    SQLite — created on first run
├── src/
│   ├── classifier.py              Transformer fake-news classifier
│   ├── source_credibility.py      Domain → credibility tier
│   ├── fact_check.py              Google Fact Check API client
│   ├── detector.py                Orchestrator — weighted score
│   ├── database.py                SQLite helpers
│   └── news_fetcher.py            NewsAPI + RSS fallback + full-text extract
├── scripts/
│   ├── run_daily_update.py        Fetch + classify today's headlines
│   └── scheduler.py               APScheduler wrapper (Option A above)
└── train/
    └── train_tfidf_baseline.py    Optional classical baseline on ISOT
```

---

## How the final score is computed

`src/detector.py` combines the three signals into a single `fake_prob ∈ [0, 1]`:

| Signal      | Weight (no fact-check) | Weight (fact-check hit) |
|-------------|-----------------------:|------------------------:|
| ML classifier | 0.55 | 0.35 |
| Source       | 0.25 | 0.15 |
| Fact-check   | 0.20 | 0.50 |

- `≥ 0.65` → **FAKE**
- `≤ 0.35` → **REAL**
- otherwise → **UNCERTAIN**

Confidence = `|fake_prob − 0.5| × 2`.

---

## Using a custom-trained classifier

If you want to show your own trained model in the project report, run
`train/train_tfidf_baseline.py` on the ISOT dataset. You can then point
`src/classifier.py` at the joblib file instead of HuggingFace by swapping the
pipeline function — see the comment in that file.

---

## What to demo in class

1. **Instant detection** — paste a known fake story in the *Check* tab and show the verdict with reasons.
2. **Live news feed** — click *Fetch & analyze latest*; walk through today's headlines with verdicts.
3. **Daily automation** — show the Windows Task Scheduler entry running the pipeline.
4. **Model comparison** (if you train the baseline) — compare TF-IDF LR vs. transformer metrics on ISOT.
