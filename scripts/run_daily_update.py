"""Fetch today's news, classify each article, and store results.

Run manually:  python scripts/run_daily_update.py
Run on a schedule: see scripts/scheduler.py  (or use Windows Task Scheduler).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src import database, news_fetcher, detector


def main():
    database.init()
    articles = news_fetcher.fetch_latest()
    print(f"Fetched {len(articles)} articles")

    new_count = 0
    analyzed_count = 0
    for a in articles:
        inserted = database.upsert_article(a)
        if not inserted:
            continue
        new_count += 1
        text = f"{a.get('title') or ''}. {a.get('content') or ''}".strip()
        if not text:
            continue
        try:
            v = detector.analyze(text, url=a["url"])
            database.save_analysis(
                url=a["url"],
                label=v.label,
                fake_prob=v.fake_prob,
                confidence=v.confidence,
                analysis_json=json.dumps(v.to_dict(), default=str),
            )
            analyzed_count += 1
            print(f"  [{v.label}] {a.get('title')[:80]}")
        except Exception as e:
            print(f"  [ERR] {a.get('url')}: {e}")

    print(f"Done. {new_count} new articles, {analyzed_count} analyzed.")
    print("Stats:", database.stats())


if __name__ == "__main__":
    main()
