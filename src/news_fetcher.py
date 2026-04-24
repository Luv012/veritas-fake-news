from __future__ import annotations
import os
import requests
import feedparser
from datetime import datetime, timezone

_NEWSAPI = "https://newsapi.org/v2/top-headlines"

# Free fallback RSS feeds — used when no NEWSAPI_KEY is set.
_RSS_FEEDS = [
    ("BBC", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Reuters", "https://www.reutersagency.com/feed/?best-topics=top-news&post_type=best"),
    ("NPR", "https://feeds.npr.org/1001/rss.xml"),
    ("The Guardian", "https://www.theguardian.com/world/rss"),
    ("The Hindu", "https://www.thehindu.com/news/national/feeder/default.rss"),
]


def _from_newsapi(country: str = "us", page_size: int = 50) -> list[dict]:
    key = os.environ.get("NEWSAPI_KEY")
    if not key:
        return []
    r = requests.get(
        _NEWSAPI,
        params={"country": country, "pageSize": page_size, "apiKey": key},
        timeout=15,
    )
    r.raise_for_status()
    articles = []
    for a in r.json().get("articles", []):
        articles.append({
            "url": a.get("url"),
            "title": a.get("title"),
            "source": (a.get("source") or {}).get("name"),
            "published_at": a.get("publishedAt"),
            "content": a.get("description") or a.get("content") or "",
        })
    return [a for a in articles if a.get("url")]


def _from_rss() -> list[dict]:
    articles = []
    for source_name, feed_url in _RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception:
            continue
        for entry in feed.entries[:20]:
            articles.append({
                "url": entry.get("link"),
                "title": entry.get("title"),
                "source": source_name,
                "published_at": entry.get("published") or datetime.now(timezone.utc).isoformat(),
                "content": entry.get("summary", ""),
            })
    return [a for a in articles if a.get("url")]


def fetch_latest() -> list[dict]:
    """Fetch today's headlines. NewsAPI if key set, else RSS fallback."""
    articles = _from_newsapi()
    if not articles:
        articles = _from_rss()
    # Dedupe by URL
    seen = set()
    unique = []
    for a in articles:
        if a["url"] in seen:
            continue
        seen.add(a["url"])
        unique.append(a)
    return unique


def extract_full_article(url: str) -> dict:
    """Fetch full article text using newspaper3k. Falls back to empty content on failure."""
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        return {
            "title": art.title,
            "content": art.text,
            "published_at": art.publish_date.isoformat() if art.publish_date else None,
        }
    except Exception as e:
        return {"title": None, "content": "", "published_at": None, "error": str(e)}
