from __future__ import annotations
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "news.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    source TEXT,
    published_at TEXT,
    content TEXT,
    fetched_at TEXT NOT NULL,
    label TEXT,
    fake_prob REAL,
    confidence REAL,
    analysis_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles(fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_label ON articles(label);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with connect() as c:
        c.executescript(_SCHEMA)


def upsert_article(row: dict) -> bool:
    """Insert an article if not already present. Returns True if inserted."""
    init()
    with connect() as c:
        try:
            c.execute(
                """INSERT INTO articles (url, title, source, published_at, content, fetched_at)
                   VALUES (:url, :title, :source, :published_at, :content, :fetched_at)""",
                {**row, "fetched_at": datetime.now(timezone.utc).isoformat()},
            )
            return True
        except sqlite3.IntegrityError:
            return False


def save_analysis(url: str, label: str, fake_prob: float, confidence: float, analysis_json: str):
    init()
    with connect() as c:
        c.execute(
            "UPDATE articles SET label=?, fake_prob=?, confidence=?, analysis_json=? WHERE url=?",
            (label, fake_prob, confidence, analysis_json, url),
        )


def recent(limit: int = 50, label: str | None = None) -> list[dict]:
    init()
    with connect() as c:
        if label:
            rows = c.execute(
                "SELECT * FROM articles WHERE label=? ORDER BY fetched_at DESC LIMIT ?",
                (label, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM articles ORDER BY fetched_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def stats() -> dict:
    init()
    with connect() as c:
        total = c.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        by_label = dict(c.execute(
            "SELECT label, COUNT(*) FROM articles WHERE label IS NOT NULL GROUP BY label"
        ).fetchall())
        return {"total": total, "by_label": by_label}
