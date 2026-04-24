from __future__ import annotations
import os
import requests

_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def search(query: str, max_results: int = 5) -> list[dict]:
    key = os.environ.get("GOOGLE_FACTCHECK_KEY")
    if not key or not query.strip():
        return []
    try:
        r = requests.get(
            _ENDPOINT,
            params={"query": query[:300], "key": key, "pageSize": max_results, "languageCode": "en"},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    results = []
    for claim in data.get("claims", []):
        reviews = claim.get("claimReview", []) or [{}]
        first = reviews[0]
        results.append({
            "claim": claim.get("text"),
            "claimant": claim.get("claimant"),
            "publisher": (first.get("publisher") or {}).get("name"),
            "rating": first.get("textualRating"),
            "url": first.get("url"),
        })
    return results


def verdict_from_rating(rating: str | None) -> float | None:
    """Map a textual rating to fake_prob in [0, 1], or None if unknown."""
    if not rating:
        return None
    r = rating.lower()
    fake_markers = ["false", "incorrect", "misleading", "pants on fire", "fake", "inaccurate", "debunk"]
    real_markers = ["true", "correct", "accurate", "mostly true"]
    if any(m in r for m in fake_markers):
        return 0.9
    if any(m in r for m in real_markers):
        return 0.1
    if "half" in r or "mixed" in r or "partly" in r:
        return 0.5
    return None
