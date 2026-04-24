from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import urlparse
import tldextract

_DATA = Path(__file__).resolve().parent.parent / "data" / "source_credibility.json"

with _DATA.open() as f:
    _DB = json.load(f)

_LOOKUP: dict[str, str] = {}
for tier, domains in _DB.items():
    for d in domains:
        _LOOKUP[d.lower()] = tier


def _root_domain(url_or_domain: str) -> str | None:
    if not url_or_domain:
        return None
    candidate = url_or_domain.strip().lower()
    if "://" not in candidate:
        candidate = "http://" + candidate
    host = urlparse(candidate).hostname or ""
    ext = tldextract.extract(host)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}"


def rate(url_or_domain: str) -> dict:
    """Return {'tier': high|medium|low|unknown, 'score': 0..1, 'domain': ...}."""
    domain = _root_domain(url_or_domain)
    if not domain:
        return {"tier": "unknown", "score": 0.5, "domain": None}
    tier = _LOOKUP.get(domain, "unknown")
    score = {"high": 0.9, "medium": 0.6, "low": 0.1, "unknown": 0.5}[tier]
    return {"tier": tier, "score": score, "domain": domain}
