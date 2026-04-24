from __future__ import annotations
from dataclasses import dataclass, asdict
from . import classifier, source_credibility, fact_check


@dataclass
class Verdict:
    label: str           # FAKE | REAL | UNCERTAIN
    confidence: float    # 0..1
    fake_prob: float     # 0..1 combined
    ml: dict
    source: dict
    fact_checks: list
    fact_check_prob: float | None
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# Source tier → prior probability of being fake.
# This is the main signal. ML is only a tiebreaker.
_SOURCE_PRIOR = {"high": 0.10, "medium": 0.35, "unknown": 0.50, "low": 0.85}

# ML and fact-check weights. Source prior fills the remainder.
_ML_WEIGHT = 0.20        # pretrained fake-news models are unreliable on real news
_FACTCHECK_WEIGHT = 0.55  # when a human fact-check hits, trust it


def analyze(text: str, url: str | None = None) -> Verdict:
    reasons: list[str] = []

    ml = classifier.classify(text)
    reasons.append(f"ML classifier: {ml['label']} (fake_prob={ml['fake_prob']:.2f})")

    src = source_credibility.rate(url) if url else {"tier": "unknown", "score": 0.5, "domain": None}
    if src["domain"]:
        reasons.append(f"Source {src['domain']} → tier: {src['tier']}")
    elif url:
        reasons.append(f"Source {url} → not in credibility list (unknown)")
    else:
        reasons.append("No URL provided → source credibility skipped")

    checks = fact_check.search(text[:250]) if text else []
    fc_prob: float | None = None
    if checks:
        mapped = [fact_check.verdict_from_rating(c["rating"]) for c in checks]
        mapped = [m for m in mapped if m is not None]
        if mapped:
            fc_prob = sum(mapped) / len(mapped)
            reasons.append(f"Fact-check matches: {len(checks)} (avg fake_prob={fc_prob:.2f})")
    else:
        reasons.append("No matching fact-checks found")

    prior = _SOURCE_PRIOR[src["tier"]]

    if fc_prob is not None:
        source_weight = 1.0 - _ML_WEIGHT - _FACTCHECK_WEIGHT
        combined = (
            source_weight * prior
            + _ML_WEIGHT * ml["fake_prob"]
            + _FACTCHECK_WEIGHT * fc_prob
        )
    else:
        source_weight = 1.0 - _ML_WEIGHT
        combined = source_weight * prior + _ML_WEIGHT * ml["fake_prob"]

    if combined >= 0.65:
        label = "FAKE"
    elif combined <= 0.35:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    confidence = abs(combined - 0.5) * 2

    return Verdict(
        label=label,
        confidence=round(confidence, 3),
        fake_prob=round(combined, 3),
        ml=ml,
        source=src,
        fact_checks=checks,
        fact_check_prob=round(fc_prob, 3) if fc_prob is not None else None,
        reasons=reasons,
    )
