from __future__ import annotations
import os
from functools import lru_cache

_MODEL_NAME = os.environ.get("FAKE_NEWS_MODEL", "hamzab/roberta-fake-news-classification")

_FAKE_TOKENS = ("fake", "false", "unreliable", "misinfo", "label_0")
_REAL_TOKENS = ("true", "real", "reliable", "label_1")


@lru_cache(maxsize=1)
def _pipeline():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    return pipeline("text-classification", model=mdl, tokenizer=tok,
                    truncation=True, max_length=512, top_k=None)


def _fake_prob_from_scores(scores: list[dict]) -> tuple[float, str]:
    """Pick fake probability from all label scores. Robust to FAKE/TRUE vs LABEL_0/LABEL_1."""
    fake = None
    real = None
    for s in scores:
        lbl = s["label"].lower()
        if any(tok in lbl for tok in _FAKE_TOKENS):
            fake = s["score"]
        elif any(tok in lbl for tok in _REAL_TOKENS):
            real = s["score"]
    if fake is None and real is not None:
        fake = 1.0 - real
    if fake is None:
        # Unknown label scheme — fall back to first score
        fake = scores[0]["score"]
    raw = ", ".join(f"{s['label']}={s['score']:.2f}" for s in scores)
    return float(fake), raw


def classify(text: str) -> dict:
    """Return {'label': 'FAKE'|'REAL', 'fake_prob': float 0..1}."""
    text = (text or "").strip()
    if not text:
        return {"label": "REAL", "fake_prob": 0.5, "note": "empty input"}
    pipe = _pipeline()
    out = pipe(text[:4000])
    # pipeline with top_k=None returns List[List[dict]] or List[dict] depending on version
    scores = out[0] if out and isinstance(out[0], list) else out
    fake_prob, raw = _fake_prob_from_scores(scores)
    label = "FAKE" if fake_prob >= 0.5 else "REAL"
    return {"label": label, "fake_prob": round(fake_prob, 4), "raw_scores": raw}
