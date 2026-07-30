"""Pre-trained sentiment models for ad-hoc text (Step 5).

transformers/torch are imported lazily inside SentimentScorer so this module can be
imported (and its pure functions tested) without the heavy ML deps installed."""
from __future__ import annotations

# model_key -> {id: HF repo, lang: de|en, label_map: None | {raw_label: canon_label}}
# label_map overrides canon_label() for models whose labels canon_label can't resolve
# (e.g. "LABEL_0"). Leave None when the model emits positive/negative/neutral-like labels.
MODELS: dict[str, dict] = {
    "gfinbert":   {"id": "scherrmann/GermanFinBERT_SC_Sentiment", "lang": "de", "label_map": None},
    "oliverguhr": {"id": "oliverguhr/german-sentiment-bert",      "lang": "de", "label_map": None},
    "mdraw":      {"id": "mdraw/german-news-sentiment-bert",      "lang": "de", "label_map": None},
    "finbert_en": {"id": "ProsusAI/finbert",                       "lang": "en", "label_map": None},
}

GERMAN_MODELS = ["gfinbert", "oliverguhr", "mdraw"]
ENGLISH_MODELS = ["finbert_en"]

_CANON = {
    "positive": "positive", "positiv": "positive", "pos": "positive",
    "negative": "negative", "negativ": "negative", "neg": "negative",
    "neutral": "neutral", "neutra": "neutral",
}


def canon_label(raw: str) -> str:
    """Normalize a model's raw label to positive|negative|neutral (best effort)."""
    return _CANON.get((raw or "").strip().lower(), (raw or "").strip().lower())


def signed_from_probs(probs: dict[str, float]) -> tuple[str, float]:
    """(label, signed_score) where signed = P(positive) - P(negative), label = argmax."""
    if not probs:
        raise ValueError("signed_from_probs: empty probs (model returned no labels)")
    label = max(probs, key=probs.get)
    score = round(float(probs.get("positive", 0.0)) - float(probs.get("negative", 0.0)), 6)
    return label, score


def normalize(label_scores: list[dict], label_map: dict | None = None) -> tuple[str, float]:
    """Convert one text's pipeline output (list of {label, score}) to (label, signed)."""
    probs: dict[str, float] = {}
    for d in label_scores:
        raw = d["label"]
        mapped = (label_map or {}).get(raw)
        lab = mapped if mapped is not None else canon_label(raw)
        probs[lab] = float(d["score"])
    return signed_from_probs(probs)


class SentimentScorer:
    """Wraps one HF text-classification model. Lazy-loads on first score()."""

    def __init__(self, model_key: str):
        self.key = model_key
        self.cfg = MODELS[model_key]
        self._pipe = None

    def _load(self) -> None:
        from transformers import pipeline  # lazy: heavy import
        self._pipe = pipeline(
            "text-classification", model=self.cfg["id"],
            top_k=None, truncation=True,
        )

    def score(self, texts: list[str]) -> list[tuple[str, float]]:
        """Return [(label, signed_score)] aligned with `texts`."""
        if self._pipe is None:
            self._load()
        raw = self._pipe(list(texts), batch_size=16)
        return [normalize(item, self.cfg["label_map"]) for item in raw]
