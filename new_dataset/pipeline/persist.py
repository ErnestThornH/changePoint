"""Fit + persist the best model (logreg_embed: German-FinBERT embeddings → LogisticRegression)
for live reaction inference in the dashboard. Run: python -m new_dataset.pipeline.persist"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STORE_DIR = "new_dataset/model_store"
MODEL_FILE = "logreg_embed.joblib"
META_FILE = "meta.json"


def fit_and_save(out_dir: str = STORE_DIR, generated_on: str | None = None) -> dict:
    """Embed all labeled texts and fit a LogisticRegression (same config as logreg_embed) on
    ALL data; dump model + meta. Heavy imports lazy."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from new_dataset.pipeline.labels import build_labeled
    from new_dataset.pipeline.models import _embed, BASE_MODEL

    items = build_labeled()
    X = _embed([it["text"] for it in items])
    y = [it["label"] for it in items]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    clf.fit(X, y)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / MODEL_FILE)
    meta = {"base_model": BASE_MODEL, "labels": list(clf.classes_),
            "n_train": len(items), "generated_on": generated_on}
    (out / META_FILE).write_text(json.dumps(meta, indent=2))
    print(f"[persist] fit on n={len(items)} → {out / MODEL_FILE} (classes={list(clf.classes_)})")
    return meta


def load(out_dir: str = STORE_DIR):
    """Return (model, meta) or (None, None) if the store is missing."""
    import joblib
    mp = Path(out_dir) / MODEL_FILE
    if not mp.exists():
        return None, None
    meta = json.loads((Path(out_dir) / META_FILE).read_text())
    return joblib.load(mp), meta


def predict(texts, out_dir: str = STORE_DIR) -> list[dict]:
    """[{label, proba:{down,flat,up}}] per text; or [{note}] if the store is missing.
    Probabilities mapped via model.classes_ (NOT assumed order)."""
    model, _ = load(out_dir)
    if model is None:
        note = "Modell nicht gebaut — `python -m new_dataset.pipeline.persist` ausführen."
        return [{"note": note} for _ in texts]
    from new_dataset.pipeline.models import _embed
    X = _embed(list(texts))
    classes = list(model.classes_)
    res = []
    for row in model.predict_proba(X):
        proba = {c: float(p) for c, p in zip(classes, row)}
        res.append({"label": classes[int(np.argmax(row))], "proba": proba})
    return res


if __name__ == "__main__":
    from datetime import datetime
    fit_and_save(generated_on=datetime.now().isoformat(timespec="seconds"))
