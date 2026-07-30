"""3-class reaction label (up/down/flat) from the 10-min price delta + labeled-data builder."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EPS = 0.1
LABEL_HORIZON = "price_delta_pct_10m"


def label_3class(delta_pct, eps: float = EPS):
    """'up' if delta>eps, 'down' if delta<-eps, 'flat' if |delta|<=eps, None if missing."""
    if delta_pct is None or (isinstance(delta_pct, float) and pd.isna(delta_pct)):
        return None
    d = float(delta_pct)
    if d > eps:
        return "up"
    if d < -eps:
        return "down"
    return "flat"


def build_labeled(dataset_json: str = "new_dataset/adhoc_functioning.json",
                  stats_csv: str = "new_dataset/event_window_stats.csv",
                  eps: float = EPS) -> list[dict]:
    """Join records (text) with the 10-min delta on `id`; label 3-class; drop rows with no
    10-min delta. text = clean_fulltext(Titel + content)."""
    from text_clean import clean_fulltext
    records = {r["id"]: r for r in json.loads(Path(dataset_json).read_text())}
    stats = pd.read_csv(stats_csv)
    out = []
    for _, row in stats.iterrows():
        rid = row["id"]
        lab = label_3class(row.get(LABEL_HORIZON), eps)
        rec = records.get(rid)
        if lab is None or rec is None:
            continue
        out.append({"id": rid, "text": clean_fulltext(rec.get("title", ""), rec.get("content", "")),
                    "label": lab})
    return out
