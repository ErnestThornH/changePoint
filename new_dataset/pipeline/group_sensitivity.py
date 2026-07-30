"""Gruppierte Sensitivitätsprüfung (Ticker-GroupKFold) für logreg_embed sowie
Kennzahlen zu Ticker-Overlap des Holdout-Splits und Cross-Session-Anteil der
10-Minuten-Slots. Ergänzt Kapitel 5.4/3.6/3.4; deterministisch (seed 0).

Beantwortet eine Reviewer-Sorge zu Datenleckage über drei Kennzahlen:
(a) ticker-gruppierte CV-Macro-F1 für logreg_embed (statt zufälliger 5-Fold-CV),
(b) Ticker-Overlap zwischen Train- und Test-Split des bestehenden Holdouts,
(c) Anteil der 10-Minuten-Messfenster, die eine Handelstag-Grenze überschreiten.

Run: python -m new_dataset.pipeline.group_sensitivity
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from new_dataset.pipeline.dataset import load_records
from new_dataset.pipeline.eodhd_prices import load_or_fetch
from new_dataset.pipeline.evaluation import stratified_holdout
from new_dataset.pipeline.exchanges import get_exch, to_exchange_hours
from new_dataset.pipeline.labels import build_labeled
from new_dataset.pipeline.models import BASE_MODEL, LABELS
from new_dataset.pipeline.slots import BAR_MINUTES

DATASET_JSON = "new_dataset/adhoc_functioning.json"
STATS_CSV = "new_dataset/event_window_stats.csv"
OUT_CSV = "new_dataset/figures/tables/tbl_group_sensitivity.csv"


def holdout_ticker_overlap(train_tickers, test_tickers) -> float:
    """Anteil der Test-Zeilen, deren Ticker auch im Train-Split vorkommt (pure, kein I/O)."""
    train_set = set(train_tickers)
    test_list = list(test_tickers)
    if not test_list:
        return 0.0
    shared = sum(1 for t in test_list if t in train_set)
    return shared / len(test_list)


def _embed_all(texts, batch: int = 16) -> np.ndarray:
    """Einmaliges Laden von Tokenizer + Modell, gebatchtes CLS-Embedding.

    Spiegelt ``models._embed`` (gleiches BASE_MODEL, gleiches Truncation/Pooling), lädt
    Tokenizer/Modell aber nur EINMAL für den gesamten Textbestand statt pro Aufruf.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(list(texts[i:i + batch]), truncation=True, max_length=256, padding=True,
                      return_tensors="pt").to(dev)
            out.append(model(**enc).last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 768))


def group_cv_macro_f1(embeddings, labels, tickers, n_splits: int = 5) -> list[float]:
    """Ticker-GroupKFold: pro Fold LogisticRegression (wie logreg_embed) fitten, macro-F1
    auf dem Fold-Holdout auswerten. Deterministisch (GroupKFold hat keinen Zufallsanteil;
    LogisticRegression mit random_state=0)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    X = np.asarray(embeddings)
    y = np.asarray(labels)
    groups = np.asarray(tickers)
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        scores.append(float(f1_score(y[test_idx], preds, labels=LABELS,
                                      average="macro", zero_division=0)))
    return scores


def cross_session_share_10m(dataset_json: str = DATASET_JSON,
                            stats_csv: str = STATS_CSV) -> tuple[int, int]:
    """(n_cross, n_total) über alle Meldungen mit vorhandenem 10-Min-Preis-KPI.

    Fensterdefinition mirrort ``slots.slot_change`` exakt: Der 5-Min-Bar, in den die Meldung
    fällt, wird verworfen; das 10-Min-Fenster sind die ZWEI vollen 5-Min-Bars unmittelbar
    danach. Cross-Session = das letzte Bar dieses Fensters liegt an einem SPÄTEREN Handelstag
    als das letzte Bar VOR der Meldung (oder es existiert gar kein Vorher-Bar, weil die
    Meldung außerhalb der gecachten Session-Historie liegt).
    """
    stats = pd.read_csv(stats_csv)
    ids_with_10m = set(stats.loc[stats["price_delta_pct_10m"].notna(), "id"])
    records = {r["id"]: r for r in load_records(dataset_json)}
    step = pd.Timedelta(minutes=BAR_MINUTES)

    n_cross = 0
    n_total = 0
    for rid in ids_with_10m:
        rec = records.get(rid)
        if rec is None:
            continue
        bars = load_or_fetch(rec)
        bars = to_exchange_hours(bars, get_exch(rec["ticker"]))
        if bars is None or bars.empty:
            continue
        news_dt = rec["news_dt"]
        slot = pd.Timestamp(news_dt).floor(f"{BAR_MINUTES}min")
        pre = bars[bars.index < slot]
        post = bars[bars.index >= slot + step].head(2)
        if len(post) < 2:
            continue
        n_total += 1
        last_post_date = post.index[-1].date()
        if pre.empty or last_post_date > pre.index[-1].date():
            n_cross += 1
    return n_cross, n_total


def main() -> int:
    print("[group_sensitivity] lade Korpus (wie train.py) ...")
    items = build_labeled()
    print(f"[group_sensitivity] items={len(items)}")

    raw = {r["id"]: r for r in json.loads(Path(DATASET_JSON).read_text())}
    id_to_ticker = {rid: r.get("ticker", "") for rid, r in raw.items()}

    trainval, test = stratified_holdout(items, test_frac=0.15, seed=0)
    train_tickers = [id_to_ticker[it["id"]] for it in trainval]
    test_tickers = [id_to_ticker[it["id"]] for it in test]
    overlap = holdout_ticker_overlap(train_tickers, test_tickers)
    print(f"[group_sensitivity] holdout: train={len(trainval)} test={len(test)} "
          f"ticker_overlap={overlap * 100:.2f}%")

    print("[group_sensitivity] embedde trainval-Texte (einmaliges Laden von Tokenizer/Modell) ...")
    Xtr = _embed_all([it["text"] for it in trainval])
    ytr = [it["label"] for it in trainval]
    fold_scores = group_cv_macro_f1(Xtr, ytr, train_tickers, n_splits=5)
    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores))
    print(f"[group_sensitivity] group_cv fold_scores={[round(s, 4) for s in fold_scores]}")
    print(f"[group_sensitivity] group_cv_macro_f1={mean_f1:.4f}±{std_f1:.4f}")

    print("[group_sensitivity] cross-session-Anteil 10-Min-Fenster ...")
    n_cross, n_total = cross_session_share_10m()
    cross_pct = (n_cross / n_total * 100.0) if n_total else 0.0
    print(f"[group_sensitivity] cross_session_10m={n_cross}/{n_total} ({cross_pct:.2f}%)")

    rows = [
        ("group_cv_macro_f1_mean", f"{mean_f1:.4f}"),
        ("group_cv_macro_f1_std", f"{std_f1:.4f}"),
        ("group_cv_fold_scores", ";".join(f"{s:.4f}" for s in fold_scores)),
        ("holdout_ticker_overlap_pct", f"{overlap * 100:.4f}"),
        ("cross_session_10m_n", str(n_cross)),
        ("cross_session_10m_total", str(n_total)),
        ("cross_session_10m_pct", f"{cross_pct:.4f}"),
    ]
    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["kennzahl", "wert"]).to_csv(out_path, index=False)
    print(f"[group_sensitivity] wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
