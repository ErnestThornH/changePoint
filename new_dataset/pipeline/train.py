"""Orchestrate the 3-class reaction-prediction comparison: stratified hold-out test (once) +
stratified 5-fold CV (mean±std) per model, leakage-safe. Run: python -m new_dataset.pipeline.train"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from new_dataset.pipeline.labels import build_labeled
from new_dataset.pipeline.evaluation import stratified_holdout, evaluate_cv, metrics
from new_dataset.pipeline.models import MODELS, LABELS

RESULTS_CSV = "new_dataset/model_results.csv"
EVAL_DIR = "new_dataset/model_eval"


def _confusion_png(conf: dict, title: str, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    M = np.array([[conf[a][b] for b in LABELS] for a in LABELS])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS); ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Vorhersage"); ax.set_ylabel("Wahr"); ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(M[i, j]), ha="center", va="center")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def main(k: int = 5, seeds=(0, 1, 2), resume: bool = False) -> int:
    items = build_labeled()
    dist = Counter(it["label"] for it in items)
    print(f"[train] labeled={len(items)} dist={dict(dist)}")
    trainval, test = stratified_holdout(items, test_frac=0.15, seed=0)
    maj = Counter(it["label"] for it in trainval).most_common(1)[0][0]
    majority_baseline = sum(1 for it in test if it["label"] == maj) / len(test)

    rows = []
    test_metrics: dict[str, dict] = {}
    done = set()
    if resume and Path(RESULTS_CSV).exists():
        rows = pd.read_csv(RESULTS_CSV).to_dict("records")
        done = {r["model"] for r in rows if pd.notna(r.get("cv_macro_f1_mean"))}
        print(f"[train] resume: überspringe {sorted(done)}")
    for name, fit_predict in MODELS.items():
        if name in done:
            continue
        try:
            cv = evaluate_cv(trainval, fit_predict, k=k, seeds=seeds, labels=tuple(LABELS))
            preds = fit_predict(trainval, test)              # final fit on all trainval
            m = metrics([t["label"] for t in test], preds, tuple(LABELS))
            test_metrics[name] = m
            _confusion_png(m["confusion"], f"{name} (Test)", f"{EVAL_DIR}/confusion_{name}.png")
            rows.append({"model": name, "cv_macro_f1_mean": round(cv["macro_f1_mean"], 4),
                         "cv_macro_f1_std": round(cv["macro_f1_std"], 4),
                         "test_macro_f1": round(m["macro_f1"], 4),
                         "test_accuracy": round(m["accuracy"], 4),
                         "majority_baseline": round(majority_baseline, 4)})
            print(f"[train] {name}: cv_f1={cv['macro_f1_mean']:.3f}±{cv['macro_f1_std']:.3f} "
                  f"test_f1={m['macro_f1']:.3f} test_acc={m['accuracy']:.3f}")
        except Exception as e:
            print(f"[train] {name} FAILED: {type(e).__name__}: {e}")
            rows.append({"model": name, "cv_macro_f1_mean": None, "cv_macro_f1_std": None,
                         "test_macro_f1": None, "test_accuracy": None,
                         "majority_baseline": round(majority_baseline, 4)})
        # nach jedem Modell speichern — ein abgebrochener Fine-Tune kostet nicht alles
        Path(RESULTS_CSV).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    if test_metrics:
        cv_mean = {r["model"]: r["cv_macro_f1_mean"] for r in rows if r["model"] in test_metrics}
        best = max(cv_mean, key=cv_mean.get)  # Wahl per CV; Test bleibt reine Berichtsgroesse
        Path(f"{EVAL_DIR}/perclass_best.json").write_text(
            json.dumps({"model": best, **test_metrics[best]}))
        print(f"[train] perclass_best={best}")
    print(f"[train] majority_baseline(test)={majority_baseline:.3f}; wrote {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(resume="--resume" in sys.argv))
