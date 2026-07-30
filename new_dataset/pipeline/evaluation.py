"""Leakage-safe evaluation: stratified hold-out, stratified k-fold, metrics, CV harness."""
from __future__ import annotations

import numpy as np


def stratified_holdout(items, test_frac=0.15, seed=0):
    """(trainval, test) stratified by item['label']; deterministic."""
    rng = np.random.RandomState(seed)
    by = {}
    for i, it in enumerate(items):
        by.setdefault(it["label"], []).append(i)
    test_idx = []
    for lab, idx in by.items():
        idx = list(idx); rng.shuffle(idx)
        ntest = int(round(len(idx) * test_frac))
        test_idx += idx[:ntest]
    test_set = set(test_idx)
    test = [items[i] for i in range(len(items)) if i in test_set]
    trainval = [items[i] for i in range(len(items)) if i not in test_set]
    return trainval, test


def stratified_kfold_indices(labels, k=5, seed=0):
    """Yield (train_idx, test_idx) for k stratified folds; every index test once."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    fold_of = np.empty(len(labels), dtype=int)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]; rng.shuffle(idx)
        for i, j in enumerate(idx):
            fold_of[j] = i % k
    for f in range(k):
        yield np.where(fold_of != f)[0], np.where(fold_of == f)[0]


def _f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec) if (prec + rec) else 0.0), prec, rec


def metrics(y_true, y_pred, labels=("up", "down", "flat")) -> dict:
    """macro-F1 (headline), accuracy, per-class P/R/F1, confusion (true->pred counts)."""
    y_true, y_pred = list(y_true), list(y_pred)
    conf = {a: {b: 0 for b in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        if t in conf and p in conf[t]:
            conf[t][p] += 1
    per = {}
    f1s = []
    for lab in labels:
        tp = conf[lab][lab]
        fp = sum(conf[o][lab] for o in labels if o != lab)
        fn = sum(conf[lab][o] for o in labels if o != lab)
        f1, prec, rec = _f1(tp, fp, fn)
        per[lab] = {"precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else float("nan")
    return {"macro_f1": float(np.mean(f1s)), "accuracy": acc, "per_class": per, "confusion": conf}


def evaluate_cv(items, fit_predict, k=5, seeds=(0, 1, 2), labels=("up", "down", "flat")) -> dict:
    """Stratified k-fold CV; per (seed,fold) call fit_predict(train_items, test_items)->preds;
    aggregate out-of-fold macro-F1 with mean±std across seeds."""
    y = [it["label"] for it in items]
    per_seed = []
    for seed in seeds:
        oof_true, oof_pred = [], []
        for tr, te in stratified_kfold_indices(y, k, seed):
            train = [items[i] for i in tr]; test = [items[i] for i in te]
            preds = fit_predict(train, test)
            oof_true += [items[i]["label"] for i in te]
            oof_pred += list(preds)
        per_seed.append(metrics(oof_true, oof_pred, labels)["macro_f1"])
    return {"macro_f1_mean": float(np.mean(per_seed)),
            "macro_f1_std": float(np.std(per_seed)), "seeds": list(seeds)}
