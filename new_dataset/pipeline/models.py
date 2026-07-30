"""Model spectrum for the 3-class reaction task. Each fit_predict(train, test) -> list[label].
Baselines + classical (on frozen German-FinBERT embeddings) + pretrained-sentiment + fine-tune."""
from __future__ import annotations

import numpy as np

BASE_MODEL = "scherrmann/GermanFinBERT_SC_Sentiment"
LABELS = ["up", "down", "flat"]
_SENT_TO_3 = {"positive": "up", "negative": "down", "neutral": "flat"}


def majority_fit_predict(train, test):
    from collections import Counter
    maj = Counter(t["label"] for t in train).most_common(1)[0][0]
    return [maj] * len(test)


def pretrained_sentiment_fit_predict(train, test):
    """Map pre-trained gfinbert label -> up/down/flat. No training (reference)."""
    from sentiment_models import SentimentScorer
    res = SentimentScorer("gfinbert").score([t["text"] for t in test])
    return [_SENT_TO_3.get(lab, "flat") for lab, _ in res]


def _embed(texts, batch=16):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(list(texts[i:i+batch]), truncation=True, max_length=256, padding=True,
                      return_tensors="pt").to(dev)
            out.append(model(**enc).last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 768))


def logreg_embed_fit_predict(train, test):
    from sklearn.linear_model import LogisticRegression
    Xtr = _embed([t["text"] for t in train]); Xte = _embed([t["text"] for t in test])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    clf.fit(Xtr, [t["label"] for t in train])
    return list(clf.predict(Xte))


def rf_embed_fit_predict(train, test):
    from sklearn.ensemble import RandomForestClassifier
    Xtr = _embed([t["text"] for t in train]); Xte = _embed([t["text"] for t in test])
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)
    clf.fit(Xtr, [t["label"] for t in train])
    return list(clf.predict(Xte))


def finetuned_finbert_fit_predict(train, test, epochs=8, lr=2e-4, val_frac=0.15, seed=0):
    """German-FinBERT, frozen backbone + 3-class head, AdamW, early stopping, grad-clip 1.0,
    class-weighted CE."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    lab2i = {l: i for i, l in enumerate(LABELS)}
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True).to(dev)
    for nm, p in model.named_parameters():
        if not nm.startswith("classifier"):
            p.requires_grad = False
    # class weights from train
    import numpy as _np
    counts = _np.array([sum(1 for t in train if t["label"] == l) for l in LABELS], dtype=float)
    w = torch.tensor((counts.sum() / (counts + 1e-9)) / 3.0, dtype=torch.float32, device=dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)

    # inner stratified val split for early stopping
    rng = _np.random.RandomState(seed)
    idx = list(range(len(train))); rng.shuffle(idx)
    nval = max(1, int(len(idx) * val_frac))
    val_i, tr_i = set(idx[:nval]), idx[nval:]
    Xtr = [train[i]["text"] for i in tr_i]; ytr = [lab2i[train[i]["label"]] for i in tr_i]
    Xval = [train[i]["text"] for i in val_i]; yval = [lab2i[train[i]["label"]] for i in val_i]

    def enc(texts):
        return tok(list(texts), truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)

    def val_loss():
        model.eval()
        with torch.no_grad():
            logits = model(**enc(Xval)).logits
            return float(lossf(logits, torch.tensor(yval, device=dev)).item())

    best, best_state, patience, bad = float("inf"), None, 2, 0
    for ep in range(epochs):
        model.train()
        perm = _np.random.RandomState(seed + ep).permutation(len(Xtr))
        for i in range(0, len(Xtr), 16):
            j = perm[i:i+16]
            opt.zero_grad()
            out = model(**enc([Xtr[k] for k in j]))
            loss = lossf(out.logits, torch.tensor([ytr[k] for k in j], device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = val_loss()
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        Xte = [t["text"] for t in test]
        for i in range(0, len(Xte), 16):
            logits = model(**enc(Xte[i:i+16])).logits
            preds += [LABELS[k] for k in logits.argmax(-1).cpu().numpy()]
    return preds


MODELS = {
    "majority": majority_fit_predict,
    "logreg_embed": logreg_embed_fit_predict,
    "rf_embed": rf_embed_fit_predict,
    "pretrained_sentiment": pretrained_sentiment_fit_predict,
    "finetuned_finbert": finetuned_finbert_fit_predict,
}
