"""Recompute sentiment<->reaction correlation on the n=527 corpus (Auswertung 5.3).
Scores each ad-hoc text with pretrained German-FinBERT (signed = P(pos)-P(neg)),
joins on id to the reaction KPIs, writes a Spearman table + a scatter figure."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from text_clean import clean_fulltext
from sentiment_models import SentimentScorer

DATASET = "new_dataset/adhoc_functioning.json"
STATS = "new_dataset/event_window_stats.csv"
KPIS = ["price_delta_pct_5m", "price_delta_pct_10m", "price_delta_pct_15m",
        "price_delta_pct_30m", "price_delta_pct_60m", "sofort_delta_pct"]
FIG = Path("new_dataset/figures/f13_sentiment_price.png")
TBL = Path("new_dataset/figures/tables/tbl_sentiment_corr.csv")


def build_frame() -> pd.DataFrame:
    records = {r["id"]: r for r in json.loads(Path(DATASET).read_text())}
    stats_df = pd.read_csv(STATS)
    rows = []
    for _, row in stats_df.iterrows():
        rec = records.get(row["id"])
        if rec is None:
            continue
        rows.append({"id": row["id"],
                     "text": clean_fulltext(rec.get("title", ""), rec.get("content", "")),
                     **{k: row.get(k) for k in KPIS}})
    return pd.DataFrame(rows)


def main() -> None:
    df = build_frame()
    scorer = SentimentScorer("gfinbert")
    res = scorer.score(df["text"].tolist())
    df["sentiment"] = [s for _, s in res]
    df["sent_label"] = [l for l, _ in res]
    out = []
    for k in KPIS:
        d = df[["sentiment", k]].dropna()
        r, p = stats.spearmanr(d["sentiment"], d[k])
        out.append({"kpi": k, "spearman_r": round(float(r), 4),
                    "spearman_p": "< 0.001" if p < 0.001 else round(float(p), 4),
                    "n": len(d)})
    pd.DataFrame(out).to_csv(TBL, index=False)
    d = df[["sentiment", "price_delta_pct_10m"]].dropna()
    r10, p10 = stats.spearmanr(d["sentiment"], d["price_delta_pct_10m"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(d["sentiment"], d["price_delta_pct_10m"], s=10, alpha=0.5)
    m, b = np.polyfit(d["sentiment"], d["price_delta_pct_10m"], 1)
    xs = np.linspace(d["sentiment"].min(), d["sentiment"].max(), 100)
    ax.plot(xs, m * xs + b, lw=1.0, color="C1",
            label=f"lineare Regression (Fit im linearen Raum, Steigung {m:.1f})")
    ax.set_xlabel("Sentiment-Score (P(pos) - P(neg))")
    ax.set_ylabel("Kursänderung 10 Min (%, symlog)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Sentiment vs. Kurs-Δ (10 Min) — Spearman ρ = {r10:.3f}")
    fig.tight_layout(); fig.savefig(FIG, dpi=130); plt.close(fig)
    d10 = df[["sent_label", "sentiment", "price_delta_pct_10m"]].dropna()
    rows_lab = []
    for lab in ["negative", "neutral", "positive"]:
        sub = d10[d10["sent_label"] == lab]["price_delta_pct_10m"]
        rows_lab.append({"label": lab, "n": len(sub), "mean": round(sub.mean(), 4),
                         "median": round(sub.median(), 4)})
    nn = d10[d10["sent_label"] != "neutral"]
    hit = ((nn["sent_label"] == "positive") & (nn["price_delta_pct_10m"] > 0) |
           (nn["sent_label"] == "negative") & (nn["price_delta_pct_10m"] < 0)).mean()
    pear = d10["sentiment"].corr(d10["price_delta_pct_10m"])
    rows_lab.append({"label": "hit_rate_nonneutral", "n": len(nn), "mean": round(float(hit), 4), "median": None})
    rows_lab.append({"label": "pearson_10m", "n": len(d10), "mean": round(float(pear), 4), "median": None})
    pd.DataFrame(rows_lab).to_csv("new_dataset/figures/tables/tbl_sentiment_labels.csv", index=False)
    print(pd.DataFrame(rows_lab).to_string(index=False))
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
