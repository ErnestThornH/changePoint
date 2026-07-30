"""Wählt deterministisch drei Fallbeispiele für Kapitel 5.5.

Der Betreuer wünschte konkrete Beispiele: eine Meldung mit eindeutig positivem
Text und passender Kursreaktion, eine mit eindeutig negativem Text und passender
Reaktion sowie einen Fehlschlag (eindeutiges Sentiment, aber gegenläufige
Reaktion). Die Auswahl ist reine argmax/argmin-Logik ohne Zufall: German-FinBERT
bewertet jeden Meldungstext (signed = P(pos) - P(neg)), die 10-Minuten-Reaktion
stammt aus ``event_window_stats.csv``. Gewählt werden nur Meldungen, für die
bereits ein ±24-h-Chart unter ``new_dataset/charts/<id>.png`` existiert; dessen
Kopie wird als Abbildungen 16–20 (je Typ tagsüber und außerbörslich) abgelegt.
"""
from __future__ import annotations

import shutil
from datetime import time as dt_time
from pathlib import Path

import pandas as pd

from new_dataset.sentiment_reaction import build_frame
from new_dataset.pipeline.dataset import load_records
from new_dataset.pipeline.eodhd_prices import load_or_fetch
from new_dataset.pipeline.exchanges import get_exch, to_exchange_hours
from sentiment_models import SentimentScorer

CHARTS = Path("new_dataset/charts")
FIGURES = Path("new_dataset/figures")
DATASET = "new_dataset/adhoc_functioning.json"
CLEAR = 0.5   # Schwelle für ein "eindeutiges" Sentiment (|signed|)
OPPOSITE = 2.0  # Mindest-Gegenreaktion in Prozent für den Fehlschlag

TARGETS = {
    "pos": "f16_beispiel_pos.png",
    "neg_tag": "f17_beispiel_neg_tag.png",
    "neg": "f18_beispiel_neg.png",
    "fail_tag": "f19_beispiel_fehlschlag_tag.png",
    "fail": "f20_beispiel_fehlschlag.png",
}


def _has_chart(_id: str) -> bool:
    return (CHARTS / f"{_id}.png").exists()


def scored_frame() -> pd.DataFrame:
    """id, ticker, published_utc, title, sentiment (signed), sent_label, delta."""
    import json

    df = build_frame()
    scorer = SentimentScorer("gfinbert")
    res = scorer.score(df["text"].tolist())
    df["sentiment"] = [s for _, s in res]
    df["sent_label"] = [l for l, _ in res]
    df["delta"] = df["price_delta_pct_10m"]

    meta = {r["id"]: r for r in json.loads(Path(DATASET).read_text())}
    df["ticker"] = df["id"].map(lambda i: meta.get(i, {}).get("ticker", ""))
    df["published"] = df["id"].map(lambda i: (meta.get(i, {}).get("published_utc", "") or "")[:10])
    df["title"] = df["id"].map(lambda i: meta.get(i, {}).get("title", ""))

    df = df.dropna(subset=["sentiment", "delta"]).copy()
    df["chart"] = df["id"].map(_has_chart)
    df = df[df["chart"]].copy()

    recs = {r["id"]: r for r in load_records(DATASET)}

    def _intraday(_id: str) -> bool:
        rec = recs.get(_id)
        if rec is None:
            return False
        return _is_intraday(rec, get_exch(rec["ticker"]))

    df["intraday"] = df["id"].map(_intraday)
    return df


def pick(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Deterministische Auswahl der drei Fallbeispiele (leere Series-Fallbacks)."""
    pos_pool = df[(df["sent_label"] == "positive") | (df["sentiment"] >= CLEAR)]
    neg_pool = df[(df["sent_label"] == "negative") | (df["sentiment"] <= -CLEAR)]

    picks: dict[str, pd.Series] = {}
    if not pos_pool.empty:
        picks["pos"] = _quiet_pos_pick(pos_pool)
    if not neg_pool.empty:
        picks["neg"] = neg_pool.loc[neg_pool["delta"].idxmin()]

    intra = df[df["intraday"]]
    neg_intra = intra[(intra["sent_label"] == "negative") | (intra["sentiment"] <= -CLEAR)]
    if not neg_intra.empty:
        picks["neg_tag"] = neg_intra.loc[neg_intra["delta"].idxmin()]

    clear = df[df["sentiment"].abs() >= CLEAR].copy()
    # Gegenreaktion: positiv gemeint aber gefallen (bzw. umgekehrt). opp > 0 = gegenläufig.
    clear["opp"] = -clear["sentiment"].apply(lambda s: 1.0 if s > 0 else -1.0) * clear["delta"]
    fail_pool = clear[clear["opp"] >= OPPOSITE]
    if not fail_pool.empty:
        picks["fail"] = fail_pool.loc[fail_pool["opp"].idxmax()]

    fail_intra = clear[clear["intraday"]] if "intraday" in clear else clear.iloc[0:0]
    fail_intra = fail_intra[fail_intra["opp"] >= OPPOSITE]
    if not fail_intra.empty:
        picks["fail_tag"] = fail_intra.loc[fail_intra["opp"].idxmax()]
    return picks


# --- Ersatz-Fallbeispiel 1: positive Meldung mit ruhigem Vorlauf --------------

POS_SCORE_MIN = 0.9    # Mindest-Signed-Score für "eindeutig positiv"
POS_DELTA_MIN = 5.0    # Mindest-10-Min-Reaktion in Prozent
DRIFT_STRICT = 1.0     # zulässige Vorlauf-Drift in Prozent (Betrag)
DRIFT_RELAX = 1.5      # gelockerte Schwelle, falls < n Kandidaten
DRIFT_BARS = 6         # 6 Bars * 5 Min = 30 Minuten Vorlauf


def _is_intraday(rec: dict, exch: dict) -> bool:
    """True, wenn die Meldezeit (lokal) innerhalb der Handelssession liegt."""
    (oh, om), (ch, cm) = exch["open"], exch["close"]
    t = pd.Timestamp(rec["news_dt"]).time()
    return dt_time(oh, om) <= t < dt_time(ch, cm)


def _pre_news_drift(rec: dict) -> float | None:
    """Kursdrift der 30 Min VOR der Meldung in Prozent, oder None wenn nicht messbar.

    Lade-Logik wie ``random_baseline``: ``load_or_fetch`` + ``to_exchange_hours``.
    Letzte Bar strikt vor der Meldung = t-1; Referenz 6 Bars davor = t-7.
    Drift = Close(t-1) / Close(t-7) - 1 (in %).
    """
    bars = load_or_fetch(rec)
    bars = to_exchange_hours(bars, get_exch(rec["ticker"]))
    if bars is None or bars.empty:
        return None
    pre = bars[bars.index < pd.Timestamp(rec["news_dt"])]
    close = pre["Close"].dropna()
    if len(close) < DRIFT_BARS + 1:
        return None
    last = float(close.iloc[-1])
    earlier = float(close.iloc[-(DRIFT_BARS + 1)])
    if earlier == 0:
        return None
    return (last / earlier - 1.0) * 100.0


def _quiet_pos_pick(pos_pool: pd.DataFrame) -> pd.Series:
    """Argmax-Delta unter den positiven Meldungen mit ruhigem Vorlauf.

    Wendet dasselbe Ruhe-Kriterium wie ``positive_candidates`` an (Score >= 0,9,
    Delta >= 5 %, |Vorlauf-Drift| <= 1 %), damit die (a)-Auswahl reproduzierbar
    dasselbe Beispiel liefert. Fällt auf den reinen Delta-Argmax zurück, falls
    kein ruhiger Kandidat existiert.
    """
    strong = pos_pool[(pos_pool["sentiment"] >= POS_SCORE_MIN)
                      & (pos_pool["delta"] >= POS_DELTA_MIN)]
    if not strong.empty:
        recs = {r["id"]: r for r in load_records(DATASET)}
        quiet_intra, quiet_any = [], []
        for _id in strong["id"]:
            rec = recs.get(_id)
            if rec is None:
                continue
            drift = _pre_news_drift(rec)
            if drift is None or abs(drift) > DRIFT_STRICT:
                continue
            quiet_any.append(_id)
            if _is_intraday(rec, get_exch(rec["ticker"])):
                quiet_intra.append(_id)
        # Intraday bevorzugen (der Ruhe-Sprung ist im Handelsband sichtbar),
        # sonst außerbörsliche Ruhe-Kandidaten; jeweils Delta-Argmax.
        for ids in (quiet_intra, quiet_any):
            quiet = strong[strong["id"].isin(ids)]
            if not quiet.empty:
                return quiet.loc[quiet["delta"].idxmax()]
    return pos_pool.loc[pos_pool["delta"].idxmax()]


def positive_candidates(n: int = 4) -> pd.DataFrame:
    """Kandidaten für ein positives Fallbeispiel mit ruhigem Vorlauf.

    Kriterien: Label positiv, Signed-Score >= 0,9, ``price_delta_pct_10m`` >= +5,
    Chart existiert, |Vorlauf-Drift| <= 1,0 % (bei < n Treffern auf 1,5 % gelockert).
    Intraday-Meldungen werden bevorzugt; fehlen sie, wird mit außerbörslichen
    aufgefüllt und gekennzeichnet. Rückgabe: Top-n nach Delta absteigend.
    """
    df = scored_frame()
    pool = df[(df["sent_label"] == "positive")
              & (df["sentiment"] >= POS_SCORE_MIN)
              & (df["delta"] >= POS_DELTA_MIN)].copy()

    recs = {r["id"]: r for r in load_records(DATASET)}
    rows = []
    for _, r in pool.iterrows():
        rec = recs.get(r["id"])
        if rec is None:
            continue
        drift = _pre_news_drift(rec)
        if drift is None:
            continue
        exch = get_exch(rec["ticker"])
        rows.append({
            "id": r["id"], "ticker": r["ticker"],
            "when": pd.Timestamp(rec["news_dt"]).strftime("%Y-%m-%d %H:%M"),
            "title": str(r["title"]), "score": float(r["sentiment"]),
            "delta": float(r["delta"]), "drift": float(drift),
            "intraday": _is_intraday(rec, exch),
            "chart": str(CHARTS / f"{r['id']}.png"),
        })
    cand = pd.DataFrame(rows)

    threshold, relaxed = DRIFT_STRICT, False
    quiet = cand[cand["drift"].abs() <= DRIFT_STRICT] if not cand.empty else cand
    if len(quiet) < n:
        threshold, relaxed = DRIFT_RELAX, True
        quiet = cand[cand["drift"].abs() <= DRIFT_RELAX] if not cand.empty else cand

    # Intraday bevorzugen, dann außerbörslich auffüllen; jeweils nach Delta sortiert.
    intra = quiet[quiet["intraday"]].sort_values("delta", ascending=False)
    after = quiet[~quiet["intraday"]].sort_values("delta", ascending=False)
    picked = pd.concat([intra, after]).head(n)
    picked = picked.sort_values("delta", ascending=False).reset_index(drop=True)

    print("=== Positive Fallbeispiel-Kandidaten (ruhiger Vorlauf) ===")
    print(f"Pool: {len(pool)} positiv/Score>= {POS_SCORE_MIN}/Delta>= {POS_DELTA_MIN} %  "
          f"| mit messbarer Drift: {len(cand)}  "
          f"| Ruhe-Schwelle |Drift|<= {threshold:.1f} %"
          + ("  (GELOCKERT von 1,0 auf 1,5 %)" if relaxed else "")
          + f"  -> {len(quiet)} Kandidaten")
    if len(intra) < n:
        print(f"Hinweis: nur {len(intra)} Intraday-Kandidaten -> mit "
              f"{max(0, min(len(after), n - len(intra)))} außerbörslichen aufgefüllt.")
    print()
    for i, row in picked.iterrows():
        print(f"[{i + 1}] id={row['id']}  ticker={row['ticker']}  {row['when']}")
        print(f"    titel : {row['title'][:80]}")
        print(f"    score={row['score']:.4f}  delta_10m={row['delta']:+.2f} %  "
              f"vorlauf_drift={row['drift']:+.2f} %  "
              f"intraday={'ja' if row['intraday'] else 'nein'}")
        print(f"    chart : {row['chart']}")
    if picked.empty:
        print("(keine Kandidaten)")
    return picked


def _report(kind: str, row: pd.Series) -> None:
    print(f"[{kind}] id={row['id']}  ticker={row['ticker']}  published={row['published']}")
    print(f"       titel : {str(row['title'])[:80]}")
    print(f"       sentiment: {row['sent_label']} ({row['sentiment']:+.4f})   "
          f"delta_10m: {row['delta']:+.2f} %")


def main() -> dict[str, pd.Series]:
    df = scored_frame()
    picks = pick(df)
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = {"pos": "POSITIV", "neg_tag": "NEGATIV (TAG)", "neg": "NEGATIV",
              "fail_tag": "FEHLSCHLAG (TAG)", "fail": "FEHLSCHLAG"}
    for kind in ("pos", "neg_tag", "neg", "fail_tag", "fail"):
        row = picks.get(kind)
        if row is None:
            print(f"[{labels[kind]}] kein Kandidat gefunden")
            continue
        _report(labels[kind], row)
        dst = FIGURES / TARGETS[kind]
        shutil.copyfile(CHARTS / f"{row['id']}.png", dst)
        print(f"       chart -> {dst}")
    ids = [str(r["id"]) for r in picks.values()]
    if len(set(ids)) != len(ids):
        print("WARNUNG: doppelte Auswahl über die Fallbeispiel-Typen:", ids)
    return picks


if __name__ == "__main__":
    main()
