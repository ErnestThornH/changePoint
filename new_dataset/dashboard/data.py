"""Pure data/format helpers for the new-dataset Gradio dashboard. No gradio import;
heavy modules (torch/transformers/sentiment/sklearn) are lazy inside functions so these
helpers unit-test without the ML stack."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from new_dataset.pipeline.labels import label_3class

DATASET_JSON = "new_dataset/adhoc_functioning.json"
STATS_CSV = "new_dataset/event_window_stats.csv"
CHARTS_DIR = "new_dataset/charts"
SCATTER_DIR = "new_dataset/scatter"
EVAL_DIR = "new_dataset/model_eval"
RESULTS_CSV = "new_dataset/model_results.csv"
STORE_DIR = "new_dataset/model_store"

REACTION_DE = {"up": "Auf", "down": "Ab", "flat": "Flat"}

FINDING = {
    "n": 527, "dist": {"up": 307, "down": 190, "flat": 30},
    "best_model": "logreg_embed", "best_test_macro_f1": 0.482, "majority_macro_f1": 0.247,
    "spearman": 0.44,
    "text": ("Befund: Das Text-Sentiment (Volltext) trägt ein moderates, hochsignifikantes "
             "Richtungssignal für die 10-Minuten-Reaktion (Spearman ρ≈0,44); die exakte "
             "3-Klassen-Zuordnung bleibt schwer: logreg_embed schlägt die Majority-Baseline "
             "(Macro-F1 0,482 vs. 0,247), die seltene Flat-Klasse bleibt unzuverlässig."),
}

OVERVIEW_COLS = ["id", "published_local", "ticker", "company", "reaction",
                 "sofort_delta_pct", "price_delta_pct_10m", "link"]

KPI_DISPLAY = [
    ("sofort_delta_pct", "Sofort-Δ Preis"),
    ("price_delta_pct_5m", "Δ Preis 5min"), ("price_delta_pct_10m", "Δ Preis 10min"),
    ("price_delta_pct_15m", "Δ Preis 15min"), ("price_delta_pct_30m", "Δ Preis 30min"),
    ("volume_delta_pct_5m", "Δ Volumen 5min"), ("volume_delta_pct_10m", "Δ Volumen 10min"),
    ("volume_delta_pct_15m", "Δ Volumen 15min"), ("volume_delta_pct_30m", "Δ Volumen 30min"),
    ("vola_pre_1h_pct", "Vola 1h vor"), ("vola_post_1h_pct", "Vola 1h nach"),
]

RESULTS_HEADERS = {"model": "Modell", "cv_macro_f1_mean": "CV Macro-F1",
                   "cv_macro_f1_std": "CV Std", "test_macro_f1": "Test Macro-F1",
                   "test_accuracy": "Test Accuracy", "majority_baseline": "Majority-Baseline"}


def _is_missing(v) -> bool:
    return v is None or v == "" or pd.isna(v)


def _pct(v) -> str:
    return "–" if _is_missing(v) else f"{float(v):+.2f} %"


def _json_safe(value):
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _json_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """NaN/NaT → None. Gradio embeds values in a config JSON; NaN is not JSON-compliant and
    makes the frontend fail to hydrate (blank page). astype(object) so None survives."""
    return df.astype(object).where(df.notna(), None)


def load_overview(dataset_json: str = DATASET_JSON, stats_csv: str = STATS_CSV) -> pd.DataFrame:
    """Join records (text/meta) with stats on `id`; add reaction class from 10-min delta."""
    records = json.loads(Path(dataset_json).read_text())
    recs = pd.DataFrame([{"id": r["id"], "ticker": r.get("ticker"), "company": r.get("company"),
                          "published_local": r.get("published_local"), "link": r.get("link")}
                         for r in records])
    stats = pd.read_csv(stats_csv)[["id", "price_delta_pct_10m", "sofort_delta_pct"]]
    df = recs.merge(stats, on="id", how="left")
    df["reaction"] = df["price_delta_pct_10m"].map(label_3class)
    df = df[OVERVIEW_COLS].sort_values("published_local", ascending=False).reset_index(drop=True)
    return _json_safe_df(df)


def overview_display(df: pd.DataFrame) -> pd.DataFrame:
    """Anzeige-Sicht der Übersicht ohne technische id-Spalte (Menschen brauchen sie nicht;
    der Zeilenklick löst die id über den Zeilenindex des ungefilterten DataFrames auf)."""
    return df.drop(columns=["id"])


def filter_overview(df: pd.DataFrame, query: str = "", reaction: str = "alle",
                    year: str = "alle", venue: str = "alle") -> pd.DataFrame:
    out = df
    q = (query or "").strip().lower()
    if q:
        comp = out["company"].fillna("").astype(str).str.lower()
        tick = out["ticker"].fillna("").astype(str).str.lower()
        out = out[comp.str.contains(q) | tick.str.contains(q)]
    if reaction and reaction != "alle":
        out = out[out["reaction"] == reaction]
    if year and year != "alle":
        out = out[out["published_local"].fillna("").astype(str).str.slice(0, 4) == str(year)]
    if venue and venue != "alle":
        out = out[out["ticker"].fillna("").astype(str).str.split(".").str[-1] == venue]
    return out.reset_index(drop=True)


def filter_options(df: pd.DataFrame) -> dict:
    years = sorted({str(p)[:4] for p in df["published_local"] if p}, reverse=True)
    venues = sorted({str(t).split(".")[-1] for t in df["ticker"] if t and "." in str(t)})
    return {"years": years, "venues": venues}


def get_detail(record_id: str, dataset_json: str = DATASET_JSON, stats_csv: str = STATS_CSV,
               charts_dir: str = CHARTS_DIR) -> dict:
    records = {r["id"]: r for r in json.loads(Path(dataset_json).read_text())}
    rec = records.get(record_id)
    if not rec:
        return {"error": f"Unbekannte ID: {record_id}"}
    stats = pd.read_csv(stats_csv)
    srow = stats[stats["id"] == record_id]
    kpis = {k: _json_safe(v) for k, v in srow.iloc[0].to_dict().items()} if len(srow) else {}
    chart = Path(charts_dir) / f"{record_id}.png"
    return {"id": record_id, "company": rec.get("company"), "ticker": rec.get("ticker"),
            "isin": rec.get("isin"), "title": rec.get("title"), "summary": rec.get("summary"),
            "link": rec.get("link"), "published_local": rec.get("published_local"),
            "kpis": kpis, "chart_path": str(chart) if chart.exists() else None}


def detail_header(detail: dict) -> str:
    if detail.get("error"):
        return f"**{detail['error']}**"
    company = detail.get("company") or detail.get("title", "")
    date = (detail.get("published_local") or "")[:10]
    parts = [f"### {company}",
             f"`{detail.get('ticker', '')}` · {date} · ISIN {detail.get('isin', '')}"]
    title = (detail.get("title") or "").strip()
    if title and title != company:
        parts.append(f"**{title}**")
    summary = (detail.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    link = (detail.get("link") or "").strip()
    if link:
        parts.append(f"[→ Original-Meldung]({link})")
    return "\n\n".join(parts)


def kpi_table(kpis: dict) -> pd.DataFrame:
    rows = [[label, _pct(kpis.get(col))] for col, label in KPI_DISPLAY]
    return pd.DataFrame(rows, columns=["Kennzahl", "Wert"])


def model_results_table(results_csv: str = RESULTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    best = df["test_macro_f1"].idxmax()
    df["model"] = [("★ " if i == best else "") + str(m) for i, m in zip(df.index, df["model"])]
    df = df.rename(columns=RESULTS_HEADERS)
    return _json_safe_df(df)


def confusion_path(model: str, eval_dir: str = EVAL_DIR) -> str | None:
    p = Path(eval_dir) / f"confusion_{model}.png"
    return str(p) if p.exists() else None


def scatter_path(minutes: str, scatter_dir: str = SCATTER_DIR) -> str | None:
    p = Path(scatter_dir) / f"scatter_{minutes}min.png"
    return str(p) if p.exists() else None


def live_sentiment(text: str) -> dict:
    """German-FinBERT sentiment of one text. Lazy heavy imports."""
    from text_clean import clean
    cleaned = clean(text or "", "")
    if not cleaned.strip():
        return {"error": "Bitte Ad-hoc-Text einfügen."}
    from sentiment_models import SentimentScorer
    label, score = SentimentScorer("gfinbert").score([cleaned])[0]
    return {"label": label, "score": round(float(score), 4)}


def live_reaction(text: str, store_dir: str = STORE_DIR) -> list[dict]:
    """Predicted 10-min reaction from the persisted best model. Lazy import of persist."""
    from text_clean import clean
    cleaned = clean(text or "", "")
    if not cleaned.strip():
        return [{"note": "Bitte Ad-hoc-Text einfügen."}]
    from new_dataset.pipeline import persist
    return persist.predict([cleaned], out_dir=store_dir)
