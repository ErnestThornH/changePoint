"""Publication-quality statistics & figures suite over all ad-hoc news.

Pure helpers (descriptive_stats, ecdf) import no matplotlib. Figure functions lazily
import matplotlib with the Agg backend, mirroring scatter.py conventions. Each figure
returns (out_path, caption_de); it returns (None, caption) when data is missing/empty so
build_all can skip it gracefully. Captions and axis labels are German. build_all writes
PNGs to FIGURES_DIR, CSV tables to TABLES_DIR, and a JSON manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from new_dataset.pipeline.labels import label_3class

FIGURES_DIR = Path("new_dataset/figures")
TABLES_DIR = FIGURES_DIR / "tables"
STATS_CSV = "new_dataset/event_window_stats.csv"
DATASET_JSON = "new_dataset/adhoc_functioning.json"
RESULTS_CSV = "new_dataset/model_results.csv"
MANIFEST = FIGURES_DIR / "figures_manifest.json"
HORIZONS = [5, 10, 15, 30, 60]
SOURCE = "Eigene Darstellung"
REACTION_DE = {"up": "Auf", "down": "Ab", "flat": "Flat"}

KPI_COLS = (
    [f"price_delta_pct_{m}m" for m in HORIZONS]
    + [f"volume_delta_pct_{m}m" for m in HORIZONS]
    + ["sofort_delta_pct", "vola_pre_1h_pct", "vola_post_1h_pct"]
)


# --------------------------------------------------------------------------- #
# Pure helpers (no matplotlib)
# --------------------------------------------------------------------------- #
def descriptive_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Descriptive statistics per column: mean, median, std, min, max, iqr (q75-q25).
    Coerces to numeric and drops NaN per column. Indexed by column name."""
    rows = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        rows[c] = {
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "iqr": q75 - q25,
        }
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=["mean", "median", "std", "min", "max", "iqr"])


def ecdf(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF: returns (x_sorted, cum_share) with cum_share = (1..n)/n in (0,1]."""
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    x = np.sort(s.to_numpy())
    n = len(x)
    y = np.arange(1, n + 1) / n if n else np.array([])
    return x, y


# --------------------------------------------------------------------------- #
# Small internal utilities
# --------------------------------------------------------------------------- #
def _save(fig, out_path: str):
    import matplotlib.pyplot as plt
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _numcol(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def _month_series(records_or_df) -> pd.Series:
    """Extract a datetime series from records (published_local) or a df (published_utc)."""
    if isinstance(records_or_df, pd.DataFrame):
        col = "published_local" if "published_local" in records_or_df.columns else "published_utc"
        if col not in records_or_df.columns:
            return pd.Series(dtype="datetime64[ns]")
        return pd.to_datetime(records_or_df[col], errors="coerce", utc=True).dropna()
    vals = [r.get("published_local") or r.get("published_utc") for r in records_or_df]
    return pd.to_datetime(pd.Series(vals), errors="coerce", utc=True).dropna()


# --------------------------------------------------------------------------- #
# Figure functions
# --------------------------------------------------------------------------- #
def fig_meldungen_zeit(records_or_df, out_path: str):
    cap = "Anzahl der Ad-hoc-Meldungen pro Monat."
    dt = _month_series(records_or_df)
    if dt.empty:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = dt.dt.tz_localize(None).dt.to_period("M").value_counts().sort_index()
    labels = [str(p) for p in counts.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(counts)), counts.to_numpy(), color="#4C72B0")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Monat")
    ax.set_ylabel("Anzahl Meldungen")
    ax.set_title("Ad-hoc-Meldungen pro Monat")
    _save(fig, out_path)
    return out_path, cap


def fig_boerse_verteilung(records, out_path: str):
    cap = "Verteilung der Meldungen nach Börsenplatz (Ticker-Suffix)."
    tickers = ([r.get("ticker", "") for r in records]
               if not isinstance(records, pd.DataFrame)
               else records.get("ticker", pd.Series(dtype=str)).tolist())
    suffixes = [t.split(".", 1)[1] if isinstance(t, str) and "." in t else "(ohne)"
                for t in tickers if t]
    if not suffixes:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = pd.Series(suffixes).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(counts)), counts.to_numpy(), color="#4C72B0")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_xlabel("Börsenplatz")
    ax.set_ylabel("Anzahl Meldungen")
    ax.set_title("Meldungen nach Börsenplatz")
    _save(fig, out_path)
    return out_path, cap


def fig_textlaenge(records, out_path: str):
    cap = "Verteilung der Textlänge des bereinigten Meldungstexts (Titel + Volltext) in Zeichen."
    if isinstance(records, pd.DataFrame):
        return None, cap
    from text_clean import clean_fulltext
    lengths = [len(clean_fulltext(r.get("title") or "", r.get("content") or "")) for r in records]
    lengths = [x for x in lengths if x > 0]
    if not lengths:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Textlänge (Zeichen)")
    ax.set_ylabel("Anzahl Meldungen")
    ax.set_title("Verteilung der Textlänge")
    _save(fig, out_path)
    return out_path, cap


def fig_hist_price(df: pd.DataFrame, out_path: str):
    cap = "Verteilung der Kursänderung 10 Minuten nach der Meldung (Anzeige auf ±30 % begrenzt)."
    s = _numcol(df, "price_delta_pct_10m")
    if s.empty:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    disp = s.clip(-30, 30)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(disp, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlim(-30, 30)
    ax.set_xlabel("Kursänderung % (nach 10 min)")
    ax.set_ylabel("Anzahl Meldungen")
    ax.set_title("Verteilung Kurs-Δ (10 Min)")
    _save(fig, out_path)
    return out_path, cap


def fig_ecdf_price(df: pd.DataFrame, out_path: str):
    cap = "Kumulierte Verteilung des Betrags der Kursänderung 10 Minuten nach der Meldung."
    s = _numcol(df, "price_delta_pct_10m").abs()
    if s.empty:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = ecdf(s)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(x, y, where="post", color="#4C72B0")
    ax.set_xlabel("|Kursänderung| % (nach 10 min)")
    ax.set_ylabel("Anteil")
    ax.set_ylim(0, 1.02)
    ax.set_title("Kumulierte Verteilung |Kurs-Δ| (10 Min)")
    _save(fig, out_path)
    return out_path, cap


def fig_price_horizon(df: pd.DataFrame, out_path: str):
    cap = ("Median und Interquartilsbereich der Kursänderung je Zeithorizont "
           "(5 bis 60 Minuten).")
    horizons, med, q1, q3 = [], [], [], []
    for m in HORIZONS:
        s = _numcol(df, f"price_delta_pct_{m}m")
        if s.empty:
            continue
        horizons.append(m)
        med.append(s.median())
        q1.append(s.quantile(0.25))
        q3.append(s.quantile(0.75))
    if not horizons:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = range(len(horizons))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="grey", lw=0.6)
    ax.fill_between(x, q1, q3, color="#4C72B0", alpha=0.2,
                    label="Interquartilsbereich (Q1–Q3)")
    ax.plot(x, med, color="#4C72B0", marker="o", label="Median")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(m) for m in horizons])
    ax.set_xlabel("Horizont (Minuten)")
    ax.set_ylabel("Kursänderung (%)")
    ax.set_title("Median und Interquartilsbereich der Kursänderung je Horizont")
    ax.legend(loc="upper left", frameon=False)
    _save(fig, out_path)
    return out_path, cap


def fig_hist_volume(df: pd.DataFrame, out_path: str):
    cap = ("Verteilung der 10-Minuten-Volumenänderung (logarithmische Skala) "
           "mit markiertem Median.")
    s = _numcol(df, "volume_delta_pct_10m")
    if s.empty:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = s[s > 0]
    n_nonpos = int((s <= 0).sum())
    median = s.median()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xscale("log")
    bins = np.logspace(-1, 6, 29)
    ax.hist(pos, bins=bins, color="#55A868", edgecolor="white",
            label="Meldungen mit Volumenanstieg")
    if n_nonpos:
        # Anteil ohne Anstieg (<= 0 %) als separater grauer Balken links
        share_de = f"{n_nonpos / len(s) * 100:.1f}".replace(".", ",")
        ax.bar(10 ** -1.5, n_nonpos, width=10 ** -1.5 - 10 ** -2,
               align="center", color="#B0B0B0", edgecolor="white",
               label=f"Änderung ≤ 0 % ({share_de} %)")
    ax.axvline(median, color="#C44E52", lw=1.4, ls="--")
    de_median = f"{median:,.0f}".replace(",", ".")
    ax.annotate(f"Median {de_median} %", xy=(median, 0.6),
                xycoords=("data", "axes fraction"), xytext=(-8, 0),
                textcoords="offset points", ha="right", color="#C44E52",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("Volumenänderung nach 10 Minuten (%, log. Skala; grau: Werte ≤ 0 %)")
    ax.set_ylabel("Anzahl Meldungen")
    ax.set_title("Verteilung der 10-Minuten-Volumenänderung")
    ax.legend(loc="upper right", frameon=False)
    _save(fig, out_path)
    return out_path, cap


def fig_ecdf_volume(df: pd.DataFrame, out_path: str):
    cap = "Kumulierte Verteilung der Volumenänderung 10 Minuten nach der Meldung."
    s = _numcol(df, "volume_delta_pct_10m")
    if s.empty:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = ecdf(s)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(x, y, where="post", color="#55A868")
    ax.set_xscale("symlog")
    ax.set_xlabel("Volumenänderung % (nach 10 min, symlog)")
    ax.set_ylabel("Anteil")
    ax.set_ylim(0, 1.02)
    ax.set_title("Kumulierte Verteilung Volumen-Δ (10 Min)")
    _save(fig, out_path)
    return out_path, cap


def fig_volume_by_class(df: pd.DataFrame, out_path: str):
    cap = "Volumenänderung (10 Min) gruppiert nach Reaktionsklasse (Auf/Ab/Flat)."
    if "reaction" not in df.columns or "volume_delta_pct_10m" not in df.columns:
        return None, cap
    data, labels = [], []
    for lab in ("up", "down", "flat"):
        s = pd.to_numeric(df.loc[df["reaction"] == lab, "volume_delta_pct_10m"],
                          errors="coerce").dropna()
        if not s.empty:
            data.append(s.to_numpy())
            labels.append(REACTION_DE[lab])
    if not data:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_yscale("symlog")
    ax.set_xlabel("Reaktionsklasse")
    ax.set_ylabel("Volumenänderung % (symlog)")
    ax.set_title("Volumen-Δ je Reaktionsklasse")
    _save(fig, out_path)
    return out_path, cap


def fig_price_vs_volume(df: pd.DataFrame, out_path: str):
    cap = ("Streudiagramme Kurs-Δ vs. Volumen-Δ je Horizont (5/10/15/30/60 Minuten) "
           "mit Spearman-Korrelation.")
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    horizons = HORIZONS
    cols = {n: (f"price_delta_pct_{n}m", f"volume_delta_pct_{n}m") for n in horizons}
    if not all(c in df.columns for pair in cols.values() for c in pair):
        return None, cap
    nrows = (len(horizons) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows), sharex=False)
    for ax in axes.flat[len(horizons):]:
        ax.set_visible(False)
    for ax, n in zip(axes.flat, horizons):
        pair = df[list(cols[n])].apply(pd.to_numeric, errors="coerce").dropna()
        x, y, m = pair.iloc[:, 0], pair.iloc[:, 1], len(pair)
        ax.scatter(x, y, s=10, alpha=0.5)
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_yscale("symlog")
        title = f"{n} Min (n={m})"
        if m >= 3 and x.nunique() > 1 and y.nunique() > 1:
            sr, _ = stats.spearmanr(x, y)
            title = f"{n} Min (ρ={sr:.2f}, n={m})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"Kursänderung % (nach {n} min)", fontsize=9)
        ax.set_ylabel("Volumenänderung % (symlog)", fontsize=9)
    fig.suptitle("Kurs-Δ vs. Volumen-Δ je Horizont", fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)
    return out_path, cap


def fig_class_balance(df: pd.DataFrame, out_path: str):
    cap = "Klassenverteilung der Kursreaktion (Auf/Ab/Flat)."
    if "reaction" not in df.columns:
        return None, cap
    counts = df["reaction"].value_counts()
    order = [lab for lab in ("up", "down", "flat") if lab in counts.index]
    if not order:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vals = [counts[lab] for lab in order]
    labels = [REACTION_DE[lab] for lab in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(vals)), vals, color=["#4C72B0", "#C44E52", "#8C8C8C"][:len(vals)])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Reaktionsklasse")
    ax.set_ylabel("Anzahl Meldungen")
    ax.set_title("Klassenverteilung der Kursreaktion")
    _save(fig, out_path)
    return out_path, cap


def fig_vola_pre_post(df: pd.DataFrame, out_path: str):
    cap = "Volatilität der 5-Minuten-Schlusskurse (Variationskoeffizient, in Prozent) in der Stunde vor und nach der Meldung (Boxplot)."
    pre = _numcol(df, "vola_pre_1h_pct")
    post = _numcol(df, "vola_post_1h_pct")
    if pre.empty and post.empty:
        return None, cap
    data, labels = [], []
    if not pre.empty:
        data.append(pre.to_numpy()); labels.append("Vorher (1h)")
    if not post.empty:
        data.append(post.to_numpy()); labels.append("Nachher (1h)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Volatilität %")
    ax.set_title("Volatilität vor vs. nach der Meldung")
    _save(fig, out_path)
    return out_path, cap


def fig_corr_heatmap(df: pd.DataFrame, out_path: str):
    cap = "Spearman-Korrelationen zwischen den Kennzahlen (Kurs, Volumen, Volatilität)."
    cols = [c for c in KPI_COLS if c in df.columns]
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    if num.dropna(how="all").empty or len(cols) < 2:
        return None, cap
    corr = num.corr(method="spearman")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Spearman-Korrelationen der Kennzahlen")
    _save(fig, out_path)
    return out_path, cap


def fig_model_macro_f1(results_df: pd.DataFrame, out_path: str):
    cap = ("Test-Macro-F1 je Modell im Vergleich; die horizontale Linie markiert "
           "die Mehrheitsklassen-Baseline (Macro-F1).")
    if results_df is None or results_df.empty or "test_macro_f1" not in results_df.columns:
        return None, cap
    d = results_df.copy()
    d["test_macro_f1"] = pd.to_numeric(d["test_macro_f1"], errors="coerce")
    d = d.dropna(subset=["test_macro_f1"])
    if d.empty:
        return None, cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(d)), d["test_macro_f1"].to_numpy(), color="#4C72B0")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["model"].tolist(), rotation=30, ha="right")
    maj = d.loc[d["model"] == "majority", "test_macro_f1"]
    if not maj.empty:
        b = float(maj.iloc[0])
        b_de = f"{b:.2f}".replace(".", ",")  # deutsches Dezimalkomma in der Legende
        ax.axhline(b, color="grey", ls="--", lw=1,
                   label=f"Baseline Mehrheitsklasse (Macro-F1) von {b_de}")
        ax.legend()
    ax.set_xlabel("Modell")
    ax.set_ylabel("Test-Macro-F1")
    ax.set_title("Modellgüte (Test-Macro-F1)")
    _save(fig, out_path)
    return out_path, cap


def fig_perclass_f1(out_path: str):
    """Per-class precision/recall/F1 for rf_embed. model_results.csv has no per-class
    metrics and we must not retrain, so this is skipped gracefully."""
    cap = "Precision/Recall/F1 je Klasse (rf_embed) — nicht verfügbar, ohne Neu-Training übersprungen."
    return None, cap


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _write_table(df: pd.DataFrame, name: str):
    tables_dir = FIGURES_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / name)


def tbl_korpus_overview(df: pd.DataFrame, records: list, results_df: pd.DataFrame):
    n_records = len(records) if records is not None else 0
    n_stats = len(df)
    n_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
    dt = _month_series(df if not records else records)
    span = ""
    if not dt.empty:
        span = f"{dt.min().date()} – {dt.max().date()}"
    reaction_counts = df["reaction"].value_counts() if "reaction" in df.columns else pd.Series(dtype=int)
    rows = [
        ("Anzahl Meldungen (Korpus)", n_records),
        ("Anzahl Meldungen mit Ereignisfenster", n_stats),
        ("Anzahl Ticker", n_tickers),
        ("Zeitraum", span),
        ("Reaktion Auf", int(reaction_counts.get("up", 0))),
        ("Reaktion Ab", int(reaction_counts.get("down", 0))),
        ("Reaktion Flat", int(reaction_counts.get("flat", 0))),
    ]
    out = pd.DataFrame(rows, columns=["Kennzahl", "Wert"]).set_index("Kennzahl")
    _write_table(out, "tbl_korpus_overview.csv")


def tbl_kpi_descriptive(df: pd.DataFrame):
    out = descriptive_stats(df, KPI_COLS).round(2)
    _write_table(out.fillna(""), "tbl_kpi_descriptive.csv")


def tbl_model_results(results_df: pd.DataFrame):
    if results_df is None or results_df.empty:
        _write_table(pd.DataFrame(), "tbl_model_results.csv")
        return
    _write_table(results_df.set_index("model") if "model" in results_df.columns
                 else results_df, "tbl_model_results.csv")


def tbl_perclass_metrics():
    """Per-Klassen-Metriken des besten Testmodells (von train.py hinterlegt), sonst leer."""
    src = Path("new_dataset/model_eval/perclass_best.json")
    if not src.exists():
        _write_table(pd.DataFrame(), "tbl_perclass_metrics.csv")
        return
    m = json.loads(src.read_text())
    de = {"up": "Auf", "down": "Ab", "flat": "Flat"}
    rows = {}
    for lab, pc in m["per_class"].items():
        rows[de.get(lab, lab)] = {"precision": round(pc["precision"], 3),
                                  "recall": round(pc["recall"], 3),
                                  "f1": round(pc["f1"], 3),
                                  "support": sum(m["confusion"][lab].values())}
    pcs = list(m["per_class"].values())
    rows["Macro"] = {"precision": round(sum(p["precision"] for p in pcs) / len(pcs), 3),
                     "recall": round(sum(p["recall"] for p in pcs) / len(pcs), 3),
                     "f1": round(sum(p["f1"] for p in pcs) / len(pcs), 3),
                     "support": sum(sum(m["confusion"][l].values()) for l in m["confusion"])}
    out = pd.DataFrame.from_dict(rows, orient="index")[["precision", "recall", "f1", "support"]]
    out.index.name = "klasse"
    _write_table(out, "tbl_perclass_metrics.csv")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_all() -> list[dict]:
    """Load data, generate all figures + tables, write the manifest, return manifest list."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "tables").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(STATS_CSV)
    df["reaction"] = pd.to_numeric(df.get("price_delta_pct_10m"), errors="coerce").map(label_3class)

    records = None
    if Path(DATASET_JSON).exists():
        records = json.loads(Path(DATASET_JSON).read_text())

    results_df = None
    if Path(RESULTS_CSV).exists():
        results_df = pd.read_csv(RESULTS_CSV)

    specs = [
        ("f01_meldungen_zeit", lambda p: fig_meldungen_zeit(records if records else df, p)),
        ("f02_boerse_verteilung", lambda p: fig_boerse_verteilung(records if records else df, p)),
        ("f03_textlaenge", lambda p: fig_textlaenge(records if records else df, p)),
        ("f06_hist_price", lambda p: fig_hist_price(df, p)),
        ("f07_ecdf_price", lambda p: fig_ecdf_price(df, p)),
        ("f08_price_horizon", lambda p: fig_price_horizon(df, p)),
        ("f09_hist_volume", lambda p: fig_hist_volume(df, p)),
        ("f10_ecdf_volume", lambda p: fig_ecdf_volume(df, p)),
        ("f11_volume_by_class", lambda p: fig_volume_by_class(df, p)),
        ("f12_price_vs_volume", lambda p: fig_price_vs_volume(df, p)),
        ("f05_class_balance", lambda p: fig_class_balance(df, p)),
        ("f04_vola_pre_post", lambda p: fig_vola_pre_post(df, p)),
        ("f14_corr_heatmap", lambda p: fig_corr_heatmap(df, p)),
        ("f15_model_macro_f1", lambda p: fig_model_macro_f1(results_df, p)),
        ("perclass_f1_skip", lambda p: fig_perclass_f1(p)),
    ]

    manifest = []
    skipped = []
    for fid, fn in specs:
        out_path = str(FIGURES_DIR / f"{fid}.png")
        path, caption = fn(out_path)
        if path is None:
            skipped.append(fid)
            continue
        manifest.append({
            "id": fid,
            "file": Path(path).name,
            "caption_de": caption,
            "source": SOURCE,
        })

    # Tables
    tbl_korpus_overview(df, records, results_df)
    tbl_kpi_descriptive(df)
    tbl_model_results(results_df)
    tbl_perclass_metrics()

    (FIGURES_DIR / "figures_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"figures: {len(manifest)} PNGs written, 4 tables, "
          f"{len(skipped)} skipped ({', '.join(skipped) or 'none'}) -> {FIGURES_DIR}")
    return manifest


if __name__ == "__main__":
    build_all()
