"""Zufallszeitpunkt-Baseline zur flachen ±0,1-%-Schwelle.

Der Betreuer wandte ein, die feste Schwelle von ±0,1 % sei zu eng, weil sich
Kurse ohnehin ständig bewegen. Diese Baseline misst deshalb dieselbe
10-Minuten-Kursänderung wie für echte Ad-hoc-Meldungen (identische
``slots.slot_change``-Logik), aber an zufälligen Nicht-News-Zeitpunkten
innerhalb derselben, bereits gecachten Kursreihen. Bewegt der Kurs sich an
Zufallszeitpunkten deutlich häufiger flach als nach Meldungen, so trennt die
Schwelle Nachrichten von Grundrauschen.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from new_dataset.pipeline.slots import slot_change, BAR_MINUTES
from new_dataset.pipeline.exchanges import get_exch, to_exchange_hours
from new_dataset.pipeline.eodhd_prices import load_or_fetch
from new_dataset.pipeline.dataset import load_records

HORIZON = 10
FLAT_EPS = 0.1  # feste Schwelle in Prozent
OUT_CSV = "new_dataset/random_baseline.csv"
FIG_PNG = "new_dataset/figures/f28_random_baseline.png"
NEWS_CSV = "new_dataset/event_window_stats.csv"


def candidate_times(bars: pd.DataFrame, news_utc, k: int, exclusion_min: int = 60) -> list:
    """Alle Bar-Zeitstempel, die als fiktiver Meldungszeitpunkt taugen.

    Ein Zeitpunkt ist verwendbar, wenn in der Reihe mindestens ``k`` vollständige
    Bars davor UND danach liegen (damit ``slot_change`` bei Horizont
    ``k*BAR_MINUTES`` einen Wert liefert) und er außerhalb von
    ±``exclusion_min`` Minuten um ``news_utc`` liegt.
    """
    if bars is None or bars.empty:
        return []
    idx = bars.index
    excl = exclusion_min * 60
    out = []
    for i, t in enumerate(idx):
        if i < k or i > len(idx) - 1 - k:
            continue
        if abs((t - news_utc).total_seconds()) <= excl:
            continue
        out.append(t)
    return out


def sample_times(bars: pd.DataFrame, news_utc, n: int, seed: int,
                 k: int = 2, exclusion_min: int = 60) -> list:
    """Deterministische Stichprobe von ``min(n, #Kandidaten)`` Zeitpunkten (sortiert)."""
    cands = candidate_times(bars, news_utc, k, exclusion_min)
    if not cands:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(cands, min(n, len(cands))))


def _summary(vals: np.ndarray) -> tuple[int, float, float]:
    """(n, Median |Δ|, Anteil |Δ| <= FLAT_EPS) für ein Array von Prozent-Deltas."""
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 0, float("nan"), float("nan")
    return v.size, float(np.median(np.abs(v))), float((np.abs(v) <= FLAT_EPS).mean())


def _figure(news: np.ndarray, rnd: np.ndarray, path: str) -> None:
    """Überlagertes Dichte-Histogramm (symlog) der News- vs. Zufalls-Deltas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo = min(np.nanmin(news), np.nanmin(rnd))
    hi = max(np.nanmax(news), np.nanmax(rnd))
    span = max(abs(lo), abs(hi))
    lin = np.linspace(-FLAT_EPS, FLAT_EPS, 5)
    pos = np.geomspace(FLAT_EPS, span, 40)
    edges = np.unique(np.concatenate([-pos[::-1], lin, pos]))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(news, bins=edges, density=True, alpha=0.55,
            color="#c0392b", label="Ad-hoc-Meldungen")
    ax.hist(rnd, bins=edges, density=True, alpha=0.55,
            color="#2c7fb8", label="Zufallszeitpunkte")
    ax.axvspan(-FLAT_EPS, FLAT_EPS, color="0.7", alpha=0.35, zorder=0)
    ax.set_xscale("symlog", linthresh=FLAT_EPS)
    ax.set_xlabel("Kursänderung 10 Minuten nach dem Zeitpunkt (%)")
    ax.set_ylabel("Dichte")
    ax.set_title("Kursbewegung: Ad-hoc-Meldungen vs. Zufallszeitpunkte")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(n_per_news: int = 3, seed: int = 0) -> pd.DataFrame:
    """Für jede gecachte Kursreihe Zufalls-Deltas berechnen, CSV + Abbildung schreiben."""
    records = load_records()
    rows = []
    for i, rec in enumerate(records):
        bars = load_or_fetch(rec)
        bars_exch = to_exchange_hours(bars, get_exch(rec["ticker"]))
        if bars_exch is None or bars_exch.empty:
            continue
        news_dt = rec["news_dt"]
        # Seed pro Meldung versetzt -> reproduzierbar und über Reihen dekorreliert.
        times = sample_times(bars_exch, news_dt, n_per_news, seed=seed + i)
        for t in times:
            res = slot_change(bars_exch, t, HORIZON)
            if res is not None:
                rows.append({"id": rec["id"], "ts": t, "delta_pct": res[0]})

    df = pd.DataFrame(rows, columns=["id", "ts", "delta_pct"])
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    news_vals = pd.read_csv(NEWS_CSV)["price_delta_pct_10m"].to_numpy(dtype=float)
    rnd_vals = df["delta_pct"].to_numpy(dtype=float)
    _figure(news_vals, rnd_vals, FIG_PNG)

    rn, rmed, rshare = _summary(rnd_vals)
    nn, nmed, nshare = _summary(news_vals)
    print("=== Zufallszeitpunkt-Baseline (10-Min-Kursänderung) ===")
    print(f"Zufallszeitpunkte : n={rn:4d}  Median|Δ|={rmed:6.4f}%  "
          f"Anteil |Δ|<={FLAT_EPS}% = {rshare:6.4f} ({rshare * 100:.1f} %)")
    print(f"Ad-hoc-Meldungen  : n={nn:4d}  Median|Δ|={nmed:6.4f}%  "
          f"Anteil |Δ|<={FLAT_EPS}% = {nshare:6.4f} ({nshare * 100:.1f} %)")
    print(f"CSV : {OUT_CSV}")
    print(f"PNG : {FIG_PNG}")
    return df


if __name__ == "__main__":
    main()
