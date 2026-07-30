"""Price-vs-volume scatter plots per horizon (ported from event_window_scatter)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from new_dataset.pipeline.slots import HORIZONS

SCATTER_DIR = Path("new_dataset/scatter")


def render(df: pd.DataFrame, minutes: int, out_path: str) -> int:
    """Scatter X=price Δ%, Y=volume Δ% at one horizon; annotate Pearson/Spearman + N.
    Returns number of points plotted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xcol, ycol = f"price_delta_pct_{minutes}m", f"volume_delta_pct_{minutes}m"
    pair = df[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
    x, y, n = pair[xcol], pair[ycol], len(pair)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=12, alpha=0.5)
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    title = f"{minutes} min — Preis Δ% vs Volumen Δ% (n={n})"
    if n >= 3 and x.nunique() > 1 and y.nunique() > 1:
        pr, _ = stats.pearsonr(x, y)
        sr, _ = stats.spearmanr(x, y)
        title = f"{minutes} min  (r={pr:.2f}, ρ={sr:.2f}, n={n})"
    ax.set_xlabel(f"Aktienkursänderung % (nach {minutes} min)")
    ax.set_yscale("symlog", linthresh=10.0)
    ax.set_ylabel(f"Volumenänderung % (Slot ±{minutes} min) (symlog)")
    ax.set_title(title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    return n


def render_all(df: pd.DataFrame, out_dir: str | None = None) -> list[str]:
    """Render all four horizon scatters; return the list of output paths."""
    d = Path(out_dir) if out_dir else SCATTER_DIR
    paths = []
    for n in HORIZONS:
        p = str(d / f"scatter_{n}min.png")
        render(df, n, p)
        paths.append(p)
    return paths
