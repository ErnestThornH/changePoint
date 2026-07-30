"""Recompute per-news reaction KPIs (price/volume deltas) from fetched bars."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from new_dataset.pipeline.slots import slot_change, HORIZONS, BAR_MINUTES
from new_dataset.pipeline.exchanges import get_exch, to_exchange_hours
from new_dataset.pipeline.eodhd_prices import load_or_fetch
from new_dataset.pipeline.dataset import load_records

OUT_CSV = "new_dataset/event_window_stats.csv"
FUNCTIONING_HORIZON = 5

COLUMNS = ["id", "ticker", "published_utc", "functioning"] + \
    [f"{m}_{n}m" for n in HORIZONS for m in ("price_delta_pct", "volume_delta_pct")] + \
    ["sofort_delta_pct", "vola_pre_1h_pct", "vola_post_1h_pct"]


def sofort_delta(bars_exch, news_dt, flat_eps_bars: int = 1):
    """Immediate move across the dropped news bar: close(first post bar) / close(last pre
    bar) - 1, in %. Uses the same slot boundary as slots.slot_change (k=1). None if a side
    is missing."""
    if bars_exch is None or bars_exch.empty:
        return None
    step = pd.Timedelta(minutes=BAR_MINUTES)
    slot = pd.Timestamp(news_dt).floor(f"{BAR_MINUTES}min")
    pre = bars_exch[bars_exch.index < slot]["Close"].dropna()
    post = bars_exch[bars_exch.index >= slot + step]["Close"].dropna()
    if pre.empty or post.empty or pre.iloc[-1] == 0:
        return None
    return (post.iloc[0] / pre.iloc[-1] - 1) * 100.0


def volatility_pct(bars_exch, news_dt, side: str, minutes: int = 60):
    """std(Close)/mean(Close)*100 over the `minutes` window before ('pre') or after ('post')
    the news bar. None if < 2 bars or zero mean."""
    if bars_exch is None or bars_exch.empty:
        return None
    step = pd.Timedelta(minutes=BAR_MINUTES)
    slot = pd.Timestamp(news_dt).floor(f"{BAR_MINUTES}min")
    if side == "pre":
        win = bars_exch[(bars_exch.index < slot) & (bars_exch.index >= slot - pd.Timedelta(minutes=minutes))]
    else:
        win = bars_exch[(bars_exch.index >= slot + step) & (bars_exch.index <= slot + pd.Timedelta(minutes=minutes))]
    c = win["Close"].dropna()
    if len(c) < 2 or c.mean() == 0:
        return None
    return float(c.std() / c.mean() * 100.0)


def compute_row(record: dict, bars_exch: pd.DataFrame) -> dict:
    """One stats row: per-horizon price/volume deltas (None if not computable) + functioning."""
    row = {"id": record.get("id", ""), "ticker": record.get("ticker", ""),
           "published_utc": record.get("published_utc", ""), "functioning": False}
    for n in HORIZONS:
        row[f"price_delta_pct_{n}m"] = None
        row[f"volume_delta_pct_{n}m"] = None
    row["sofort_delta_pct"] = None
    row["vola_pre_1h_pct"] = None
    row["vola_post_1h_pct"] = None
    if bars_exch is None or bars_exch.empty:
        return row
    news_dt = record["news_dt"]
    for n in HORIZONS:
        res = slot_change(bars_exch, news_dt, n)
        if res is not None:
            row[f"price_delta_pct_{n}m"], row[f"volume_delta_pct_{n}m"] = res
    row["functioning"] = row[f"price_delta_pct_{FUNCTIONING_HORIZON}m"] is not None
    row["sofort_delta_pct"] = sofort_delta(bars_exch, news_dt) if not (bars_exch is None or bars_exch.empty) else None
    row["vola_pre_1h_pct"] = volatility_pct(bars_exch, news_dt, "pre") if not (bars_exch is None or bars_exch.empty) else None
    row["vola_post_1h_pct"] = volatility_pct(bars_exch, news_dt, "post") if not (bars_exch is None or bars_exch.empty) else None
    return row


def compute_all(dataset_path: str | None = None, out_csv: str = OUT_CSV, force: bool = False) -> pd.DataFrame:
    """Load records → fetch bars → exchange-hours → compute_row → write the stats CSV."""
    records = load_records(dataset_path) if dataset_path else load_records()
    rows = []
    for rec in records:
        bars = load_or_fetch(rec, force=force)
        bars_exch = to_exchange_hours(bars, get_exch(rec["ticker"]))
        rows.append(compute_row(rec, bars_exch))
    df = pd.DataFrame(rows, columns=COLUMNS)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
