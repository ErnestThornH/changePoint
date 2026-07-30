"""Strictly-EODHD 5-minute OHLCV fetch, one CSV per news (keyed by record id).
Self-contained — mirrors eodh_intraday's EODHD call shape; no old cache, no yfinance."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

INTRADAY_DIR = Path("new_dataset/intraday")
BASE_URL = "https://eodhd.com/api/intraday/{symbol}"
WINDOW_HOURS = 96


def _token() -> str | None:
    return os.environ.get("EODHD_TOKEN") or (
        Path("apikey_free.txt").read_text().strip() if os.path.isfile("apikey_free.txt") else None)


def cache_path(record: dict) -> Path:
    return INTRADAY_DIR / f"{record['id']}.csv"


def _parse_response(rows) -> pd.DataFrame:
    """EODHD intraday rows (list of dicts) → OHLCV DataFrame, tz-naive UTC index."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for raw, std in [("date time", "datetime"), ("Datetime", "datetime"), ("open", "Open"),
                     ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")]:
        if raw in df.columns:
            df = df.rename(columns={raw: std})
    if "datetime" not in df.columns:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.NA
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def fetch_bars(ticker: str, start, end) -> pd.DataFrame:
    """5-min OHLCV for [start, end] (date str or datetime). Empty DataFrame on any error."""
    token = _token()
    if not token:
        return pd.DataFrame()
    def _ts(d):
        return int(pd.to_datetime(d).timestamp())
    params = {"api_token": token, "fmt": "json", "interval": "5m",
              "from": _ts(start), "to": _ts(end)}
    try:
        r = requests.get(BASE_URL.format(symbol=ticker), params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame()
    rows = data if isinstance(data, list) else (data.get("data", []) if isinstance(data, dict) else [])
    return _parse_response(rows)


def load_or_fetch(record: dict, force: bool = False,
                  pre_hours: int | None = None, post_hours: int | None = None) -> pd.DataFrame:
    """Return the per-news 5-min bars; fetch + cache to new_dataset/intraday/{id}.csv on miss."""
    p = cache_path(record)
    if p.exists() and p.stat().st_size > 0 and not force:
        df = pd.read_csv(p, parse_dates=["datetime"]).set_index("datetime")
        return df
    pub = pd.to_datetime(record["published_utc"])
    start = (pub - pd.Timedelta(hours=pre_hours or WINDOW_HOURS)).strftime("%Y-%m-%d")
    end = (pub + pd.Timedelta(hours=post_hours or WINDOW_HOURS)).strftime("%Y-%m-%d")
    df = fetch_bars(record["ticker"], start, end)
    if df is None or df.empty:
        return pd.DataFrame()
    INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
    out = df.reset_index().rename(columns={df.index.name or "index": "datetime"})
    if out.columns[0] != "datetime":
        out = out.rename(columns={out.columns[0]: "datetime"})
    out.to_csv(p, index=False)
    return df
