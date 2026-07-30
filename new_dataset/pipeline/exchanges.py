"""Exchange metadata + tz/hours helpers (ported from intraday_analysis, +US)."""
from __future__ import annotations

from datetime import time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd

EXCHANGE_INFO: dict[str, dict] = {
    "XETRA":  {"tz": "Europe/Berlin", "open": (9, 0),  "close": (17, 30), "name": "XETRA"},
    "F":      {"tz": "Europe/Berlin", "open": (8, 0),  "close": (22, 0),  "name": "Frankfurt"},
    "STU":    {"tz": "Europe/Berlin", "open": (8, 0),  "close": (22, 0),  "name": "Stuttgart"},
    "BE":     {"tz": "Europe/Berlin", "open": (8, 0),  "close": (22, 0),  "name": "Berlin"},
    "MU":     {"tz": "Europe/Berlin", "open": (8, 0),  "close": (22, 0),  "name": "München"},
    "HM":     {"tz": "Europe/Berlin", "open": (8, 0),  "close": (22, 0),  "name": "Hamburg"},
    "DU":     {"tz": "Europe/Berlin", "open": (8, 0),  "close": (22, 0),  "name": "Düsseldorf"},
    "SW":     {"tz": "Europe/Zurich", "open": (9, 0),  "close": (17, 30), "name": "SIX Swiss"},
    "VI":     {"tz": "Europe/Vienna", "open": (9, 0),  "close": (17, 35), "name": "Wien"},
    "LSE":    {"tz": "Europe/London", "open": (8, 0),  "close": (16, 30), "name": "London"},
    "US":     {"tz": "America/New_York", "open": (9, 30), "close": (16, 0), "name": "US"},
    "NYSE":   {"tz": "America/New_York", "open": (9, 30), "close": (16, 0), "name": "NYSE"},
    "NASDAQ": {"tz": "America/New_York", "open": (9, 30), "close": (16, 0), "name": "NASDAQ"},
}
_DEFAULT = EXCHANGE_INFO["XETRA"]


def get_exch(ticker: str) -> dict:
    """Exchange info for an EODHD ticker CODE.SUFFIX (default XETRA if unknown)."""
    suffix = ticker.rsplit(".", 1)[1].upper() if ticker and "." in ticker else ""
    return EXCHANGE_INFO.get(suffix, _DEFAULT)


def midpoint(exch: dict) -> dt_time:
    """Local-time midpoint of the trading session: (open+close)/2."""
    (oh, om), (ch, cm) = exch["open"], exch["close"]
    total = (oh * 60 + om + ch * 60 + cm) // 2
    return dt_time(total // 60, total % 60)


def to_exchange_hours(df: pd.DataFrame, exch: dict) -> pd.DataFrame:
    """Convert a tz-naive (UTC) or tz-aware index to the exchange tz (tz-naive) and keep
    only in-trading-hours rows (incl. open, excl. close)."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    df = df.copy()
    tz = ZoneInfo(exch["tz"])
    if df.index.tz is not None:
        df.index = df.index.tz_convert(tz).tz_localize(None)
    else:
        df.index = df.index.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    (oh, om), (ch, cm) = exch["open"], exch["close"]
    mask = (df.index.time >= dt_time(oh, om)) & (df.index.time < dt_time(ch, cm))
    return df[mask]
