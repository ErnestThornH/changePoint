"""Cross-session event-window slot computation (ported verbatim from event_window_scatter)."""
from __future__ import annotations

import pandas as pd

HORIZONS = [5, 10, 15, 30, 60]
BAR_MINUTES = 5


def slot_change(df: pd.DataFrame, news_dt, minutes: int):
    """(price_pct, volume_pct) for one ad-hoc at horizon `minutes`, or None.

    df: in-hours 5-min bars across one OR MORE sessions (DatetimeIndex, exchange-local
    tz-naive), columns Close, Volume, sorted ascending. The partial bar whose 5-min slot
    contains `news_dt` is dropped; pre-slot = the k bars immediately before it, post-slot =
    the k immediately after — taken from the nearest available session(s).
    price = close(last post)/close(last pre)-1 (point-to-point, %); volume = sum(post)/sum(pre)-1.
    """
    if df is None or df.empty:
        return None
    k = minutes // BAR_MINUTES
    step = pd.Timedelta(minutes=BAR_MINUTES)
    slot = pd.Timestamp(news_dt).floor(f"{BAR_MINUTES}min")

    pre = df[df.index < slot].tail(k)
    post = df[df.index >= slot + step].head(k)
    if len(pre) < k or len(post) < k:
        return None

    pre_vol = float(pre["Volume"].fillna(0.0).sum())
    post_vol = float(post["Volume"].fillna(0.0).sum())
    if pre_vol == 0 or post_vol == 0:
        return None
    volume_pct = (post_vol / pre_vol - 1) * 100.0

    pre_close = pre["Close"].dropna()
    post_close = post["Close"].dropna()
    if pre_close.empty or post_close.empty:
        return None
    price_pct = (post_close.iloc[-1] / pre_close.iloc[-1] - 1) * 100.0

    return price_pct, volume_pct
