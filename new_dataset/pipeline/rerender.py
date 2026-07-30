"""Batch re-render all per-news charts from cache with news-time-aware panel selection.
Widens the fetch only for in-hours second-half records missing the following trading day.
Run: python -m new_dataset.pipeline.rerender"""
from __future__ import annotations

import sys

from new_dataset.pipeline.dataset import load_records
from new_dataset.pipeline.eodhd_prices import load_or_fetch
from new_dataset.pipeline.exchanges import get_exch, to_exchange_hours
from new_dataset.pipeline.charts import render, is_second_half, has_following_session

WIDE_POST_HOURS = 192


def rerender_all(records=None) -> dict:
    records = records if records is not None else load_records()
    rep = {"records": 0, "no_bars": 0, "second_half_layout": 0,
           "widened": 0, "still_missing_fallback": 0}
    for rec in records:
        rep["records"] += 1
        exch = get_exch(rec["ticker"])
        bars_exch = to_exchange_hours(load_or_fetch(rec), exch)
        second = is_second_half(rec["news_dt"], exch)
        if second and (bars_exch is None or bars_exch.empty
                       or not has_following_session(bars_exch, rec["news_dt"])):
            bars_exch = to_exchange_hours(
                load_or_fetch(rec, force=True, post_hours=WIDE_POST_HOURS), exch)
            rep["widened"] += 1
        if bars_exch is None or bars_exch.empty:
            rep["no_bars"] += 1
            continue
        if second:
            if has_following_session(bars_exch, rec["news_dt"]):
                rep["second_half_layout"] += 1
            else:
                rep["still_missing_fallback"] += 1
        render(rec, bars_exch)
    print(f"[rerender] {rep}")
    return rep


if __name__ == "__main__":
    rerender_all()
    sys.exit(0)
