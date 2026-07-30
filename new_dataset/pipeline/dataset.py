import json
from pathlib import Path

import pandas as pd

REQUIRED = [
    "id", "isin", "ticker", "company", "country", "title", "summary",
    "content", "link", "published_utc", "published_local", "exchange_tz",
]


def validate_record(r: dict) -> list[str]:
    """Return a list of problem strings for a record. Empty list = valid."""
    problems = []
    for key in REQUIRED:
        val = r.get(key)
        if not isinstance(val, str) or not val.strip():
            problems.append(f"{key}: missing or empty")
    ticker = r.get("ticker")
    if isinstance(ticker, str) and ticker.strip() and "." not in ticker:
        problems.append("ticker: missing exchange suffix (expected CODE.SUFFIX)")
    published_local = r.get("published_local")
    if isinstance(published_local, str) and published_local.strip():
        try:
            pd.to_datetime(published_local, errors="raise")
        except Exception:
            problems.append("published_local: unparseable")
    return problems


def load_records(path="new_dataset/adhoc_functioning.json") -> list[dict]:
    """Load a JSON array of records, skipping invalid ones.

    Each kept record gets a tz-naive ``news_dt`` pandas Timestamp attached.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset file not found: {p}")
    records = json.loads(p.read_text())
    kept = []
    for r in records:
        if validate_record(r):
            continue
        r["news_dt"] = pd.Timestamp(pd.to_datetime(r["published_local"]))
        kept.append(r)
    return kept
