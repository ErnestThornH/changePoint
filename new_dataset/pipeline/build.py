"""Orchestrator: validate -> fetch (once) -> chart + KPIs -> scatters -> reports.
Run: python -m new_dataset.pipeline.build  [--force]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from new_dataset.pipeline.dataset import load_records
from new_dataset.pipeline.eodhd_prices import load_or_fetch
from new_dataset.pipeline.exchanges import get_exch, to_exchange_hours
from new_dataset.pipeline.kpis import compute_row, COLUMNS
from new_dataset.pipeline.slots import HORIZONS
from new_dataset.pipeline.charts import render as render_chart
from new_dataset.pipeline.scatter import render_all as render_all_scatters

EVENT_STATS_CSV = "new_dataset/event_window_stats.csv"
FAILED_JSON = "new_dataset/fetch_failed.json"
REPORT_JSON = "new_dataset/coverage_report.json"


def main(dataset_path: str | None = None, force: bool = False) -> int:
    records = load_records(dataset_path) if dataset_path else load_records()
    rows, failed = [], []
    for rec in records:
        bars = load_or_fetch(rec, force=force)
        bars_exch = to_exchange_hours(bars, get_exch(rec["ticker"]))
        if bars_exch is None or bars_exch.empty:
            failed.append({"id": rec["id"], "ticker": rec["ticker"], "reason": "no_bars"})
        else:
            render_chart(rec, bars_exch)
        rows.append(compute_row(rec, bars_exch))

    df = pd.DataFrame(rows, columns=COLUMNS)
    Path(EVENT_STATS_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EVENT_STATS_CSV, index=False)
    render_all_scatters(df)

    functioning = {str(n): int((df[f"price_delta_pct_{n}m"].notna()
                                & df[f"volume_delta_pct_{n}m"].notna()).sum()) for n in HORIZONS}
    dataset_flag = int(sum(1 for r in records if (r.get("intraday") or {}).get("functioning")))
    report = {"records": len(records), "failed_fetch": len(failed),
              "functioning": functioning, "dataset_functioning_flag": dataset_flag}

    Path(FAILED_JSON).write_text(json.dumps(failed, ensure_ascii=False, indent=2))
    Path(REPORT_JSON).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[build] records={len(records)} failed_fetch={len(failed)} functioning={functioning}")
    return 0


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
