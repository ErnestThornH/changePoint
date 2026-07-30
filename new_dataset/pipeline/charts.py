"""Rich per-news chart: two split session panels (clear cut between trading days), per-session
volume, KPI/stats box, overnight-gap annotation, exchange/tz title, bottom metadata box.
Ported/adapted from intraday_analysis.make_chart. Line style; no PELT/candlesticks. Agg."""
from __future__ import annotations

import textwrap
from datetime import time as dt_time, timedelta
from pathlib import Path

import pandas as pd

from new_dataset.pipeline.kpis import compute_row
from new_dataset.pipeline.slots import HORIZONS
from new_dataset.pipeline.exchanges import get_exch, midpoint

CHARTS_DIR = Path("new_dataset/charts")
C_PRICE = "steelblue"
C_NEWS = "#f4a261"
C_GAP = "#dde3ea"
MIN_PRICE_POINTS = 10


def _split_sessions(bars_exch: pd.DataFrame, news_dt) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group in-hours bars by calendar date; next = earliest session whose last bar >= news,
    prev = the session immediately before it. Either may be empty."""
    if bars_exch is None or bars_exch.empty:
        return pd.DataFrame(), pd.DataFrame()
    news_dt = pd.Timestamp(news_dt)
    by_day = {d: g for d, g in bars_exch.groupby(bars_exch.index.normalize())}
    days = sorted(by_day)
    next_day = next((d for d in days if by_day[d].index[-1] >= news_dt), None)
    if next_day is None:                               # news after all bars -> last session is 'prev'
        return by_day[days[-1]], pd.DataFrame()
    prev_days = [d for d in days if d < next_day]
    prev = by_day[prev_days[-1]] if prev_days else pd.DataFrame()
    return prev, by_day[next_day]


def is_second_half(news_dt, exch) -> bool:
    """True iff the news lands during trading hours AND at/after the session midpoint."""
    t = pd.Timestamp(news_dt).time()
    (oh, om), (ch, cm) = exch["open"], exch["close"]
    outside = t < dt_time(oh, om) or t >= dt_time(ch, cm)
    return (not outside) and t >= midpoint(exch)


def has_following_session(bars_exch: pd.DataFrame, news_dt) -> bool:
    """True iff some in-hours session has a calendar date after the news date."""
    if bars_exch is None or bars_exch.empty:
        return False
    nd = pd.Timestamp(news_dt).normalize()
    return any(d > nd for d in bars_exch.index.normalize().unique())


def select_panels(bars_exch: pd.DataFrame, news_dt, exch) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For in-hours second-half news with a following trading day, return
    (news_day, following); otherwise the default (prev, next) from _split_sessions."""
    news_dt = pd.Timestamp(news_dt)
    if is_second_half(news_dt, exch) and has_following_session(bars_exch, news_dt):
        by_day = {d: g for d, g in bars_exch.groupby(bars_exch.index.normalize())}
        nd = news_dt.normalize()
        news_day = by_day.get(nd, pd.DataFrame())
        following_days = sorted(d for d in by_day if d > nd)
        following = by_day[following_days[0]] if following_days else pd.DataFrame()
        if not news_day.empty and not following.empty:
            return news_day, following
    return _split_sessions(bars_exch, news_dt)


def _draw_session(ax_p, ax_v, df: pd.DataFrame, alpha: float = 1.0):
    if df is None or df.empty:
        return
    c = df["Close"].dropna()
    if len(c) <= 8:
        ax_p.step(c.index, c.values, where="post", color=C_PRICE, linewidth=1.1, alpha=alpha)
        ax_p.plot(c.index, c.values, "o", color=C_PRICE, markersize=4, alpha=alpha * 0.85)
    else:
        ax_p.plot(c.index, c.values, color=C_PRICE, linewidth=1.1, alpha=alpha)
    if "Volume" in df.columns:
        v = df["Volume"].fillna(0)
        bar_w = timedelta(minutes=4) / timedelta(days=1)
        ax_v.bar(v.index, v.values, width=bar_w, color=C_PRICE, alpha=0.45 * alpha)


def _configure_xaxis(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", labelsize=8)


def _fpct(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    a = abs(v)
    if a >= 1e5:
        return f"{v:+.0e}%"
    if a >= 1000:
        return f"{v:+.0f}%"
    return f"{v:+.2f}%"


def _fabs(v):
    return f"{v:.2f}%" if v is not None and not (isinstance(v, float) and pd.isna(v)) else "–"


def _add_stats_box(ax, row: dict, news_dt, tz_label: str):
    labels = [f"{n}m" for n in HORIZONS]
    hdr = f"{'':<9}" + "".join(f"{l:>10}" for l in labels)
    p_row = f"{'Preis Δ':<9}" + "".join(f"{_fpct(row.get(f'price_delta_pct_{n}m')):>10}" for n in HORIZONS)
    v_row = f"{'Vol   Δ':<9}" + "".join(f"{_fpct(row.get(f'volume_delta_pct_{n}m')):>10}" for n in HORIZONS)
    text = (
        f"Meldung {pd.Timestamp(news_dt).strftime('%H:%M')}  {pd.Timestamp(news_dt).strftime('%d.%m.%Y')} ({tz_label})\n"
        f"{hdr}\n{p_row}\n{v_row}\n"
        f"Δ sofort: {_fpct(row.get('sofort_delta_pct'))}\n"
        f"Vola 1h: vor {_fabs(row.get('vola_pre_1h_pct'))}  nach {_fabs(row.get('vola_post_1h_pct'))}"
    )
    ax.text(0.985, 0.03, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, fontfamily="monospace", zorder=20,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#888888",
                      linewidth=0.8, alpha=0.92))


def _annotate_gap(ax, ref_price: float, react_price: float, xy):
    delta = (react_price / ref_price - 1) * 100 if ref_price else 0.0
    flat = abs(delta) < 0.05
    col = "#666666" if flat else ("green" if delta >= 0 else "red")
    kw = dict(xy=xy, textcoords="offset points", xytext=(10, 14 if delta >= 0 else -28),
              fontsize=9, fontweight="bold", color=col,
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=col,
                        linewidth=0.8, alpha=0.9))
    if not flat:
        kw["arrowprops"] = dict(arrowstyle="->", color=col, lw=0.9)
    ax.annotate(f"Overnight-Gap: {delta:+.2f}%\n({ref_price:.2f} → {react_price:.2f})", **kw)


def _add_warning_box(ax, n_points: int):
    ax.text(0.5, 0.5, f"⚠ Wenige Handelsdaten — nur {n_points} Kurspunkt(e)\nkeine belastbare Reaktion",
            transform=ax.transAxes, ha="center", va="center", fontsize=11, fontweight="bold",
            color="#b35900", zorder=30,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff3cd", edgecolor="#e0a800",
                      linewidth=1.4, alpha=0.95))


def _add_metadata_box(ax, record: dict):
    ax.axis("off")
    title = textwrap.fill((record.get("title") or "")[:160], width=115)
    summary = textwrap.fill((record.get("summary") or "")[:450], width=115)
    ax.text(0.01, 0.97,
            f"Titel:    {title}\n"
            f"ISIN:     {record.get('isin','')}   |   WKN: {record.get('wkn','')}   |   Quelle: {record.get('source','')}\n"
            f"Link:     {(record.get('link') or '')[:105]}\n\n"
            f"Summary:  {summary}",
            transform=ax.transAxes, fontsize=7.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#cccccc",
                      linewidth=0.8))


def render(record: dict, bars_exch: pd.DataFrame, out_path: str | None = None) -> str | None:
    """Rich two-/one-session chart. Returns the path written, or None if no bars."""
    if out_path is None:
        out_path = str(CHARTS_DIR / f"{record['id']}.png")
    if bars_exch is None or bars_exch.empty:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    news_dt = pd.Timestamp(record["news_dt"])
    exch = get_exch(record["ticker"])
    (oh, om), (ch, cm) = exch["open"], exch["close"]
    tz_label = exch["tz"].split("/")[-1]
    outside = news_dt.time() < dt_time(oh, om) or news_dt.time() >= dt_time(ch, cm)
    stats = compute_row(record, bars_exch)
    prev_df, next_df = select_panels(bars_exch, news_dt, exch)
    n_points = sum(int(d["Close"].dropna().shape[0]) for d in (prev_df, next_df)
                   if d is not None and not d.empty and "Close" in d.columns)
    if n_points == 0:
        return None
    sparse = n_points < MIN_PRICE_POINTS

    def _news_in(df):
        return (df is not None and not df.empty
                and df.index[0] <= news_dt <= df.index[-1])

    title = (f"{record.get('company','')}  ·  {record.get('ticker','')}  ·  "
             f"{news_dt.strftime('%d.%m.%Y %H:%M')} {tz_label}"
             f"{'  ⚠ außerbörslich' if outside else ''}\n"
             f"Börse: {exch.get('name','')}  {oh:02d}:{om:02d}–{ch:02d}:{cm:02d} {tz_label}  ·  Intervall: 5m")

    if prev_df is not None and not prev_df.empty and next_df is not None and not next_df.empty:
        fig = plt.figure(figsize=(16, 9), layout="constrained")
        gs = fig.add_gridspec(3, 3, height_ratios=[4, 1.2, 1.5], width_ratios=[1, 0.06, 1],
                              hspace=0.08, wspace=0.04)
        ax_pp, ax_pn = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])
        ax_vp = fig.add_subplot(gs[1, 0], sharex=ax_pp)
        ax_vn = fig.add_subplot(gs[1, 2], sharex=ax_pn)
        ax_div_p, ax_div_v = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
        ax_t = fig.add_subplot(gs[2, :])
        for a in (ax_div_p, ax_div_v):
            a.set_facecolor(C_GAP); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        news_in_next = _news_in(next_df)
        news_in_prev = _news_in(prev_df)
        ax_div_p.text(0.5, 0.5,
                      (f"Börs-\npause\n\n{news_dt.strftime('%H:%M')}\n{tz_label}\n\nNews"
                       if not (news_in_prev or news_in_next) else "Über-\nnacht-\npause"),
                      ha="center", va="center", fontsize=7.5, color="#444444", style="italic",
                      transform=ax_div_p.transAxes)
        _draw_session(ax_pp, ax_vp, prev_df, alpha=1.0 if news_in_prev else 0.7)
        _draw_session(ax_pn, ax_vn, next_df, alpha=1.0)
        if news_in_prev:
            ax_pp.axvline(news_dt, color=C_NEWS, ls=":", lw=2.0, label=f"Meldung {news_dt.strftime('%H:%M')}")
            ax_vp.axvline(news_dt, color=C_NEWS, ls=":", lw=1.2, alpha=0.7)
        elif news_in_next:
            ax_pn.axvline(news_dt, color=C_NEWS, ls=":", lw=2.0, label=f"Meldung {news_dt.strftime('%H:%M')}")
            ax_vn.axvline(news_dt, color=C_NEWS, ls=":", lw=1.2, alpha=0.7)
        if not news_in_prev and not news_in_next:
            ref = prev_df["Close"].dropna(); rct = next_df["Close"].dropna()
            if not ref.empty and not rct.empty and ref.iloc[-1] > 0:
                _annotate_gap(ax_pn, float(ref.iloc[-1]), float(rct.iloc[0]), (rct.index[0], float(rct.iloc[0])))
        for ax in (ax_pp, ax_vp, ax_pn, ax_vn):
            _configure_xaxis(ax); ax.grid(True, alpha=0.22)
        plt.setp(ax_pp.get_xticklabels(), visible=False)
        plt.setp(ax_pn.get_xticklabels(), visible=False)
        ax_pn.yaxis.set_label_position("right"); ax_pn.yaxis.tick_right()
        ax_vn.yaxis.set_label_position("right"); ax_vn.yaxis.tick_right()
        ax_pp.set_ylabel("Kurs", fontsize=9); ax_vp.set_ylabel("Vol", fontsize=8)
        ax_pp.set_title(prev_df.index[0].strftime("%d.%m.%Y") + ("  · News" if news_in_prev else ""),
                        fontsize=9, loc="left", color="#222" if news_in_prev else "#555")
        ax_pn.set_title(next_df.index[0].strftime("%d.%m.%Y") + ("  · News" if news_in_next else ""),
                        fontsize=9, loc="left", color="#222")
        news_panel = ax_pp if news_in_prev else ax_pn
        _add_stats_box(news_panel, stats, news_dt, tz_label)
        if sparse:
            _add_warning_box(news_panel, n_points)
        _add_metadata_box(ax_t, record)
        fig.suptitle(title, fontsize=10)
    else:
        df = next_df if (next_df is not None and not next_df.empty) else prev_df
        fig = plt.figure(figsize=(15, 9), layout="constrained")
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 1.2, 1.5], hspace=0.08)
        ax_p = fig.add_subplot(gs[0]); ax_v = fig.add_subplot(gs[1], sharex=ax_p)
        ax_t = fig.add_subplot(gs[2])
        _draw_session(ax_p, ax_v, df, alpha=1.0)
        if _news_in(df):
            ax_p.axvline(news_dt, color=C_NEWS, ls=":", lw=2.0, label=f"Meldung {news_dt.strftime('%H:%M')}")
            ax_v.axvline(news_dt, color=C_NEWS, ls=":", lw=1.2, alpha=0.7)
        _configure_xaxis(ax_p); _configure_xaxis(ax_v)
        plt.setp(ax_p.get_xticklabels(), visible=False)
        ax_p.set_ylabel("Kurs", fontsize=9); ax_v.set_ylabel("Vol", fontsize=8)
        ax_p.grid(True, alpha=0.22); ax_v.grid(True, alpha=0.2)
        _add_stats_box(ax_p, stats, news_dt, tz_label)
        if sparse:
            _add_warning_box(ax_p, n_points)
        _add_metadata_box(ax_t, record)
        fig.suptitle(title, fontsize=10)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
