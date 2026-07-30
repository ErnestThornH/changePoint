"""Article → advice: fetch a news URL, resolve company/ticker (EODHD search), last close
(EODHD EOD), sentiment + trained price-model recommendation. Network calls never raise;
heavy imports lazy."""
from __future__ import annotations

import re

from text_clean import trim_eqs_lines as trim_eqs_boilerplate

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; changePoint-dashboard/1.0)"}
MIN_TEXT = 200
_PUBLISHED_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\s*/\s*(\d{2}):(\d{2})")
_ISIN_RE = re.compile(r"ISIN[:\s]*([A-Z]{2}[A-Z0-9]{9}\d)")
_WKN_RE = re.compile(r"WKN[:\s]*([A-Z0-9]{6})")
# Venue-Ranking übernommen aus dem erprobten Scrapling-Resolver: XETRA zuerst,
# dann Hauptbörsen, deutsche Regionalparkette zuletzt.
_PREFERENCE = ["XETRA", "SW", "PA", "MC", "AS", "LSE", "US", "VI", "MI", "ST", "CO", "HE", "OL", "LS"]
_FLOOR = {"F", "STU", "DU", "HM", "MU", "BE", "HA", "GER"}
REACTION_DE = {"up": "Kaufen", "down": "Verkaufen", "flat": "Halten"}
_RICHTUNG_DE = {"up": "Auf", "down": "Ab", "flat": "Flat"}
DISCLAIMER = ("⚠️ Demonstration eines schwachen 10-Minuten-Reaktionsmodells "
              "(Test-Macro-F1 ≈ 0,48); **keine Anlageberatung**. Die Empfehlung illustriert "
              "die Modellvorhersage, sie ist keine Kauf- oder Verkaufsempfehlung.")


_COMPANY_LINE_RE = re.compile(
    r"^\s*(.{3,90}?\b(?:AG|SE|KGaA|GmbH|N\.V\.|S\.A\.|SA|plc|ASA|Oyj))\s*:\s*(.+)", re.S)


def split_company_title(text: str) -> tuple[str, str]:
    """(company, title) aus der ersten Zeile eingefügten EQS-Texts („Firma AG: Titel"),
    sonst ("", "")."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    m = _COMPANY_LINE_RE.match(first)
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", "")


def short_name(company: str) -> str:
    """Erste zwei Wörter eines langen Firmennamens (EODHD-Suche mag Kurznamen)."""
    words = [w for w in (company or "").split() if w]
    return " ".join(words[:2]) if len(words) > 2 else ""


def extract_ids(text: str) -> dict:
    """ISIN/WKN aus Fließtext (EQS-Fußzeile o. ä.); {isin, wkn} mit None bei Fehltreffer."""
    im = _ISIN_RE.search(text or "")
    wm = _WKN_RE.search(text or "")
    return {"isin": im.group(1) if im else None, "wkn": wm.group(1) if wm else None}


def parse_article_html(html: str) -> dict:
    """{title, company, text, isin, wkn} aus Artikel-HTML, oder {error}.
    EQS-bewusst: h1.news-details__title trägt Firma (Zeile 1) und Headline (Rest);
    Body bevorzugt aus div.news-details__content. Generischer Fallback für andere Seiten."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    title, company = "", ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    h1 = soup.find("h1", class_="news-details__title") or soup.find("h1")
    if h1 and h1.get_text(strip=True):
        lines = [ln.strip() for ln in h1.get_text("\n").split("\n") if ln.strip()]
        if len(lines) > 1:
            company, title = lines[0], " ".join(lines[1:])
        else:
            title = lines[0]
    eqs_body = soup.find("div", class_="news-details__content")
    if eqs_body is not None:
        lines = [ln.strip() for ln in eqs_body.get_text("\n").split("\n") if ln.strip()]
        full = " ".join(lines)
        text = "\n".join(trim_eqs_boilerplate(lines))
        ids = extract_ids(full)          # ISIN/WKN stehen im EQS-Fußteil
    else:
        body = soup.find("article")
        ps = body.find_all("p") if body else soup.find_all("p")
        text = re.sub(r"\s+", " ", " ".join(p.get_text(" ", strip=True) for p in ps)).strip()
        ids = extract_ids(text)
    if len(text) < MIN_TEXT:
        return {"error": "Artikeltext zu kurz/nicht lesbar (evtl. Paywall oder JS-Seite)."}
    published = None
    m = _PUBLISHED_RE.search(re.sub(r"<[^>]+>", " ", html))
    if m:
        d, mo, y, hh, mi = map(int, m.groups())
        published = f"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mi:02d}:00"   # Börsenzeit (CET/CEST)
    return {"title": title, "company": company, "text": text, "published": published, **ids}


def fetch_article(url: str, timeout: int = 15) -> dict:
    """{title, company, text, isin, wkn} from a news URL, or {error}."""
    import requests
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        return {"error": f"URL nicht abrufbar: {type(e).__name__}"}
    return parse_article_html(r.text)


def _eodhd_get(path: str, **params):
    """GET https://eodhd.com/api/{path} with token; parsed JSON or None on any failure."""
    import requests
    from new_dataset.pipeline.eodhd_prices import _token
    tok = _token()
    if not tok:
        return None
    try:
        r = requests.get(f"https://eodhd.com/api/{path}",
                         params={"api_token": tok, "fmt": "json", **params}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def rank_candidates(results: list[dict], limit: int = 5) -> list[dict]:
    """EODHD-Suchtreffer → Kandidaten, Aktien bevorzugt, Venue-Ranking wie Scrapling.
    Zeilen ohne Type-Feld werden toleriert; ETFs/Fonds fliegen raus."""
    rows = [r for r in results
            if r.get("Code") and r.get("Exchange") and r.get("Type") in (None, "Common Stock")]
    def rank(row):
        ex = row.get("Exchange", "")
        if ex in _PREFERENCE:
            return _PREFERENCE.index(ex)
        if ex in _FLOOR:
            return 100
        return 50
    return [{"symbol": f"{r['Code']}.{r['Exchange']}", "name": r.get("Name", ""),
             "exchange": r["Exchange"], "country": r.get("Country", "")}
            for r in sorted(rows, key=rank)[:limit]]


def resolve_ticker(query: str, limit: int = 5) -> list[dict]:
    """EODHD symbol search → ranked candidates. [] on no match."""
    data = _eodhd_get(f"search/{query}")
    if not isinstance(data, list) or not data:
        return []
    return rank_candidates(data, limit=limit)


def close_before(symbol: str, published_local: str) -> dict | None:
    """Letzter 5-Minuten-Schlusskurs VOR dem Meldungszeitpunkt (analog zum Pre-Slot der
    Slot-Methodik) → {datetime, close}, oder None. published_local ist Börsenzeit (Berlin)."""
    import pandas as pd
    from new_dataset.pipeline.eodhd_prices import fetch_bars
    try:
        pub_utc = (pd.Timestamp(published_local).tz_localize("Europe/Berlin")
                   .tz_convert("UTC").tz_localize(None))
    except Exception:
        return None
    start = (pub_utc - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (pub_utc + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bars = fetch_bars(symbol, start, end)
    if bars is None or bars.empty:
        return None
    pre = bars[bars.index < pub_utc]["Close"].dropna()
    if pre.empty:
        return None
    ts_local = (pre.index[-1].tz_localize("UTC").tz_convert("Europe/Berlin")
                .strftime("%d.%m.%Y %H:%M"))
    return {"date": ts_local, "close": float(pre.iloc[-1])}


def latest_close(symbol: str) -> dict | None:
    """Newest EOD close for the symbol → {date, close}, or None."""
    data = _eodhd_get(f"eod/{symbol}", order="d")
    if not isinstance(data, list) or not data:
        return None
    row = data[0]
    if "close" not in row or "date" not in row:
        return None
    try:
        return {"date": row["date"], "close": float(row["close"])}
    except (TypeError, ValueError):
        return None


def advise(direction, proba: dict | None = None) -> dict:
    """Map model direction → recommendation. Pure."""
    if direction not in REACTION_DE:
        return {"aktion": "—", "konfidenz": None, "begründung": "keine Modellvorhersage"}
    konf = max(proba.values()) if proba else None
    return {"aktion": REACTION_DE[direction], "konfidenz": konf,
            "begründung": f"Das Modell erwartet die 10-Minuten-Kursrichtung „{_RICHTUNG_DE[direction]}“."}


def analyze_url(url: str = "", ticker_override: str = "", pasted_text: str = "") -> dict:
    """Orchestrate URL/text → sentiment + price-model recommendation + ticker/last-close.
    Partial-safe: advice comes from text even if ticker/price resolution fails."""
    from new_dataset.dashboard.data import live_sentiment
    from new_dataset.pipeline import persist
    notes: list[str] = []
    title, company, text = "", "", (pasted_text or "").strip()
    isin, published = None, None
    if text:
        isin = extract_ids(text)["isin"]
        company, title = split_company_title(text)
    else:
        art = fetch_article(url) if url.strip() else {"error": "Keine URL und kein Text angegeben."}
        if art.get("error"):
            notes.append(art["error"] + " Bitte den Artikeltext einfügen.")
            return {"title": "", "company": "", "ticker": "", "candidates": [], "close": None,
                    "sentiment": None, "direction": None, "proba": None, "advice": advise(None),
                    "disclaimer": DISCLAIMER, "notes": notes, "needs_text": True}
        title, company, text = art.get("title", ""), art.get("company", ""), art["text"]
        isin, published = art.get("isin"), art.get("published")

    from text_clean import clean
    cleaned = clean(text or "", "")
    sentiment = live_sentiment(text)
    pred = persist.predict([cleaned])[0]
    direction, proba = pred.get("label"), pred.get("proba")
    if "note" in pred:
        notes.append(pred["note"])

    candidates, ticker = [], (ticker_override or "").strip()
    if not ticker:
        for query in (isin, company, short_name(company), title or text[:60]):
            if not query:
                continue
            candidates = resolve_ticker(query)
            if candidates:
                ticker = candidates[0]["symbol"]
                break
        if not ticker:
            notes.append("Ticker nicht erkannt, bitte manuell eingeben.")
    close, close_kind = None, None
    if ticker:
        if published:
            close = close_before(ticker, published)
            close_kind = "pre_news"
        if close is None:
            close = latest_close(ticker)
            close_kind = "eod_latest" if close else None
            if published and close:
                notes.append("Kein Intraday-Kurs vor der Meldung verfügbar; EOD-Schluss angezeigt.")
    if ticker and close is None:
        notes.append("Kurs nicht verfügbar.")
    company = company or (candidates[0]["name"] if candidates else "")

    return {"title": title, "company": company, "ticker": ticker, "candidates": candidates,
            "close": close, "close_kind": close_kind, "published": published,
            "sentiment": sentiment, "direction": direction, "proba": proba,
            "advice": advise(direction, proba), "disclaimer": DISCLAIMER, "notes": notes,
            "needs_text": False, "text": text}
