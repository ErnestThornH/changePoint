"""Pure boilerplate stripping for ad-hoc text before sentiment scoring (Step 5).
No I/O, stdlib only."""
from __future__ import annotations

import re

# Distributor prefix at the very start (e.g. "EQS-Adhoc:", "DGAP-News:", "pta123:").
_PREFIX_RE = re.compile(
    r"^\s*(EQS-Adhoc|EQS-News|DGAP-Adhoc|DGAP-News|Adhoc|pta\d*|Original-Research)\s*:\s*",
    re.IGNORECASE,
)

# Legal / distributor phrasing dropped anywhere in the text.
# NOTE: the broad Veröffentlichung pattern must come BEFORE the narrow
# "Ad-hoc-Mitteilung nach Art. 17 MAR" pattern, otherwise the narrow one
# strips the inner anchor first and the broader sentence match fails.
_LEGAL_RES = [
    re.compile(r"Veröffentlichung einer Ad-?hoc-?Mitteilung.{0,300}?(übermittelt durch[^.]*\.)",
               re.IGNORECASE | re.DOTALL),
    re.compile(r"Ad-?hoc-?Mitteilung nach\s*(§\s*15\s*WpHG|Art\.?\s*17\s*MAR)", re.IGNORECASE),
    re.compile(r"Für den Inhalt der Mitteilung ist[^.]*\.", re.IGNORECASE),
    re.compile(r"\b(Ad-?hoc )?announcement pursuant to (Article|Art\.?)[^,.:]*", re.IGNORECASE),
    re.compile(r"pursuant to (sec\.?|section)\s*15\s*WpHG", re.IGNORECASE),
]

# Metadata lines / tokens.
_META_RES = [
    re.compile(r"Schlagwort\(?e?\)?:.*?(?=\n|$)", re.IGNORECASE),
    re.compile(r"\bISIN:\s*[A-Z]{2}[A-Z0-9]{9}[0-9]\b", re.IGNORECASE),
    re.compile(r"\bWKN:\s*\w+\b", re.IGNORECASE),
    re.compile(r"\b(0?[1-9]|[12]\d|3[01])\.(0?[1-9]|1[0-2])\.\d{4}(\s+\d{1,2}:\d{2})?\b"),  # 11.05.2015 / 11.05.2015 08:38
]

# Run-together duplicates only (e.g. crawler's "Foo AGFoo AG" -> "Foo AG"). A
# space-separated repeat like "net net" cannot match: the separator is not part of
# the doubled unit, so the two copies aren't adjacent.
_DOUBLED_WORDS_RE = re.compile(r"\b(\w[\w.\- ]{1,40}?)\1\b")
_WS_RE = re.compile(r"\s+")


def _dedupe_doubled(text: str) -> str:
    # Run a couple of passes for adjacent duplicates produced by the crawler.
    for _ in range(2):
        new = _DOUBLED_WORDS_RE.sub(r"\1", text)
        if new == text:
            break
        text = new
    return text


def clean(title: str, summary: str) -> str:
    """Return model-ready text: title + summary with distributor/legal/metadata
    boilerplate removed and whitespace collapsed. May return '' if all boilerplate."""
    title = _PREFIX_RE.sub("", (title or "").strip())
    text = f"{title}. {summary or ''}"
    for rx in _LEGAL_RES + _META_RES:
        text = rx.sub(" ", text)
    text = _dedupe_doubled(text)
    text = _WS_RE.sub(" ", text).strip()
    # Drop a dangling leading period left by an emptied title.
    text = re.sub(r"^[.\s]+", "", text)
    return text


# ---- Volltext-Aufbereitung (EQS-Detailseiten-Content) -------------------------------
_TRIM_START_RE = re.compile(r"Für den Inhalt der Mitteilung ist der Emittent|übermittelt durch", re.I)
_TRIM_END_RE = re.compile(r"^(Kontakt:|Ende der Insiderinformation|Ende der Mitteilung)|Die EQS Distributionsservices", re.I)
_MAR_VARIANT_RE = re.compile(
    r"(?:Öffentliche Bekanntgabe|Veröffentlichung)\s+(?:von|einer)\s+Insiderinformation\w*\s+"
    r"(?:nach|gemäß)\s+Artikel\s*17(?:\s*Absatz\s*1)?\s*der\s*Verordnung\s*\(EU\)\s*"
    r"(?:Nr\.\s*)?596/2014(?:\s*über\s*Marktmissbrauch[^.]{0,120}?\.)?", re.IGNORECASE)
_KONTAKT_TAIL_RE = re.compile(r"(?:IR[- ]|Presse- und Investor-Relations-|Investor[- ]Relations[- ])?Kontakt:")


def trim_eqs_lines(lines: list[str]) -> list[str]:
    """EQS-Vorspann (bis MAR-Formel/Verantwortlichkeit) und -Fußteil (Kontakt, Metadaten)
    abschneiden; ohne Marker oder bei zu kurzem Ergebnis bleiben die Zeilen unverändert."""
    start, end = 0, len(lines)
    for i, ln in enumerate(lines):
        if _TRIM_START_RE.search(ln):
            start = i + 1
    for i in range(start, len(lines)):
        if _TRIM_END_RE.search(lines[i]):
            end = i
            break
    body = lines[start:end]
    return body if len(" ".join(body)) >= 100 else lines


def clean_fulltext(title: str, content: str) -> str:
    """Trainings-/Inferenztext: Titel + EQS-bereinigter Volltext (trim + MAR-Varianten +
    Kontaktblock am Ende), danach die Standard-Bereinigung clean()."""
    lines = [l.strip() for l in (content or "").splitlines() if l.strip()]
    body = "\n".join(trim_eqs_lines(lines))
    body = _MAR_VARIANT_RE.sub(" ", body)
    last = None
    for last in _KONTAKT_TAIL_RE.finditer(body):
        pass
    if last and last.start() > 0.6 * len(body):
        body = body[:last.start()]
    return clean(title or "", body)
