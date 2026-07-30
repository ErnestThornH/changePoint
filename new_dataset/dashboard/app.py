"""New-dataset Gradio dashboard (6 tabs). Thin UI over new_dataset.dashboard.data.
Farbige KPI-Karten, farbig codierte Handlungsempfehlung; Backend (data.py,
article.py) unverändert.
Run: python -m new_dataset.dashboard.app  ->  http://127.0.0.1:7860
"""
from pathlib import Path

import gradio as gr
import pandas as pd

from new_dataset.dashboard import data as dd
from new_dataset.dashboard import article

SENT_DOT = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
ADVICE_COLOR = {"Kaufen": "#22c55e", "Verkaufen": "#ef4444", "Halten": "#f59e0b"}


def _kpi_card(value: str, label: str, color: str = "#38bdf8") -> str:
    return (f'<div style="background:#1e293b;padding:18px 12px;border-radius:10px;'
            f'text-align:center;border:1px solid #334155;">'
            f'<div style="font-size:34px;font-weight:700;color:{color};line-height:1.1;">{value}</div>'
            f'<div style="color:#94a3b8;font-size:13px;margin-top:4px;">{label}</div></div>')


def _dist_card(f: dict) -> str:
    up, dn, fl = f["dist"]["up"], f["dist"]["down"], f["dist"]["flat"]
    row = ('<div style="display:flex;justify-content:space-around;">'
           f'<div><div style="font-size:26px;font-weight:700;color:#4ade80;">{up}</div>'
           '<div style="color:#94a3b8;font-size:12px;">Auf</div></div>'
           f'<div><div style="font-size:26px;font-weight:700;color:#f87171;">{dn}</div>'
           '<div style="color:#94a3b8;font-size:12px;">Ab</div></div>'
           f'<div><div style="font-size:26px;font-weight:700;color:#fbbf24;">{fl}</div>'
           '<div style="color:#94a3b8;font-size:12px;">Flat</div></div></div>')
    return (f'<div style="background:#1e293b;padding:18px 12px;border-radius:10px;'
            f'text-align:center;border:1px solid #334155;">{row}'
            '<div style="color:#94a3b8;font-size:13px;margin-top:6px;">Klassenverteilung</div></div>')


def _advice_html(adv: dict, richtung: str) -> str:
    color = ADVICE_COLOR.get(adv["aktion"], "#94a3b8")
    konf = f"P≈{adv['konfidenz']:.2f} · " if adv.get("konfidenz") is not None else ""
    return (f'<div style="background:{color}22;border-left:5px solid {color};'
            f'padding:16px 18px;border-radius:8px;">'
            f'<h2 style="color:{color};margin:0 0 6px 0;">Empfehlung: {adv["aktion"]}</h2>'
            f'<div style="color:#cbd5e1;">{adv["begründung"]}</div>'
            f'<div style="color:#94a3b8;font-size:13px;margin-top:8px;">'
            f'{konf}Richtung: {richtung}</div></div>')


def _sentiment_df(text: str) -> pd.DataFrame:
    s = dd.live_sentiment(text)
    if s.get("error"):
        return pd.DataFrame([["Hinweis", s["error"]]], columns=["Feld", "Wert"])
    dot = SENT_DOT.get(s["label"], "⚪")
    return pd.DataFrame([[f"{dot} German-FinBERT", s["label"], f"{s['score']:+.3f}"]],
                        columns=["Modell", "Label", "Score"])


def build_app() -> "gr.Blocks":
    overview = dd.load_overview()
    opts = dd.filter_options(overview)
    results = dd.model_results_table()
    f = dd.FINDING
    best = f"{f['best_test_macro_f1']:.2f}".replace(".", ",")
    majority = f"{f['majority_macro_f1']:.2f}".replace(".", ",")
    rho = str(f["spearman"]).replace(".", ",")

    with gr.Blocks(title="Ad-hoc Reaktion · BA Dashboard") as app:
        gr.Markdown("# Ad-hoc Reaktion · BA Dashboard")

        with gr.Tab("Übersicht"):
            gr.Markdown("## Leitfrage\nSagt der **Text** einer Ad-hoc-Meldung die unmittelbare "
                        "Kursreaktion vorher?")
            with gr.Row():
                gr.HTML(_kpi_card(f["n"], "Meldungen"))
                gr.HTML(_dist_card(f))
                gr.HTML(_kpi_card(best, f"Bestes Modell (Macro-F1)<br>vs. {majority} Majority", "#2dd4bf"))
                gr.HTML(_kpi_card(f"ρ≈{rho}", "Sentiment ↔ Reaktion", "#a78bfa"))

        with gr.Tab("Artikel → Empfehlung"):
            gr.Markdown("## Nachrichtenartikel → Empfehlung")
            gr.Markdown(f"<small>{article.DISCLAIMER}</small>")
            with gr.Row():
                a_url = gr.Textbox(label="Artikel-URL (vorerst nur EQS-News)",
                                   placeholder="https://www.eqs-news.com/de/news/ad-hoc/…", scale=3)
                a_tick = gr.Textbox(label="Ticker (optional)", scale=1)
            a_btn = gr.Button("Analysieren", variant="primary", size="lg")
            a_advice = gr.HTML()
            a_header = gr.Markdown()
            a_pub = gr.State(None)
            a_cands = gr.Dropdown(choices=[], label="Ticker-Kandidaten (korrigierbar)")
            with gr.Row():
                a_price = gr.Dataframe(label="Kurs", interactive=False)
                a_sent = gr.Dataframe(label="Sentiment (German-FinBERT)", interactive=False)
            a_notes = gr.Markdown()
            with gr.Accordion("Analysierter Artikeltext", open=False):
                a_textinfo = gr.Markdown()
                a_text = gr.Textbox(lines=10, interactive=False, show_label=False)

            def _price_df(ticker, close, kind=None):
                val = f"{close['close']} ({close['date']})" if close else "n/v"
                label = ("Kurs vor Ad-hoc-Meldung (letzter 5-Min-Schluss)" if kind == "pre_news"
                         else "Letzter Schluss (EOD, ohne Meldungsbezug)")
                return pd.DataFrame([["Ticker", ticker or "—"], [label, val]],
                                    columns=["Feld", "Wert"])

            def _analyze(url, ticker):
                r = article.analyze_url(url, ticker)
                hdr = f"### {r.get('company') or r.get('title') or 'Artikel'}"
                if r.get("title"):
                    hdr += f"\n\n{r['title']}"
                cands = [c["symbol"] for c in r.get("candidates", [])]
                s = r.get("sentiment") or {}
                if s.get("label"):
                    dot = SENT_DOT.get(s["label"], "⚪")
                    sent_df = pd.DataFrame([[f"{dot} German-FinBERT", s["label"], f"{s['score']:+.3f}"]],
                                           columns=["Modell", "Label", "Score"])
                else:
                    sent_df = pd.DataFrame([["Sentiment", s.get("error", "—")]], columns=["Feld", "Wert"])
                adv = r["advice"]
                richtung = dd.REACTION_DE.get(r.get("direction"), r.get("direction")) or "—"
                advice_html = _advice_html(adv, richtung)
                notes_md = "\n".join(f"- {n}" for n in r.get("notes", []))
                text = r.get("text", "")
                textinfo = (f"Extrahiert: {len(text)} Zeichen · das Reaktionsmodell nutzt die "
                            f"ersten 256 Tokens (≈ 200 Wörter), das Sentiment-Modell bis 512 Tokens.")
                return (advice_html, hdr,
                        gr.update(choices=cands, value=(cands[0] if cands else None)),
                        _price_df(r.get("ticker"), r.get("close"), r.get("close_kind")),
                        sent_df, notes_md, textinfo, text, r.get("published"))

            a_btn.click(_analyze, [a_url, a_tick],
                        [a_advice, a_header, a_cands, a_price, a_sent, a_notes, a_textinfo, a_text, a_pub])

            def _cand_price(sym, published):
                if not sym:
                    return _price_df(None, None)
                if published:
                    c = article.close_before(sym, published)
                    if c:
                        return _price_df(sym, c, "pre_news")
                return _price_df(sym, article.latest_close(sym), "eod_latest")

            a_cands.change(_cand_price, [a_cands, a_pub], a_price)

        with gr.Tab("Live-Analyse"):
            gr.Markdown("Das angezeigte Sentiment stammt vom Off-the-shelf-Modell (d) der Arbeit "
                        "(GermanFinBERT_SC_Sentiment); die Reaktionsvorhersage vom besten Modell (b), "
                        "logreg_embed.")
            txt = gr.Textbox(lines=8, label="Ad-hoc-Text einfügen (Deutsch)")
            with gr.Row():
                l_sent = gr.Dataframe(label="Sentiment (Modell d: GermanFinBERT_SC_Sentiment)", interactive=False)
                l_react = gr.Dataframe(label="Vorhergesagte Reaktion (Modell b: logreg_embed)", interactive=False)

            def _live(text):
                s_df = _sentiment_df(text)
                rec = dd.live_reaction(text)[0]
                if rec.get("note"):
                    r_df = pd.DataFrame([["Hinweis", rec["note"]]], columns=["Feld", "Wert"])
                else:
                    rows = [["Richtung", dd.REACTION_DE.get(rec["label"], rec["label"])]]
                    rows += [[f"P({dd.REACTION_DE.get(k, k)})", f"{v:.2f}"]
                             for k, v in rec["proba"].items()]
                    r_df = pd.DataFrame(rows, columns=["Feld", "Wert"])
                return s_df, r_df
            gr.Button("Analysieren", variant="primary").click(_live, txt, [l_sent, l_react])

        with gr.Tab("Datensatz"):
            with gr.Row():
                q = gr.Textbox(label="Suche (Firma / Ticker)", scale=2)
                r = gr.Dropdown(["alle", "up", "down", "flat"], value="alle", label="Reaktion")
                yr = gr.Dropdown(["alle"] + opts["years"], value="alle", label="Jahr")
                vn = gr.Dropdown(["alle"] + opts["venues"], value="alle", label="Börse")
            full = gr.State(overview)                       # gefiltertes df MIT id
            table = gr.Dataframe(value=dd.overview_display(overview), interactive=False,
                                 label="Meldungen (neueste zuerst), Zeile anklicken für Details")
            d_header = gr.Markdown()
            d_img = gr.Image(label="±24h Chart")
            with gr.Row():
                d_kpis = gr.Dataframe(label="Kennzahlen", interactive=False)
                d_sent = gr.Dataframe(label="Sentiment (German-FinBERT)", interactive=False)

            def _filter(query, reaction, year, venue):
                f_df = dd.filter_overview(dd.load_overview(), query, reaction, year, venue)
                return dd.overview_display(f_df), f_df       # Anzeige + State
            for comp in (q, r, yr, vn):
                comp.change(_filter, [q, r, yr, vn], [table, full])

            def _detail(full_df, evt: gr.SelectData):
                rid = full_df.iloc[int(evt.index[0])]["id"]   # id aus dem State, nicht aus der Anzeige
                det = dd.get_detail(rid)
                if det.get("error"):
                    return det["error"], None, pd.DataFrame(), pd.DataFrame()
                text = ((det.get("title") or "") + " " + (det.get("summary") or "")).strip()
                return dd.detail_header(det), det["chart_path"], dd.kpi_table(det["kpis"]), _sentiment_df(text)
            table.select(_detail, full, [d_header, d_img, d_kpis, d_sent])

        with gr.Tab("Reaktion"):
            gr.Markdown("## Kurs-Δ% vs. Volumen-Δ%: Slot-Methodik (Teil-Bar verworfen)")
            hsel = gr.Radio(["5", "10", "15", "30"], value="10", label="Horizont (Minuten)")
            simg = gr.Image(value=dd.scatter_path("10"), label="Scatter Kurs vs. Volumen")
            hsel.change(dd.scatter_path, hsel, simg)

        with gr.Tab("Modelle"):
            gr.Markdown("## Modellvergleich: 3 Klassen (Auf/Ab/Flat) @ 10 Min")
            gr.Dataframe(results, interactive=False, label="Ergebnisse (Macro-F1 als Leitmetrik)")
            msel = gr.Dropdown(["majority", "logreg_embed", "rf_embed",
                                "pretrained_sentiment", "finetuned_finbert"],
                               value="logreg_embed", label="Konfusionsmatrix wählen")
            cimg = gr.Image(value=dd.confusion_path("logreg_embed"), label="Konfusionsmatrix (Test)")
            msel.change(dd.confusion_path, msel, cimg)
            gr.Markdown("**Interpretation:** Das Off-the-shelf-Sentiment liegt über der "
                        "Majority-Baseline, aber unter den trainierten Modellen. Die gelernten "
                        "German-FinBERT-Embeddings (`logreg_embed`, `rf_embed`) tragen das "
                        "stärkste Signal. Die kleine Flat-Klasse (n=30) drückt den Macro-F1.")

    return app


if __name__ == "__main__":
    allowed = [str(Path("new_dataset").resolve())]
    theme = gr.themes.Monochrome(primary_hue="teal", neutral_hue="slate")
    build_app().launch(server_name="127.0.0.1", server_port=7860,
                       allowed_paths=allowed, theme=theme)
