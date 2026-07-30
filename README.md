# changePoint: Ad-hoc-Meldungen und Intraday-Marktreaktion

Sagt der Text einer deutschen Ad-hoc-Meldung die unmittelbare Kursreaktion
vorher? Dieses Repo misst die Reaktion (5 bis 60 Minuten) aus
5-Minuten-Kursdaten und sagt sie als 3-Klassen-Label (Auf/Ab/Flat) allein aus
dem Meldungstext vorher, mit German-FinBERT-Embeddings und klassischen
Klassifikatoren.

## Der schnellste Einstieg

`main.ipynb` ist ein fertig gerechneter Walkthrough: eine einzelne Meldung wird
vom Rohdatensatz bis zur Modellvorhersage verfolgt, danach folgen Gesamtbild
und Modellvergleich. Alle Zellen laufen aus den mitgelieferten Daten.

## Die Pipeline im Überblick

Die News liegen fertig bei; ein Scraper ist nicht Teil dieses Repos.

    adhoc_functioning.json        527 gelabelte Ad-hoc-Meldungen (2022 bis 2026)
    new_dataset/intraday/         5-Minuten-OHLCV-Cache, eine CSV je Meldung
            |
            v
    pipeline.build                Slot-KPIs, +-24h-Charts, Coverage-Report
    pipeline.figures              Abbildungen und Tabellen der Auswertung
    pipeline.train                5 Modelle: CV + Holdout -> model_results.csv
    sentiment_reaction            Korrelation Sentiment <-> Reaktion
    dashboard.app                 interaktives Gradio-Dashboard

## Setup

    python3 -m venv venv
    venv/bin/pip install -r requirements.txt      # Pipeline, Plots, Dashboard
    venv/bin/pip install -r requirements-ml.txt   # zusätzlich: Training, Sentiment, Live-Analyse (transformers, torch, scikit-learn, joblib)

## Alles reproduzieren

Jeder Schritt läuft vollständig aus dem mitgelieferten Cache; ein API-Key ist
dafür nicht nötig.

    venv/bin/python -m new_dataset.pipeline.build

Erwartete Schlusszeile: `[build] records=527 failed_fetch=0 functioning={'5': 527, '10': 527, '15': 525, '30': 523, '60': 517}`

    venv/bin/python -m new_dataset.pipeline.figures

Schreibt die Abbildungen nach `new_dataset/figures/`. Erwartete Schlusszeile:
`figures: 14 PNGs written, 4 tables, 1 skipped (perclass_f1_skip)`. Der Skip ist
konstruktionsbedingt und keine fehlende Abhängigkeit; die Sentiment-Abbildung
entsteht im Schritt `sentiment_reaction`.

    venv/bin/python -m new_dataset.pipeline.train

Trainiert alle fünf Modelle (Majority-Baseline, logistische Regression und
Random Forest auf Embeddings, vortrainiertes Sentiment, fine-getuntes
German-FinBERT) mit stratifizierter 5-fach-CV über drei Seeds und schreibt
`new_dataset/model_results.csv`. Seed 0 und `test_frac` = 0,15 sind fixiert;
alle fünf Modelle reproduzieren die mitgelieferten Werte (auf einem frischen
Clone verifiziert). Braucht die ML-Extras und je nach Rechner spürbar Zeit
(auf einem MacBook rund 75 Minuten).

    venv/bin/python -m new_dataset.sentiment_reaction

Berechnet die Sentiment-Reaktions-Korrelation neu (Spearman ρ ≈ 0,44 für die
10-Minuten-Reaktion).

    venv/bin/python -m new_dataset.dashboard.app

Startet das Dashboard auf http://127.0.0.1:7860 mit dem Demonstrator
„Artikel → Empfehlung“.

## Struktur

    new_dataset/pipeline/    Datenvalidierung, Slot-KPIs, Charts, Figuren, Training
    new_dataset/dashboard/   Gradio-App inkl. „Artikel → Empfehlung“-Demonstrator
    new_dataset/intraday/    5-Minuten-OHLCV-Cache (eine CSV je Meldung)
    new_dataset/charts/      ±24h-Detailgrafik je Meldung
    new_dataset/figures/     Abbildungen und Tabellen der Auswertung
    new_dataset/model_eval/  Konfusionsmatrizen und Per-Klassen-Metriken
    new_dataset/model_store/ persistiertes Deployment-Modell (logreg_embed)
    sentiment_models.py      Sentiment-Scorer (German-FinBERT)
    text_clean.py            Textbereinigung (EQS-Boilerplate)

## API-Key (optional)

Ein EODHD-Key ist nur für frische Kursdaten und die Live-Ticker-Auflösung im
Demonstrator nötig (Umgebungsvariable `EODHD_TOKEN` oder Datei
`apikey_free.txt` im Repo-Root). Alle Auswertungen oben laufen ohne.
