# Sentient: Bengaluru Road Risk Intelligence

Sentient is an end-to-end geospatial ML system that predicts road infrastructure risk for Bengaluru from satellite and environmental stress signals, then surfaces actionable road-level prioritization via API and dashboard.

## Why this project exists

City teams need to answer:
- Which specific roads should we inspect first?
- Which zones are accumulating infrastructure stress?
- How do we turn raw satellite layers into defensible maintenance priorities?

Sentient addresses this with a reproducible data-to-decision pipeline.

## Core capabilities

- Multi-source ingestion: Sentinel-1/2, Landsat, ERA5, Nightlights, Population, OSM.
- Monthly stress feature engineering on a city grid, plus within-window temporal
  dynamics (deltas, trends, volatility), seasonality, and cross-stress interactions.
- Model sweep with automatic selection: Ridge baseline, HistGB, RandomForest,
  ExtraTrees, LightGBM, XGBoost, and a time-aware stacked ensemble
  (current best: `temporal_et` — ExtraTrees).
- CNN-temporal track (LSTM / TCN over image sequences) as a second opinion.
- Road-level ranking with per-road stress drivers, monthly risk trends,
  months-to-critical projections, and city summaries (`dashboard.json`).
- FastAPI serving both the JSON API and the interactive frontend.
- **SENTIENT Command Center** — an interactive map UI (`src/frontend/web/`)
  built for commissioners, not data scientists:
  - Real dark basemap (CARTO / OpenStreetMap) with localities and street names,
    risk-ranked roads overlaid as colored corridors (hover / click / search).
  - Hero risk snapshot with plain-language stats per city.
  - Road drill-down: priority score, stress drivers, trend sparkline vs the
    critical band, recommended action, months-to-critical countdown.
  - Time Machine: replay 2020–2024 month by month — roads re-color as their
    condition changes, driver meters (rainfall / standing water / heat) move,
    and a narrated caption explains what is degrading which part of the city.
  - Budget Planner: what-if slider that builds an auto-prioritized work order
    and estimates reactive cost avoided.
  - One-click Executive Brief (print / save as PDF).

## Project structure

```text
sentient/
  config/                  # pipeline config
  data/
    raw/                   # raw ingested data
    processed/             # monthly processed outputs
    features/              # model datasets and stats
    results/               # rankings, evaluation, manifests
  models/                  # trained models and metrics
  scripts/                 # run scripts and smoke checks
  src/
    ingestion/             # source-specific ingestors
    preprocessing/         # validation and monthly stress
    features/              # dataset and road-risk builders
    training/              # model training entrypoints
    inference/             # scoring modules
    evaluation/            # evaluation logic
    api/                   # FastAPI service
    frontend/              # Streamlit dashboard
  docs/                    # section-wise project docs
  mkdocs.yml               # docs site config (Material UI)
```

## Quick start

### 1) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2) Run full pipeline

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1 -ConfigPath "config/pipeline.bengaluru.2020_2024.json" -ModelTrack both
```

### 3) Run API

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1
```

### 4) Open the Command Center

The API serves the interactive frontend directly — start the API and open:

```text
http://localhost:8000
```

(The legacy Streamlit dashboard is still available via
`powershell -ExecutionPolicy Bypass -File scripts/run_frontend.ps1`.)

## Documentation (modern UI)

Full docs are in `docs/` and configured with MkDocs Material.

Install docs tooling:

```powershell
python -m pip install mkdocs mkdocs-material
```

Run docs locally:

```powershell
mkdocs serve
```

Open `http://127.0.0.1:8000`.

## API endpoints

- `GET /metadata`
- `GET /risk/latest`
- `GET /risk/ranking`
- `GET /risk/by_zone`
- `GET /risk/heatmap`
- `GET /risk/roads`
- `GET /dashboard` — full payload for the Command Center UI
- `GET /` — the Command Center itself

## Main outputs

- `data/results/risk_scores.parquet`
- `data/results/risk_scores_cnn_temporal.parquet`
- `data/results/road_risk_ranking.json`
- `data/results/dashboard.json`
- `data/results/evaluation.json`
- `data/results/data_inventory_manifest.json`

## Runbook

Detailed operational commands and stage-by-stage execution:

- `RUNBOOK.md`