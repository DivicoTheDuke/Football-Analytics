# Football Analytics Platform

A production-shaped portfolio project for football intelligence teams. It turns event data into decision-support products for analysts, coaches and recruitment staff through an interactive dashboard, a REST API, reproducible machine-learning pipelines and documented model governance.

> **Important:** the bundled dataset is synthetic. The architecture and analytics are realistic, but outputs are not suitable for sporting decisions until the system is connected to licensed club data and validated with domain experts.

## What the platform demonstrates

- End-to-end delivery from domain requirements to deployed user-facing products
- Data ingestion, validation, cleaning and feature engineering
- Expected goals modelling with match-aware train/test separation
- Model diagnostics including ROC AUC, PR AUC, Brier score, log loss and calibration
- Expected threat modelling from passes and carries
- Team performance metrics such as field tilt, PPDA, progression and shot quality
- Player profiling and similarity search for recruitment workflows
- Passing networks and possession-chain analysis
- Interactive Streamlit dashboard for non-technical users
- FastAPI endpoints for integration with internal tools
- Automated tests, Docker support and GitHub Actions CI
- Model cards, data governance, product requirements, experiment design and deployment documentation

## Product views

The Streamlit application contains six analyst workflows:

1. **Match Centre** – score, xG, possession, field tilt, PPDA and momentum
2. **Shot & xG Lab** – shot maps, shot quality and model predictions
3. **Possession & Threat** – expected threat by zone, player and action
4. **Passing Network** – team structure, central players and connection volume
5. **Player Recruitment** – profiles, percentile views and similarity search
6. **Model Governance** – evaluation metrics, calibration and limitations

## Technology stack

Python, pandas, NumPy, scikit-learn, SciPy, NetworkX, Plotly, Streamlit, FastAPI, Pydantic, pytest, Docker and GitHub Actions.

## Repository structure

```text
Football-Analytics/
├── app.py                         # Streamlit analyst product
├── api.py                         # FastAPI integration service
├── config/app.toml                # Runtime configuration
├── data/demo/                     # Synthetic demo data
├── docs/                          # Architecture, methods and governance
├── notebooks/                     # Reproducible walkthrough
├── src/football_analytics/
│   ├── cli.py                     # Command-line workflows
│   ├── data.py                    # Ingestion and filtering
│   ├── demo.py                    # Synthetic event generator
│   ├── features.py                # Football feature engineering
│   ├── metrics.py                 # Team and player metrics
│   ├── providers/                 # External data-provider adapters
│   ├── networks.py                # Passing-network analysis
│   ├── quality.py                 # Data-quality controls
│   ├── reporting.py               # Self-contained HTML reports
│   ├── scouting.py                # Player similarity
│   ├── service.py                 # Cached application services
│   ├── xg.py                      # Expected-goals model
│   └── xt.py                      # Expected-threat model
├── tests/                         # Automated tests
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install

```bash
pip install -e ".[dev]"
```

### 3. Generate or refresh demo data

```bash
football-analytics generate-demo --matches 28 --seed 42
```

### 4. Validate data

```bash
football-analytics validate-data
```

### 5. Train and evaluate xG

```bash
football-analytics train-xg
```

### 6. Run the dashboard

```bash
streamlit run app.py
```

### 7. Run the API

```bash
uvicorn api:app --reload
```

Interactive API documentation is then available at `/docs`.

### 8. Generate a match report

```bash
football-analytics build-report --match-id M0001
```

### 9. Run tests

```bash
pytest
```

## Docker

```bash
docker compose up --build
```

The dashboard runs on port `8501` and the API on port `8000`.

## Working with real data

The internal event model is provider-neutral. To connect real data, create an adapter that maps provider fields to the schema documented in [`data/README.md`](data/README.md). The project includes a tested StatsBomb-style adapter and integration guidance, but no proprietary or copyrighted dataset is bundled.

Before a club uses the outputs operationally, the following are mandatory:

- Validate event definitions with analysts and coaching staff
- Add provider-specific quality checks
- Re-train and calibrate models on representative competition data
- Perform temporal and competition-based out-of-sample testing
- Define access controls and retention policies
- Establish model monitoring, versioning and sign-off
- Review the user interface with the actual workflow owners

## Portfolio case study

This repository is intended to show how a research idea becomes a reliable internal product:

- Requirements are framed around analyst and coaching workflows
- Data is validated before analysis
- Models are evaluated and their limitations are visible
- Metrics are exposed through both a UI and an API
- Reusable code, tests and documentation support maintainability
- The design separates research code from production-facing services

## Author

Mischa Herzog
