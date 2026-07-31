# Football Analytics

A production-oriented **portfolio and demonstration project** for football event analytics, model governance and analyst-facing decision support. It combines Python, pandas, NumPy, scikit-learn, Plotly, Streamlit, FastAPI, pytest, Docker and GitHub Actions.

> **Honest scope:** this repository is not deployed at a professional club. The bundled matches, lineups, events, goals, roles, xG values and performance measures are synthetic. Current club and player names are sourced from the official Fantasy Premier League bootstrap endpoint and cached locally. FPL does not provide the historical event data required for sporting model validation.

## New forecasting workflows

The dashboard and API now include:

- next-season expected-points projection;
- fixture home/draw/away probabilities and most likely score;
- expected-goals forecast for each team;
- player goalscorer probabilities based on historical shot and xG shares;
- team style labels derived from passing, progression and territorial features;
- left, central and right attacking-channel tendencies;
- existing xG, xT, shot maps, momentum, passing networks, PPDA, field tilt, box entries, player profiles, clustering and similarity search.

Forecasts are probabilistic decision-support outputs, not facts. In demo mode they illustrate the software workflow only and are **not real Premier League predictions**.

## Recommended environment

Python **3.12** is the supported target. Python 3.14 is intentionally excluded because binary Data Science dependencies may not yet provide compatible wheels.

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

## Run the platform

```powershell
python -m streamlit run app.py
python -m uvicorn api:app --reload
```

CLI examples:

```powershell
football-analytics validate-data
football-analytics train-xg
football-analytics project-season --output reports\season_projection.csv
football-analytics build-report --match-id M0001
```

## Demo mode and historical mode

`config/app.toml` controls the data mode.

```toml
[data]
mode = "demo"
demo_directory = "data/demo"
historical_directory = "data/historical"
training_seasons = ["2021/22", "2022/23", "2023/24", "2024/25", "2025/26"]
evaluation_season = "2025/26"
```

For real five-season training, place legally usable, provider-normalised `events`, `matches` and `lineups` files in `data/historical`. CSV, JSON and Parquet events are supported; Parquet is recommended. Switch `mode` to `historical`, train the model, review the dataset manifest and only then start the application.

Expected evaluation design:

- development/training: 2021/22–2024/25;
- untouched temporal test: 2025/26;
- optional final refit on all five seasons only after evaluation;
- published metrics must remain the holdout metrics, not in-sample results.

## Architecture

```text
Provider data -> validation/normalisation -> curated dataset
             -> temporal xG evaluation -> versioned model artefacts
             -> xT/team-strength/style features
             -> Streamlit dashboard + FastAPI + reports
```

The application loads an existing `models/xg_model.joblib` when available. Otherwise demo mode trains a temporary model so the portfolio remains runnable. Historical mode should use an explicitly trained and reviewed artefact.

## API highlights

- `GET /health`
- `GET /matches`
- `GET /matches/{match_id}/summary`
- `GET /forecasts/season`
- `GET /forecasts/fixture?home_team=...&away_team=...`
- `GET /teams/{team}/identity`
- `GET /players/{player}/profile`
- `GET /players/{player}/similar`
- `POST /models/xg/predict`

## Limitations

Squad availability, transfers, injuries, promoted teams, manager changes, set-piece roles, expected minutes, schedule strength and provider-specific definitions materially affect a new-season forecast. The current transparent baseline does not pretend to know unavailable inputs. These should be added as versioned features only when a licensed and reproducible source exists.

See `docs/METHODOLOGY.md`, `docs/MODEL_CARD.md`, `docs/DATA_GOVERNANCE.md`, `docs/ARCHITECTURE.md` and `docs/PROVIDER_INTEGRATION.md`.

## One-time FootyStats Premier League import

FootyStats access is optional. The API key is never stored in the repository.
A complete league season may require several paginated HTTP requests, but the
command is guarded so the provider is contacted only during one explicit import
run. Subsequent dashboard recalculations use the local cache only.

```powershell
$env:FOOTYSTATS_API_KEY = "YOUR_API_KEY"

football-analytics import-footystats `
  --league-id PREMIER_LEAGUE_SEASON_ID
```

Repeat `--league-id` in the same command to import several licensed seasons:

```powershell
football-analytics import-footystats `
  --league-id SEASON_ID_1 `
  --league-id SEASON_ID_2 `
  --league-id SEASON_ID_3 `
  --league-id SEASON_ID_4 `
  --league-id SEASON_ID_5
```

The command creates local, Git-ignored files:

```text
data/provider/footystats/premier_league_raw.json
data/provider/footystats/premier_league_matches.csv
reports/footystats_season_projection.csv
reports/footystats_fixture_forecasts.csv
```

The Streamlit button **Recalculate from cached FootyStats data** never calls the
provider. It recalculates only from `premier_league_matches.csv`.

The documented `key=example&league_id=1625` data represents EPL 2018/19. It is
useful for integration testing, not for forecasting the current Premier League.

## Cached FootyStats machine-learning workflow

After the guarded provider import, the project trains match-level models locally. The provider is not contacted by model training or by the dashboard.

```powershell
$env:FOOTYSTATS_API_KEY = "example"
football-analytics import-footystats --league-id 1625
```

The import command caches the provider response and then trains the local match models. To retrain later without any provider request:

```powershell
football-analytics train-match-models
```

Generated artifacts:

```text
models/match_forecast_bundle.joblib
models/match_forecast_metadata.json
reports/ml_match_model_metrics.json
reports/ml_season_projection.csv
reports/ml_fixture_forecasts.csv
```

The match models use chronological pre-match rolling features, a logistic-regression outcome classifier and separate Poisson goal regressors. The dashboard's **Platform Overview** explains which capabilities are available from real aggregate match data, which remain synthetic demonstrations, and which require additional licensed event, tracking or player-availability data.

A cache containing only Premier League 2018/19 demonstrates the workflow but must not be presented as a current-season sporting forecast.
