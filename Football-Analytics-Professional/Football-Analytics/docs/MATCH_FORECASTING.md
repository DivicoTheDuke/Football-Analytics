# Match Forecasting and Capability Boundaries

## Local training workflow

The FootyStats provider is contacted only by the explicit `import-footystats` command. Provider responses are saved under `data/provider/footystats/`. Every later feature build, model training run, fixture forecast and season projection uses the cached CSV and makes no network request.

The dashboard action **Retrain ML models and recalculate locally** performs the following steps:

1. Load completed cached Premier League matches.
2. Create chronological pre-match features without target leakage.
3. Reserve the latest 25 percent of matches as a temporal holdout.
4. Train a multinomial logistic-regression outcome model.
5. Train separate Poisson regressors for home and away goals.
6. Combine classifier probabilities with a normalized Poisson score matrix.
7. Save model artifacts, metadata, metrics and forecast reports.

## Currently available from aggregate match data

- Home, draw and away probabilities
- Expected home and away goals
- Most likely score
- Both-teams-to-score probability
- Over-2.5 probability
- Expected-points season simulation
- Chronological model evaluation
- Rolling form and venue-specific scoring features

## Not supported by the current provider cache

The following capabilities require player-level event, tracking, availability or lineup data and are not inferred from aggregate match results:

- Real attacking-side preference
- Formation and build-up structure prediction
- Pressing height or defensive-block shape
- Player goalscorer probabilities
- Player role, minutes and lineup forecasts
- Pass networks, xT and progressive-action analysis from real matches

The repository keeps synthetic event workflows for these modules to demonstrate the application architecture. The UI labels them as synthetic and must not present them as real sporting analysis.

## Current-data limitation

A single historical season is sufficient to demonstrate the engineering and machine-learning workflow, but it is not sufficient for a credible forecast of a modern Premier League season. A stronger experiment should use multiple recent seasons, current promoted teams, transfers, expected minutes, injuries and coaching changes.
