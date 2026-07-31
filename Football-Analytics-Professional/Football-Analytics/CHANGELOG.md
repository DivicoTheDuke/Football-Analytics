# Changelog

## 1.0.0 – 2026-07-29

- Added provider-neutral event model and synthetic match generator
- Added data-quality controls
- Added xG training, evaluation, calibration and model persistence
- Added empirical xT model
- Added team, player, possession and passing-network analytics
- Added player similarity and clustering
- Added Streamlit dashboard, FastAPI service and HTML reporting
- Added tests, Docker, CI and governance documentation

## Fixture probability recalculation

- Replaced fixed home/away multipliers with venue-specific league and team xG rates.
- Added continuous recency weighting with a twelve-month half-life.
- Added sample-size shrinkage to prevent extreme probabilities for teams with few matches.
- Normalized the complete finite Poisson score matrix so home/draw/away probabilities sum exactly to 100%.
- Added both-teams-to-score and over-2.5-goals probabilities.
- Added regression tests for normalization and reversed home/away fixtures.

## 2.2.0

- Added a guarded, one-time FootyStats Premier League ingestion command.
- Added raw JSON and normalized local match-history caching.
- Added match-history forecasts using provider xG when available and goals as a labelled fallback.
- Added a Streamlit recalculation button that performs no network request.
- Prevented real goalscorer claims when only aggregate match data is available.
