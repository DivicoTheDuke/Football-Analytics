# Provider Integration

The repository contains a small StatsBomb-style mapping example and a generic partitioned dataset loader. A production-quality integration must be implemented against a provider whose terms permit the intended use.

Normalised tables must expose the project's event schema, stable match and event IDs, season, competition, teams, players, coordinates, timestamps and explicit provenance fields. Coordinates must be transformed so every team attacks from x=0 toward x=105.

Recommended layout:

```text
data/historical/<provider>/<season>/events.parquet
data/historical/<provider>/<season>/matches.parquet
data/historical/<provider>/<season>/lineups.parquet
```

FPL `bootstrap-static` is not a five-season event provider. It is used only for current club/player reference names in the synthetic demo generator.

## FootyStats Premier League match-history cache

The FootyStats integration is deliberately separated into two operations:

1. `football-analytics import-footystats` is an explicit, guarded ingestion job.
   It downloads paginated league-match responses and refuses to call the provider
   again when the raw cache already exists, unless `--force` is deliberately used.
2. The Streamlit **Recalculate from cached FootyStats data** button performs no
   network request. It reads the normalized local CSV and recreates season and
   fixture forecast files under `reports/`.

FootyStats league IDs are season-specific and must be supplied explicitly. Do
not guess IDs or commit API keys. Set `FOOTYSTATS_API_KEY` only in the local
PowerShell process. The public `key=example` response documented by FootyStats
is an EPL 2018/19 example and is not evidence for a current-season forecast.

A complete season may be paginated, so one import execution can require several
HTTP requests. "One-time import" means one guarded batch run, not necessarily
one literal request.

The forecasting layer uses provider xG only when at least 80% of cached matches
contain home and away xG. Otherwise it labels and uses historical goals as a
scoring-rate proxy. Match-level data cannot support real player goalscorer,
passing-network, xT or attack-side forecasts; those require event-level data.
