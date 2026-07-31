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
