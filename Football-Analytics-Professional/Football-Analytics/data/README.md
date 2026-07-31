# Data directories

- `demo/`: public synthetic demonstration records.
- `reference/`: cached reference snapshots such as FPL bootstrap data.
- `historical/`: user-supplied, licensed historical data; not included.
- `raw/`: optional immutable provider extracts; do not commit restricted data.
- `curated/`: optional normalised Parquet datasets.

The application expects `events`, `matches` and `lineups` files. Historical events should include competition, season, match date, stable IDs and a `synthetic_data`/provenance indicator where possible.
