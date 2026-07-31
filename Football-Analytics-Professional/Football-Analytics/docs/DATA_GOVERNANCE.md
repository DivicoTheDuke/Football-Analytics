# Data Governance

Raw provider data must remain immutable and outside public version control unless redistribution is explicitly permitted. Curated datasets must record source, provider version, ingestion date, competition, season, schema version and whether records are synthetic.

Real and synthetic performance records must not be silently mixed. Use separate directories and `data.mode`. Provider event and match IDs should remain traceable. A generated `models/dataset_manifest.json` records row counts, seasons, competitions and mode used for model training.

Minimum controls include unique event IDs, valid coordinates, valid timestamps, complete match relationships, consistent shot/goal flags, expected season coverage and no overlap between temporal train and test sets.

FPL club and player identities may change. The cached bootstrap file records a snapshot, not a historical roster database. Detailed roles are synthetic demo assignments because FPL exposes only GK, DEF, MID and FWD.
