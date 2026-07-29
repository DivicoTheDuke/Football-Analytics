# Provider integration

The repository includes a StatsBomb-style adapter as a concrete example of mapping external event JSON into the canonical event schema.

## Why adapters are isolated

Provider event names, outcomes, coordinate systems and possession definitions differ. Keeping each mapping in a dedicated module prevents provider-specific assumptions from leaking into research code.

## Production checklist

1. Store raw provider payloads unchanged.
2. Record provider version, fixture ID and ingestion time.
3. Reconcile match, team and player identities against master data.
4. Normalise coordinates and attacking direction consistently.
5. Validate event counts, shots, goals and lineups against source summaries.
6. Version every taxonomy mapping.
7. Block downstream publication when a quality gate fails.
8. Add contract tests using representative provider fixtures.

## Included example

`football_analytics.providers.statsbomb.normalise_statsbomb_events` maps a limited StatsBomb-style payload to the fields required by this project. It is intentionally small and must be extended before processing a full provider feed.
