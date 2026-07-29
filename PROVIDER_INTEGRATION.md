# Data contract

The bundled demo data is synthetic and uses a provider-neutral event schema.

## `events.csv`

Key fields:

- `event_id` – unique event identifier
- `match_id` – match identifier
- `team`, `opponent` – event ownership
- `period`, `minute`, `second`, `timestamp_seconds` – chronology
- `possession_id` – possession-chain identifier
- `player`, `position`, `recipient` – participants
- `event_type` – Pass, Carry, Shot, Duel, Interception, Recovery, Pressure or Foul
- `outcome` – event result
- `x`, `y`, `end_x`, `end_y` – provider-normalised coordinates on a 105 x 68 pitch
- `body_part`, `play_pattern` – shot and possession context
- `under_pressure`, `first_time`, `assisted`, `key_pass`, `shot_goal` – Boolean context

## Provider adapters

Real providers use different taxonomies and coordinate systems. Create a dedicated adapter that:

1. maps event names and outcomes
2. transforms coordinates to 105 x 68
3. resolves player and team identifiers
4. constructs or validates possession IDs
5. retains the provider event identifier for traceability
6. records source version and ingestion timestamp

Never silently mix provider definitions. Store raw source data separately from curated analytics tables.
