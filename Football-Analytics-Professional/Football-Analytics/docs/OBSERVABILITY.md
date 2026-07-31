# Observability and monitoring

## Data monitoring

- matches received versus expected
- events per match and event-type distribution
- missing player, team and coordinate fields
- coordinate and timestamp validity
- goals and shots reconciled with official match summaries
- provider taxonomy changes

## Model monitoring

- prediction distribution by competition and month
- goal rate versus mean predicted probability
- Brier score and calibration drift
- feature distribution drift
- missing-category rate
- model version and training-window lineage

## Product monitoring

- API latency and error rate
- dashboard load time
- failed report generation
- most-used workflows
- stale-data warnings
- user feedback and unresolved metric-definition questions

## Incident response

A failed critical data-quality check should block publication. The system should retain the last known valid dataset, show a visible freshness warning and create an incident with match, provider and failed-check context.
