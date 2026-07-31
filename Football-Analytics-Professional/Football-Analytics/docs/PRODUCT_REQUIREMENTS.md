# Product requirements

## Primary users

### First-team analyst

Needs to move quickly from a match-level overview to the possessions and actions that explain a pattern, then connect quantitative evidence to video review.

### Coach

Needs concise, interpretable information with minimal statistical jargon and clear links to tactical questions.

### Recruitment analyst

Needs comparable player profiles, transparent similarity logic and enough context to avoid false equivalence across roles and competitions.

### Research engineer

Needs reproducible feature definitions, testable pipelines, model diagnostics and reusable interfaces for downstream products.

## Core user stories

- As an analyst, I can compare score, xG, field tilt, PPDA and box entries for a selected match.
- As an analyst, I can identify the highest-value progression actions and the players involved.
- As a coach, I can see shot locations and cumulative xG without reading model code.
- As a recruitment analyst, I can compare a player with statistically similar profiles and inspect the underlying metrics.
- As an engineer, I can call the same calculations through an API and generate a portable match report.
- As a model owner, I can inspect calibration, evaluation metrics and documented limitations.

## Acceptance criteria

- A match can be selected without editing code.
- All displayed metrics use one shared implementation.
- Data-quality errors are surfaced before model training.
- xG evaluation separates matches between train and test data.
- Predictions are bounded between zero and one.
- Every advanced metric has a written definition.
- The application identifies synthetic data and prevents it being mistaken for decision-grade evidence.
- Core analytics are covered by automated tests.
