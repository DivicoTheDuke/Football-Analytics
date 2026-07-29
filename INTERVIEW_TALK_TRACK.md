# Methodology

## Expected goals

The xG baseline predicts the probability of a shot becoming a goal. Features include distance, visible goal angle, body part, play pattern, pressure, first-time status and whether the shot was assisted.

Evaluation uses a group-aware split by match. Reported metrics include ROC AUC, average precision, Brier score, log loss, accuracy, precision, recall and F1. Calibration is shown separately because a model can rank shots well while still producing unreliable probabilities.

## Expected threat

The pitch is divided into zones. For every zone, the model estimates:

- probability of shooting
- probability of moving the ball
- probability of scoring after a shot
- transition probability into every destination zone

Values are calculated iteratively until convergence. A completed pass or carry receives the difference between destination and origin xT.

## Team metrics

- **Field tilt:** share of completed final-third passes between both teams
- **PPDA:** opponent completed passes in deeper areas divided by defensive actions higher up the pitch
- **Progressive action:** completed pass or carry that reduces distance to goal materially or enters the final third
- **Box entry:** completed action entering the penalty area

Definitions differ across clubs and providers. They must be agreed with analysts before operational use.

## Player similarity

Player vectors are built from passing, progression, chance creation, threat, shooting and defensive metrics. Features are standardised and cosine distance is used to retrieve similar profiles. Real recruitment work must additionally control for minutes, role, age, league strength, physical qualities and video evidence.
