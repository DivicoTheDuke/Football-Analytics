# Methodology

## Forecasting objective

The forecast layer turns historical event evidence into transparent, analyst-reviewable priors. It is not a betting model and does not claim deterministic knowledge of future matches.

## Fixture model

1. The xG model assigns a probability to each shot.
2. Match xG is aggregated by team.
3. Seasonal observations receive exponential recency weights.
4. Team attack strength is weighted xG for divided by the league mean.
5. Team defence strength is weighted xG against divided by the league mean.
6. Home and away expected goals combine the two teams' strengths with modest home/away multipliers.
7. Independent Poisson score distributions produce home-win, draw and away-win probabilities.

This is intentionally interpretable. Stronger alternatives should be compared against it rather than replacing it without a baseline.

## New-season projection

Every ordered home/away pairing is simulated analytically. Expected points are the sum of `3 × win probability + draw probability`. The table is a model expectation, not a predicted final table.

## Goalscorer probability

A player's expected-goal share combines 70% historical xG share and 30% shot share. Player expected goals are converted to probability of at least one goal using `1 - exp(-lambda)`. A production model also needs expected minutes, selection probability, penalties, set pieces, injuries and transfer status.

## Attack-side tendency

Attacking-third passes, carries and shots are grouped into right, central and left pitch channels. The output describes event-location tendency, not tactical intent. Provider coordinate orientation must be normalised before use.

## Team style

Style labels are deterministic summaries of pass completion, forward pass distance, progressive-action rate, final-third rate and actions under pressure. They are communication aids, not objective tactical identities, and should be checked with video.

## Evaluation

Historical mode uses a temporal season holdout when the configured evaluation season is present. Grouped match splitting remains a safe fallback for demo and test fixtures. Metrics that are undefined for a single-class test sample are stored as null rather than fabricated.

## Fixture probability model

Fixture forecasts use venue-specific expected-goal baselines. Team home attack, home defence, away attack and away defence rates are estimated separately. Older matches receive exponentially lower weight with a twelve-month half-life. Estimates are shrunk toward the corresponding league venue average, reducing unstable forecasts when a team has only a small number of observed matches.

Expected home and away goals parameterize independent Poisson distributions. The displayed home-win, draw and away-win probabilities are calculated from a normalized score matrix covering 0–12 goals for each team. Normalization retains a valid 100% outcome distribution despite the finite score grid. The model also reports both-teams-to-score and over-2.5-goals probabilities.

These are model estimates, not certainties. In demo mode the underlying performances are synthetic and the results are demonstrations rather than real sporting forecasts.
