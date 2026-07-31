# Research and experiment plan

## Objective

Determine whether the analytics improve analyst workflow quality, speed and consistency rather than merely producing attractive visualisations.

## Baselines

- Goals and shot counts without xG
- Simple distance-only xG model
- Current analyst workflow without the application

## Model experiments

1. Distance-only logistic regression
2. Distance and angle
3. Context features such as body part, pressure and play pattern
4. Calibration by competition and season
5. Gradient-boosted benchmark with the same split policy
6. Temporal holdout and competition holdout

## Evaluation

### Statistical

ROC AUC, average precision, log loss, Brier score, calibration error and confidence intervals by bootstrap.

### Football

Review high-residual shots with analysts, identify missing context and test whether errors are systematic by shot type, competition, player or provider.

### Product

Task completion time, number of clicks, analyst confidence, repeated-use rate, qualitative feedback and examples where the tool changed or clarified an interpretation.

## Guardrails

- Pre-register the evaluation window and metrics.
- Do not tune on the final holdout period.
- Keep provider and competition shifts visible.
- Compare against simple baselines.
- Require video review for tactical conclusions.
- Record disagreement rather than forcing a single interpretation.
