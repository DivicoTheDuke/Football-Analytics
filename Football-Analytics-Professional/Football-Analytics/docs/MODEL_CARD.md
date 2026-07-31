# Model Card: Expected Goals and Forecast Layer

## Intended use

Portfolio demonstration and analyst decision support. Outputs support structured review, video analysis and scenario discussion. They are not guaranteed outcomes, medical advice, betting advice or evidence of professional-club deployment.

## Data modes

- **Demo:** synthetic matches, events, lineups, outcomes and performance values. FPL supplies current identity references only.
- **Historical:** externally supplied, licensed and provider-normalised event data. The repository does not bundle five seasons of proprietary event data.

## xG model

Logistic regression with standardised distance and angle, boolean context features, and one-hot encoded body part and play pattern. Class weighting addresses imbalance. The model is deliberately interpretable and suitable as a baseline.

## Evaluation

Preferred split: 2021/22–2024/25 training and 2025/26 test. Report ROC AUC, average precision, Brier score, log loss, precision, recall, F1 and calibration. Undefined metrics remain null. Match IDs may never overlap across splits.

## Forecast layer

Recency-weighted attack/defence xG strengths and Poisson score probabilities. Scorer probabilities allocate team xG using player xG and shot shares. Team style and channel tendency are descriptive features, not causal claims.

## Material limitations

The baseline omits injuries, transfers, expected minutes, promoted-team adjustment, manager changes, schedule timing, market information, tactical regime shifts and competition-strength normalisation. Synthetic demo results have no sporting validity.

## Human oversight

An analyst must review data quality, provider definitions, temporal leakage, calibration, squad assumptions and relevant video before communicating outputs.
