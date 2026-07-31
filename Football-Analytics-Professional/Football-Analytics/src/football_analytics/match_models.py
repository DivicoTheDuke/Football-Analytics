from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import exp, factorial
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .match_features import FEATURE_COLUMNS, build_prematch_features, latest_fixture_features


OUTCOME_CLASSES = ("A", "D", "H")


@dataclass(frozen=True)
class MatchModelMetrics:
    train_matches: int
    test_matches: int
    split_date: str
    outcome_log_loss: float
    outcome_accuracy: float
    home_goals_mae: float
    away_goals_mae: float
    home_poisson_deviance: float
    away_poisson_deviance: float


@dataclass(frozen=True)
class MatchModelBundle:
    outcome_model: Pipeline
    home_goals_model: Pipeline
    away_goals_model: Pipeline
    feature_columns: tuple[str, ...]
    metrics: MatchModelMetrics
    training_seasons: tuple[str, ...]
    trained_at_utc: str
    training_match_count: int


@dataclass(frozen=True)
class MLFixtureForecast:
    home_team: str
    away_team: str
    home_expected_goals: float
    away_expected_goals: float
    home_win: float
    draw: float
    away_win: float
    most_likely_score: str
    both_teams_to_score: float
    over_2_5: float
    model_source: str


def _numeric_pipeline(model: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _temporal_split(frame: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    split_index = int(np.floor(len(ordered) * (1.0 - test_fraction)))
    split_index = min(max(split_index, 1), len(ordered) - 1)
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def train_match_models(
    matches: pd.DataFrame,
    *,
    random_state: int = 42,
    test_fraction: float = 0.25,
) -> MatchModelBundle:
    built = build_prematch_features(matches)
    frame = built.frame
    if len(frame) < 80:
        raise ValueError("At least 80 completed matches are required for the ML workflow")

    train, test = _temporal_split(frame, test_fraction=test_fraction)
    x_train = train[list(FEATURE_COLUMNS)]
    x_test = test[list(FEATURE_COLUMNS)]

    # Logistic regression is intentionally retained as an explainable probability baseline.
    outcome_model = _numeric_pipeline(
        LogisticRegression(max_iter=2000, random_state=random_state)
    )
    home_model = _numeric_pipeline(PoissonRegressor(alpha=0.4, max_iter=1000))
    away_model = _numeric_pipeline(PoissonRegressor(alpha=0.4, max_iter=1000))

    outcome_model.fit(x_train, train["outcome"])
    home_model.fit(x_train, train["home_goals"])
    away_model.fit(x_train, train["away_goals"])

    outcome_probability = outcome_model.predict_proba(x_test)
    outcome_prediction = outcome_model.predict(x_test)
    classes = tuple(str(value) for value in outcome_model.named_steps["model"].classes_)
    home_prediction = np.clip(home_model.predict(x_test), 0.05, 5.0)
    away_prediction = np.clip(away_model.predict(x_test), 0.05, 5.0)

    metrics = MatchModelMetrics(
        train_matches=len(train),
        test_matches=len(test),
        split_date=str(test["match_date"].min().date()),
        outcome_log_loss=float(log_loss(test["outcome"], outcome_probability, labels=list(classes))),
        outcome_accuracy=float(accuracy_score(test["outcome"], outcome_prediction)),
        home_goals_mae=float(mean_absolute_error(test["home_goals"], home_prediction)),
        away_goals_mae=float(mean_absolute_error(test["away_goals"], away_prediction)),
        home_poisson_deviance=float(
            mean_poisson_deviance(test["home_goals"], np.clip(home_prediction, 1e-6, None))
        ),
        away_poisson_deviance=float(
            mean_poisson_deviance(test["away_goals"], np.clip(away_prediction, 1e-6, None))
        ),
    )

    return MatchModelBundle(
        outcome_model=outcome_model,
        home_goals_model=home_model,
        away_goals_model=away_model,
        feature_columns=tuple(FEATURE_COLUMNS),
        metrics=metrics,
        training_seasons=tuple(sorted(frame["season"].astype(str).unique())),
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        training_match_count=len(frame),
    )


def save_match_model_bundle(bundle: MatchModelBundle, model_dir: str | Path) -> tuple[Path, Path]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / "match_forecast_bundle.joblib"
    metadata_path = model_dir / "match_forecast_metadata.json"
    joblib.dump(bundle, bundle_path)
    metadata = {
        "trained_at_utc": bundle.trained_at_utc,
        "training_seasons": list(bundle.training_seasons),
        "training_match_count": bundle.training_match_count,
        "feature_columns": list(bundle.feature_columns),
        "metrics": asdict(bundle.metrics),
        "limitations": [
            "Current FootyStats league-match cache contains aggregate match data, not event or tracking data.",
            "Player-level goalscorer, formation, pressing and attacking-channel forecasts require additional licensed data.",
            "A single historical season is suitable for workflow demonstration but not a current-season sporting claim.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return bundle_path, metadata_path


def load_match_model_bundle(path: str | Path) -> MatchModelBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Match model bundle not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, MatchModelBundle):
        raise TypeError("Loaded object is not a MatchModelBundle")
    return bundle


def _poisson(goals: int, rate: float) -> float:
    return exp(-rate) * rate**goals / factorial(goals)


def _score_matrix(home_rate: float, away_rate: float, max_goals: int = 12) -> np.ndarray:
    matrix = np.array(
        [
            [_poisson(i, home_rate) * _poisson(j, away_rate) for j in range(max_goals + 1)]
            for i in range(max_goals + 1)
        ],
        dtype=float,
    )
    return matrix / matrix.sum()


def forecast_with_match_models(
    bundle: MatchModelBundle,
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> MLFixtureForecast:
    features = latest_fixture_features(matches, home_team, away_team)
    expected_home = float(np.clip(bundle.home_goals_model.predict(features)[0], 0.15, 4.5))
    expected_away = float(np.clip(bundle.away_goals_model.predict(features)[0], 0.15, 4.5))

    # Score-derived probabilities are coherent with the displayed expected goals.
    matrix = _score_matrix(expected_home, expected_away)
    poisson_home = float(np.tril(matrix, -1).sum())
    poisson_draw = float(np.trace(matrix))
    poisson_away = float(np.triu(matrix, 1).sum())

    classifier_prob = bundle.outcome_model.predict_proba(features)[0]
    classes = [str(value) for value in bundle.outcome_model.named_steps["model"].classes_]
    classifier_map = dict(zip(classes, classifier_prob, strict=True))

    # Blend an independently trained classifier with the goal-model distribution.
    home_win = 0.5 * poisson_home + 0.5 * float(classifier_map.get("H", 0.0))
    draw = 0.5 * poisson_draw + 0.5 * float(classifier_map.get("D", 0.0))
    away_win = 0.5 * poisson_away + 0.5 * float(classifier_map.get("A", 0.0))
    total = home_win + draw + away_win
    home_win, draw, away_win = home_win / total, draw / total, away_win / total

    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    return MLFixtureForecast(
        home_team=home_team,
        away_team=away_team,
        home_expected_goals=expected_home,
        away_expected_goals=expected_away,
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        most_likely_score=f"{i}-{j}",
        both_teams_to_score=float(matrix[1:, 1:].sum()),
        over_2_5=float(
            sum(
                matrix[x, y]
                for x in range(matrix.shape[0])
                for y in range(matrix.shape[1])
                if x + y >= 3
            )
        ),
        model_source="scikit-learn hybrid: logistic outcome + Poisson goal regressors",
    )


def ml_season_projection(bundle: MatchModelBundle, matches: pd.DataFrame) -> pd.DataFrame:
    latest_season = matches.sort_values("match_date")["season"].dropna().astype(str).iloc[-1]
    latest = matches[matches["season"].astype(str).eq(latest_season)]
    teams = sorted(set(latest["home_team"]) | set(latest["away_team"]))
    rows = {
        team: {
            "team": team,
            "points": 0.0,
            "expected_goals": 0.0,
            "expected_goals_against": 0.0,
        }
        for team in teams
    }
    for home in teams:
        for away in teams:
            if home == away:
                continue
            forecast = forecast_with_match_models(bundle, matches, home, away)
            rows[home]["points"] += 3.0 * forecast.home_win + forecast.draw
            rows[away]["points"] += 3.0 * forecast.away_win + forecast.draw
            rows[home]["expected_goals"] += forecast.home_expected_goals
            rows[home]["expected_goals_against"] += forecast.away_expected_goals
            rows[away]["expected_goals"] += forecast.away_expected_goals
            rows[away]["expected_goals_against"] += forecast.home_expected_goals
    result = pd.DataFrame(rows.values())
    result["goal_difference"] = result["expected_goals"] - result["expected_goals_against"]
    result = result.sort_values(
        ["points", "goal_difference", "expected_goals"], ascending=False
    ).reset_index(drop=True)
    result.insert(0, "projected_position", np.arange(1, len(result) + 1))
    return result
