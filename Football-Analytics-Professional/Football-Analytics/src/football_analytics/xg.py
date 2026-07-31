from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import shot_frame

NUMERIC_FEATURES = ["distance_to_goal", "shot_angle"]
BOOLEAN_FEATURES = ["under_pressure", "first_time", "assisted"]
CATEGORICAL_FEATURES = ["body_part", "play_pattern"]
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


@dataclass
class XGArtifact:
    model: Pipeline
    metrics: dict
    calibration: pd.DataFrame
    test_predictions: pd.DataFrame


def build_pipeline(random_state: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES + BOOLEAN_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
    ])


def _split_indices(shots: pd.DataFrame, evaluation_season: str | None, random_state: int, test_size: float):
    if evaluation_season and "season" in shots and evaluation_season in set(shots["season"].astype(str)):
        test_mask = shots["season"].astype(str).eq(evaluation_season).to_numpy()
        train_idx, test_idx = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        if len(train_idx) and len(test_idx):
            return train_idx, test_idx, "temporal_season_holdout"
    splitter = GroupShuffleSplit(n_splits=20, test_size=test_size, random_state=random_state)
    X, y, groups = shots[ALL_FEATURES], shots["shot_goal"].astype(int), shots["match_id"]
    fallback = None
    for train_idx, test_idx in splitter.split(X, y, groups):
        fallback = (train_idx, test_idx)
        if y.iloc[train_idx].nunique() == 2 and y.iloc[test_idx].nunique() == 2:
            return train_idx, test_idx, "group_holdout"
    assert fallback is not None
    return fallback[0], fallback[1], "group_holdout_single_class_test"


def train_xg(events: pd.DataFrame, random_state: int = 42, test_size: float = 0.25, evaluation_season: str | None = None) -> XGArtifact:
    shots = shot_frame(events)
    if shots["shot_goal"].nunique() < 2:
        raise ValueError("Training data requires both goals and non-goals")
    X = shots[ALL_FEATURES].copy()
    X[BOOLEAN_FEATURES] = X[BOOLEAN_FEATURES].astype(int)
    y = shots["shot_goal"].astype(int)
    train_idx, test_idx, split_strategy = _split_indices(shots, evaluation_season, random_state, test_size)
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    if y_train.nunique() < 2:
        raise ValueError("Training split requires goals and non-goals")
    model = build_pipeline(random_state)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
    two_classes = y_test.nunique() == 2
    metrics = {
        "split_strategy": split_strategy,
        "evaluation_season": evaluation_season,
        "training_seasons": sorted(shots.iloc[train_idx].get("season", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "train_shots": int(len(train_idx)), "test_shots": int(len(test_idx)),
        "test_matches": int(shots.iloc[test_idx]["match_id"].nunique()),
        "goal_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if two_classes else None,
        "average_precision": float(average_precision_score(y_test, probabilities)) if y_test.sum() else None,
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }
    if two_classes and len(y_test) >= 8:
        prob_true, prob_pred = calibration_curve(y_test, probabilities, n_bins=min(8, len(y_test)), strategy="quantile")
        calibration = pd.DataFrame({"predicted_probability": prob_pred, "observed_goal_rate": prob_true})
    else:
        calibration = pd.DataFrame(columns=["predicted_probability", "observed_goal_rate"])
    prediction_columns = [c for c in ["event_id", "match_id", "team", "player", "x", "y", "shot_goal", "season"] if c in shots]
    test_predictions = shots.iloc[test_idx][prediction_columns].copy()
    test_predictions["xg"] = probabilities
    return XGArtifact(model, metrics, calibration, test_predictions)


def predict_xg(model: Pipeline, events: pd.DataFrame) -> pd.DataFrame:
    shots = shot_frame(events)
    X = shots[ALL_FEATURES].copy()
    X[BOOLEAN_FEATURES] = X[BOOLEAN_FEATURES].astype(int)
    shots["xg"] = model.predict_proba(X)[:, 1]
    return shots


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    return pd.DataFrame({"feature": preprocessor.get_feature_names_out(), "coefficient": model.named_steps["classifier"].coef_[0]}).sort_values("coefficient", ascending=False)


def save_artifact(artifact: XGArtifact, model_dir: str | Path) -> tuple[Path, Path]:
    model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = model_dir / "xg_model.joblib", model_dir / "xg_model_metadata.json"
    joblib.dump(artifact.model, model_path)
    metadata_path.write_text(json.dumps(artifact.metrics, indent=2), encoding="utf-8")
    artifact.calibration.to_csv(model_dir / "xg_calibration.csv", index=False)
    return model_path, metadata_path
