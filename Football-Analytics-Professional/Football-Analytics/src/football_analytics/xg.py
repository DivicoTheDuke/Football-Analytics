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
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    log_loss, precision_recall_fscore_support, roc_auc_score,
)
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
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES + BOOLEAN_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
    ])


def train_xg(events: pd.DataFrame, random_state: int = 42, test_size: float = 0.25) -> XGArtifact:
    shots = shot_frame(events)
    if shots["shot_goal"].nunique() < 2:
        raise ValueError("Training data requires both goals and non-goals")

    X = shots[ALL_FEATURES].copy()
    X[BOOLEAN_FEATURES] = X[BOOLEAN_FEATURES].astype(int)
    y = shots["shot_goal"].astype(int)
    groups = shots["match_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = build_pipeline(random_state)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    metrics = {
        "train_shots": int(len(train_idx)),
        "test_shots": int(len(test_idx)),
        "test_matches": int(shots.iloc[test_idx]["match_id"].nunique()),
        "goal_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    prob_true, prob_pred = calibration_curve(y_test, probabilities, n_bins=8, strategy="quantile")
    calibration = pd.DataFrame({"predicted_probability": prob_pred, "observed_goal_rate": prob_true})

    predictions_frame = shots.iloc[test_idx][[
        "event_id", "match_id", "team", "player", "x", "y", "shot_goal"
    ]].copy()
    predictions_frame["xg"] = probabilities

    return XGArtifact(model=model, metrics=metrics, calibration=calibration, test_predictions=predictions_frame)


def predict_xg(model: Pipeline, events: pd.DataFrame) -> pd.DataFrame:
    shots = shot_frame(events)
    X = shots[ALL_FEATURES].copy()
    X[BOOLEAN_FEATURES] = X[BOOLEAN_FEATURES].astype(int)
    shots["xg"] = model.predict_proba(X)[:, 1]
    return shots


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    return pd.DataFrame({"feature": feature_names, "coefficient": coefficients}).sort_values(
        "coefficient", ascending=False
    )


def save_artifact(artifact: XGArtifact, model_dir: str | Path) -> tuple[Path, Path]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xg_model.joblib"
    metadata_path = model_dir / "xg_model_metadata.json"
    joblib.dump(artifact.model, model_path)
    metadata_path.write_text(json.dumps(artifact.metrics, indent=2), encoding="utf-8")
    artifact.calibration.to_csv(model_dir / "xg_calibration.csv", index=False)
    return model_path, metadata_path
