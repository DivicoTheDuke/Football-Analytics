from __future__ import annotations

from functools import lru_cache
import joblib

from .config import load_settings
from .data import load_events, load_matches, load_lineups
from .features import add_spatial_features
from .xg import XGArtifact, predict_xg, train_xg
from .xt import train_xt, apply_xt


@lru_cache(maxsize=1)
def datasets():
    settings = load_settings()
    events = load_events(settings.data_dir / settings.events_file)
    matches = load_matches(settings.data_dir / settings.matches_file)
    lineups = load_lineups(settings.data_dir / settings.lineups_file)
    return matches, events, lineups


@lru_cache(maxsize=1)
def models():
    settings = load_settings()
    _, events, _ = datasets()
    model_path = settings.model_dir / "xg_model.joblib"
    metadata_path = settings.model_dir / "xg_model_metadata.json"
    calibration_path = settings.model_dir / "xg_calibration.csv"
    if model_path.exists() and metadata_path.exists():
        import json, pandas as pd
        artifact = XGArtifact(joblib.load(model_path), json.loads(metadata_path.read_text(encoding="utf-8")), pd.read_csv(calibration_path) if calibration_path.exists() else pd.DataFrame(), pd.DataFrame())
    else:
        artifact = train_xg(events, settings.random_state, settings.test_size, settings.evaluation_season if not settings.is_demo else None)
    return artifact, train_xt(events, settings.x_bins, settings.y_bins)


@lru_cache(maxsize=1)
def enriched_events():
    _, events, _ = datasets(); xg_artifact, xt_model = models()
    enriched = apply_xt(xt_model, add_spatial_features(events))
    return enriched, predict_xg(xg_artifact.model, events)


def clear_cache():
    datasets.cache_clear(); models.cache_clear(); enriched_events.cache_clear()
