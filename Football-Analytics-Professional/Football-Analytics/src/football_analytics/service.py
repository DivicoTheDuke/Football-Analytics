from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pandas as pd

from .config import load_settings
from .data import load_events, load_matches, load_lineups
from .features import add_spatial_features
from .xg import train_xg, predict_xg
from .xt import train_xt, apply_xt


@lru_cache(maxsize=1)
def datasets():
    settings = load_settings()
    events = load_events(settings.data_dir / "events.csv")
    matches = load_matches(settings.data_dir / "matches.csv")
    lineups = load_lineups(settings.data_dir / "lineups.csv")
    return matches, events, lineups


@lru_cache(maxsize=1)
def models():
    settings = load_settings()
    _, events, _ = datasets()
    xg_artifact = train_xg(events, settings.random_state, settings.test_size)
    xt_model = train_xt(events, settings.x_bins, settings.y_bins)
    return xg_artifact, xt_model


@lru_cache(maxsize=1)
def enriched_events():
    _, events, _ = datasets()
    xg_artifact, xt_model = models()
    enriched = add_spatial_features(events)
    enriched = apply_xt(xt_model, enriched)
    shots = predict_xg(xg_artifact.model, events)
    return enriched, shots


def clear_cache():
    datasets.cache_clear()
    models.cache_clear()
    enriched_events.cache_clear()
