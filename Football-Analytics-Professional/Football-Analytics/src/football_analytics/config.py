from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data/demo")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    random_state: int = 42
    test_size: float = 0.25
    x_bins: int = 16
    y_bins: int = 12


def load_settings(path: str | Path = "config/app.toml") -> Settings:
    config_path = Path(path)
    values: dict = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            values = tomllib.load(handle)

    data = values.get("data", {})
    model = values.get("model", {})

    return Settings(
        data_dir=Path(os.getenv("FOOTBALL_ANALYTICS_DATA_DIR", data.get("directory", "data/demo"))),
        model_dir=Path(os.getenv("FOOTBALL_ANALYTICS_MODEL_DIR", "models")),
        report_dir=Path(os.getenv("FOOTBALL_ANALYTICS_REPORT_DIR", "reports")),
        random_state=int(model.get("random_state", 42)),
        test_size=float(model.get("test_size", 0.25)),
        x_bins=int(model.get("x_bins", 16)),
        y_bins=int(model.get("y_bins", 12)),
    )
