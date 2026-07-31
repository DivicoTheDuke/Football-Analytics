from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


@dataclass(frozen=True)
class Settings:
    data_mode: str = "demo"
    demo_data_dir: Path = Path("data/demo")
    historical_data_dir: Path = Path("data/historical")
    events_file: str = "events.csv"
    matches_file: str = "matches.csv"
    lineups_file: str = "lineups.csv"
    training_seasons: tuple[str, ...] = ()
    evaluation_season: str | None = None
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    random_state: int = 42
    test_size: float = 0.25
    x_bins: int = 16
    y_bins: int = 12
    footystats_raw_cache: Path = Path("data/provider/footystats/premier_league_raw.json")
    footystats_matches_cache: Path = Path("data/provider/footystats/premier_league_matches.csv")
    match_model_bundle: Path = Path("models/match_forecast_bundle.joblib")

    @property
    def data_dir(self) -> Path:
        return self.historical_data_dir if self.data_mode == "historical" else self.demo_data_dir

    @property
    def is_demo(self) -> bool:
        return self.data_mode == "demo"


def load_settings(path: str | Path = "config/app.toml") -> Settings:
    values: dict = {}
    config_path = Path(path)
    if config_path.exists():
        with config_path.open("rb") as handle:
            values = tomllib.load(handle)
    data, model = values.get("data", {}), values.get("model", {})
    providers = values.get("providers", {})
    footystats = providers.get("footystats", {})
    mode = os.getenv("FOOTBALL_ANALYTICS_DATA_MODE", data.get("mode", "demo")).lower()
    if mode not in {"demo", "historical"}:
        raise ValueError("FOOTBALL_ANALYTICS_DATA_MODE must be 'demo' or 'historical'")
    return Settings(
        data_mode=mode,
        demo_data_dir=Path(os.getenv("FOOTBALL_ANALYTICS_DEMO_DATA_DIR", data.get("demo_directory", "data/demo"))),
        historical_data_dir=Path(os.getenv("FOOTBALL_ANALYTICS_HISTORICAL_DATA_DIR", data.get("historical_directory", "data/historical"))),
        events_file=str(data.get("events_file", "events.csv")),
        matches_file=str(data.get("matches_file", "matches.csv")),
        lineups_file=str(data.get("lineups_file", "lineups.csv")),
        training_seasons=tuple(str(v) for v in data.get("training_seasons", [])),
        evaluation_season=data.get("evaluation_season"),
        model_dir=Path(os.getenv("FOOTBALL_ANALYTICS_MODEL_DIR", "models")),
        report_dir=Path(os.getenv("FOOTBALL_ANALYTICS_REPORT_DIR", "reports")),
        random_state=int(model.get("random_state", 42)),
        test_size=float(model.get("test_size", 0.25)),
        x_bins=int(model.get("x_bins", 16)),
        y_bins=int(model.get("y_bins", 12)),
        footystats_raw_cache=Path(footystats.get("raw_cache", "data/provider/footystats/premier_league_raw.json")),
        footystats_matches_cache=Path(footystats.get("matches_cache", "data/provider/footystats/premier_league_matches.csv")),
        match_model_bundle=Path(model.get("match_model_bundle", "models/match_forecast_bundle.joblib")),
    )
