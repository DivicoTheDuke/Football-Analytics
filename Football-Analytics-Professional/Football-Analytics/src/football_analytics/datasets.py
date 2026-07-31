from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .data import load_events, load_matches, load_lineups

SUPPORTED_SUFFIXES = {".csv", ".json", ".parquet"}


def _read(path: Path, kind: str) -> pd.DataFrame:
    if kind == "events":
        return load_events(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if kind == "matches":
        return load_matches(path)
    if kind == "lineups":
        return load_lineups(path)
    raise ValueError(kind)


def load_partitioned_dataset(root: str | Path, kind: str, seasons: tuple[str, ...] = ()) -> pd.DataFrame:
    root = Path(root)
    candidates = [path for path in root.rglob(f"{kind}.*") if path.suffix.lower() in SUPPORTED_SUFFIXES]
    if not candidates:
        raise FileNotFoundError(f"No {kind} files found below {root}")
    frames = [_read(path, kind) for path in sorted(candidates)]
    frame = pd.concat(frames, ignore_index=True)
    if seasons and "season" in frame:
        frame = frame.loc[frame["season"].astype(str).isin(seasons)].copy()
    return frame


def build_dataset_manifest(events: pd.DataFrame, matches: pd.DataFrame, source: str, output: str | Path) -> Path:
    payload = {
        "source": source,
        "matches": int(matches["match_id"].nunique()),
        "events": int(len(events)),
        "seasons": sorted(events["season"].dropna().astype(str).unique().tolist()),
        "competitions": sorted(events["competition"].dropna().astype(str).unique().tolist()),
        "synthetic": bool(events.get("synthetic_data", pd.Series([False])).fillna(False).all()),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
