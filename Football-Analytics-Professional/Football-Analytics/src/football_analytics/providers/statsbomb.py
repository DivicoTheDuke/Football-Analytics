from __future__ import annotations

from collections.abc import Iterable
import pandas as pd


def _name(value, default=""):
    if isinstance(value, dict):
        return value.get("name", default)
    return default if value is None else str(value)


def _location(value, default=(0.0, 0.0)):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return default
    return float(value[0]), float(value[1])


def _normalise_xy(location):
    """Convert StatsBomb 120 x 80 coordinates to a 105 x 68 pitch."""
    x, y = _location(location)
    return max(0.0, min(105.0, x / 120.0 * 105.0)), max(0.0, min(68.0, y / 80.0 * 68.0))


def normalise_statsbomb_events(
    raw_events: Iterable[dict],
    *,
    match_id: str,
    competition: str = "Unknown competition",
    season: str = "Unknown season",
    match_date: str = "",
    home_team: str = "",
    away_team: str = "",
) -> pd.DataFrame:
    """Map StatsBomb-style event JSON to the canonical repository event schema.

    The function intentionally preserves only fields used by this portfolio project.
    A production adapter should also retain the complete raw payload and provider version.
    """
    rows = []
    for index, event in enumerate(raw_events, start=1):
        event_type = _name(event.get("type"), "Unknown")
        team = _name(event.get("team"))
        opponent = away_team if team == home_team else home_team if team == away_team else ""
        start_x, start_y = _normalise_xy(event.get("location"))

        detail = event.get(event_type.lower(), {}) if isinstance(event.get(event_type.lower()), dict) else {}
        if event_type == "Pass":
            detail = event.get("pass", {}) or {}
            end_x, end_y = _normalise_xy(detail.get("end_location", event.get("location")))
            recipient = _name(detail.get("recipient"))
            outcome = _name(detail.get("outcome"), "Complete")
            key_pass = bool(detail.get("shot_assist", False) or detail.get("goal_assist", False))
            body_part = _name(detail.get("body_part"))
        elif event_type == "Carry":
            detail = event.get("carry", {}) or {}
            end_x, end_y = _normalise_xy(detail.get("end_location", event.get("location")))
            recipient = ""
            outcome = "Complete"
            key_pass = False
            body_part = ""
        elif event_type == "Shot":
            detail = event.get("shot", {}) or {}
            end_x, end_y = _normalise_xy(detail.get("end_location", [120, 40]))
            recipient = ""
            outcome = _name(detail.get("outcome"), "Unknown")
            key_pass = False
            body_part = _name(detail.get("body_part"))
        else:
            end_x, end_y = start_x, start_y
            recipient = ""
            outcome = _name(detail.get("outcome"), "Complete")
            key_pass = False
            body_part = ""

        minute = int(event.get("minute", 0))
        second = int(event.get("second", 0))
        timestamp_seconds = minute * 60 + second
        shot_goal = event_type == "Shot" and outcome == "Goal"

        rows.append({
            "event_id": str(event.get("id", f"{match_id}-{index}")),
            "match_id": match_id,
            "competition": competition,
            "season": season,
            "match_date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "team": team,
            "opponent": opponent,
            "period": int(event.get("period", 1)),
            "minute": minute,
            "second": second,
            "timestamp_seconds": timestamp_seconds,
            "possession_id": int(event.get("possession", index)),
            "player": _name(event.get("player"), "Unknown player"),
            "position": _name(event.get("position")),
            "recipient": recipient,
            "event_type": event_type,
            "outcome": outcome,
            "x": round(start_x, 4),
            "y": round(start_y, 4),
            "end_x": round(end_x, 4),
            "end_y": round(end_y, 4),
            "body_part": body_part,
            "play_pattern": _name(event.get("play_pattern"), "Unknown"),
            "under_pressure": bool(event.get("under_pressure", False)),
            "first_time": bool(detail.get("first_time", False)) if event_type == "Shot" else False,
            "assisted": bool(detail.get("key_pass_id")) if event_type == "Shot" else False,
            "key_pass": key_pass,
            "shot_goal": shot_goal,
        })
    return pd.DataFrame(rows)
