from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_spatial_features

DEFENSIVE_ACTIONS = {"Duel", "Interception", "Recovery", "Pressure"}


def attach_expected_assists(events: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    shot_xg = shots.set_index("event_id")["xg"].to_dict()
    result["xa"] = 0.0
    ordered = result.sort_values(["match_id", "possession_id", "timestamp_seconds", "event_id"])
    for (_, _), possession in ordered.groupby(["match_id", "possession_id"], sort=False):
        shot_rows = possession.loc[possession["event_type"] == "Shot"]
        if shot_rows.empty:
            continue
        first_shot = shot_rows.iloc[0]
        key_passes = possession.loc[(possession["event_type"] == "Pass") & possession["key_pass"]]
        if not key_passes.empty:
            key_id = key_passes.iloc[-1]["event_id"]
            result.loc[result["event_id"] == key_id, "xa"] = shot_xg.get(first_shot["event_id"], 0.0)
    return result


def team_match_summary(events: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    frame = add_spatial_features(events)
    teams = sorted(frame["team"].dropna().unique())
    rows = []
    total_completed_passes = frame.loc[
        (frame["event_type"] == "Pass") & (frame["outcome"] == "Complete")
    ].groupby("team").size()

    for team in teams:
        own = frame.loc[frame["team"] == team]
        opp = frame.loc[frame["team"] != team]
        team_shots = shots.loc[shots["team"] == team]

        own_final_third_passes = own.loc[
            (own["event_type"] == "Pass") & (own["outcome"] == "Complete") & (own["x"] >= 70)
        ].shape[0]
        opp_final_third_passes = opp.loc[
            (opp["event_type"] == "Pass") & (opp["outcome"] == "Complete") & (opp["x"] >= 70)
        ].shape[0]
        field_tilt = 100 * own_final_third_passes / max(1, own_final_third_passes + opp_final_third_passes)

        opponent_passes_low = opp.loc[
            (opp["event_type"] == "Pass") & (opp["outcome"] == "Complete") & (opp["x"] <= 63)
        ].shape[0]
        defensive_actions_high = own.loc[
            own["event_type"].isin(DEFENSIVE_ACTIONS) & (own["x"] >= 42)
        ].shape[0]
        ppda = opponent_passes_low / max(1, defensive_actions_high)

        completed = int(total_completed_passes.get(team, 0))
        all_completed = int(total_completed_passes.sum())
        possession_proxy = 100 * completed / max(1, all_completed)

        rows.append({
            "team": team,
            "goals": int(team_shots["shot_goal"].sum()),
            "shots": int(len(team_shots)),
            "xg": float(team_shots["xg"].sum()),
            "xg_per_shot": float(team_shots["xg"].mean()) if len(team_shots) else 0.0,
            "possession_proxy": possession_proxy,
            "field_tilt": field_tilt,
            "ppda": ppda,
            "progressive_actions": int(own["progressive"].sum()),
            "final_third_entries": int(own["final_third_entry"].sum()),
            "box_entries": int(own["box_entry"].sum()),
            "pass_completion": 100 * (
                ((own["event_type"] == "Pass") & (own["outcome"] == "Complete")).sum()
                / max(1, (own["event_type"] == "Pass").sum())
            ),
        })
    return pd.DataFrame(rows)


def player_summary(events: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    frame = add_spatial_features(events)
    frame = attach_expected_assists(frame, shots)
    shot_metrics = shots.groupby(["team", "player"], as_index=False).agg(
        shots=("event_id", "size"), goals=("shot_goal", "sum"), xg=("xg", "sum")
    )

    rows = []
    for (team, player), player_events in frame.groupby(["team", "player"], sort=False):
        passes = player_events.loc[player_events["event_type"] == "Pass"]
        completed = passes.loc[passes["outcome"] == "Complete"]
        rows.append({
            "team": team,
            "player": player,
            "events": len(player_events),
            "passes": len(passes),
            "pass_completion": 100 * len(completed) / max(1, len(passes)),
            "progressive_passes": int((passes["progressive"] & (passes["outcome"] == "Complete")).sum()),
            "progressive_carries": int(((player_events["event_type"] == "Carry") & player_events["progressive"]).sum()),
            "final_third_entries": int(player_events["final_third_entry"].sum()),
            "box_entries": int(player_events["box_entry"].sum()),
            "key_passes": int(player_events["key_pass"].sum()),
            "xa": float(player_events["xa"].sum()),
            "defensive_actions": int(player_events["event_type"].isin(DEFENSIVE_ACTIONS).sum()),
            "xt_added": float(player_events.get("xt_added", pd.Series(0.0, index=player_events.index)).sum()),
        })

    result = pd.DataFrame(rows).merge(shot_metrics, on=["team", "player"], how="left")
    for col in ["shots", "goals", "xg"]:
        result[col] = result[col].fillna(0)
    return result


def xg_momentum(shots: pd.DataFrame) -> pd.DataFrame:
    frame = shots.sort_values(["period", "timestamp_seconds", "event_id"]).copy()
    frame["match_minute"] = frame["timestamp_seconds"] / 60
    frame["cumulative_xg"] = frame.groupby("team")["xg"].cumsum()
    return frame
