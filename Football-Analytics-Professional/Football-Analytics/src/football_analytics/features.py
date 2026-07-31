from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_Y = PITCH_WIDTH / 2
GOAL_WIDTH = 7.32


def add_spatial_features(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()

    dx = PITCH_LENGTH - result["x"].astype(float)
    dy = np.abs(GOAL_Y - result["y"].astype(float))
    result["distance_to_goal"] = np.sqrt(dx ** 2 + dy ** 2)

    left_post = GOAL_Y - GOAL_WIDTH / 2
    right_post = GOAL_Y + GOAL_WIDTH / 2
    angle_left = np.arctan2(left_post - result["y"], dx.clip(lower=0.01))
    angle_right = np.arctan2(right_post - result["y"], dx.clip(lower=0.01))
    result["shot_angle"] = np.abs(angle_right - angle_left)

    result["start_zone_x"] = pd.cut(result["x"], bins=[-0.01, 35, 70, 105], labels=["defensive", "middle", "attacking"])
    result["end_zone_x"] = pd.cut(result["end_x"], bins=[-0.01, 35, 70, 105], labels=["defensive", "middle", "attacking"])

    start_goal_distance = np.sqrt((PITCH_LENGTH - result["x"]) ** 2 + (GOAL_Y - result["y"]) ** 2)
    end_goal_distance = np.sqrt((PITCH_LENGTH - result["end_x"]) ** 2 + (GOAL_Y - result["end_y"]) ** 2)
    result["distance_progressed"] = start_goal_distance - end_goal_distance
    result["progressive"] = (
        result["event_type"].isin(["Pass", "Carry"])
        & (result["outcome"] == "Complete")
        & ((result["distance_progressed"] >= 10) | ((result["x"] < 70) & (result["end_x"] >= 70)))
    )
    result["final_third_entry"] = (
        result["event_type"].isin(["Pass", "Carry"])
        & (result["outcome"] == "Complete")
        & (result["x"] < 70)
        & (result["end_x"] >= 70)
    )
    result["box_entry"] = (
        result["event_type"].isin(["Pass", "Carry"])
        & (result["outcome"] == "Complete")
        & ~((result["x"] >= 88.5) & result["y"].between(13.84, 54.16))
        & ((result["end_x"] >= 88.5) & result["end_y"].between(13.84, 54.16))
    )
    return result


def shot_frame(events: pd.DataFrame) -> pd.DataFrame:
    shots = add_spatial_features(events.loc[events["event_type"] == "Shot"].copy())
    shots["body_part"] = shots["body_part"].fillna("Unknown")
    shots["play_pattern"] = shots["play_pattern"].fillna("Open Play")
    return shots


def add_possession_context(events: pd.DataFrame) -> pd.DataFrame:
    result = events.sort_values(["match_id", "period", "timestamp_seconds", "event_id"]).copy()
    grouped = result.groupby(["match_id", "possession_id"], sort=False)
    result["possession_event_index"] = grouped.cumcount() + 1
    result["possession_start_x"] = grouped["x"].transform("first")
    result["possession_duration"] = grouped["timestamp_seconds"].transform("max") - grouped["timestamp_seconds"].transform("min")
    result["possession_length"] = grouped["event_id"].transform("size")
    return result
