from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from football_analytics.metrics import player_summary, team_match_summary
from football_analytics.scouting import player_similarity
from football_analytics.service import datasets, enriched_events, models

app = FastAPI(
    title="Football Analytics API",
    version="1.0.0",
    description="Portfolio API for match, player and expected-goals analytics using synthetic data.",
)


class ShotRequest(BaseModel):
    x: float = Field(ge=0, le=105)
    y: float = Field(ge=0, le=68)
    body_part: str = "Right Foot"
    play_pattern: str = "Open Play"
    under_pressure: bool = False
    first_time: bool = False
    assisted: bool = True


@app.get("/health")
def health():
    matches, events, _ = datasets()
    return {"status": "ok", "matches": len(matches), "events": len(events)}


@app.get("/matches")
def matches():
    match_frame, _, _ = datasets()
    return match_frame.sort_values("match_date", ascending=False).to_dict(orient="records")


@app.get("/matches/{match_id}/summary")
def match_summary(match_id: str):
    events, shots = enriched_events()
    match_events = events.loc[events["match_id"] == match_id]
    match_shots = shots.loc[shots["match_id"] == match_id]
    if match_events.empty:
        raise HTTPException(404, "Unknown match")
    return team_match_summary(match_events, match_shots).round(4).to_dict(orient="records")


@app.get("/players/{player}/profile")
def player_profile(player: str):
    events, shots = enriched_events()
    players = player_summary(events, shots)
    row = players.loc[players["player"] == player]
    if row.empty:
        raise HTTPException(404, "Unknown player")
    return row.iloc[0].to_dict()


@app.get("/players/{player}/similar")
def similar_players(player: str, limit: int = 6):
    events, shots = enriched_events()
    players = player_summary(events, shots)
    try:
        return player_similarity(players, player, limit).round(4).to_dict(orient="records")
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/models/xg/metadata")
def xg_metadata():
    artifact, _ = models()
    return artifact.metrics


@app.post("/models/xg/predict")
def xg_predict(request: ShotRequest):
    artifact, _ = models()
    row = pd.DataFrame([{
        "event_id": "api", "match_id": "api", "team": "api", "player": "api",
        "event_type": "Shot", "outcome": "Unknown", "x": request.x, "y": request.y,
        "end_x": 105.0, "end_y": 34.0, "body_part": request.body_part,
        "play_pattern": request.play_pattern, "under_pressure": request.under_pressure,
        "first_time": request.first_time, "assisted": request.assisted, "shot_goal": False,
    }])
    from football_analytics.xg import predict_xg
    probability = float(predict_xg(artifact.model, row).iloc[0]["xg"])
    return {"xg": probability}
