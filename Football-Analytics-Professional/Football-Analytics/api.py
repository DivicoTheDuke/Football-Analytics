from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from football_analytics.metrics import player_summary, team_match_summary
from football_analytics.forecasting import attacking_side_profile, forecast_fixture, scorer_probabilities, season_projection, team_style_profiles
from football_analytics.scouting import player_similarity
from football_analytics.service import datasets, enriched_events, models

app = FastAPI(
    title="Football Analytics API",
    version="1.0.0",
    description="Portfolio API for transparent football analytics and probabilistic forecasts. Demo data is synthetic unless historical mode is configured.",
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


@app.get("/forecasts/season")
def forecast_season_endpoint():
    match_frame, _, _ = datasets(); _, shots = enriched_events()
    return season_projection(match_frame, shots).round(4).to_dict(orient="records")


@app.get("/forecasts/fixture")
def forecast_fixture_endpoint(home_team: str, away_team: str):
    match_frame, events, _ = datasets(); enriched, shots = enriched_events()
    teams = set(match_frame["home_team"]) | set(match_frame["away_team"])
    if home_team not in teams or away_team not in teams or home_team == away_team:
        raise HTTPException(400, "Choose two different known teams")
    forecast = forecast_fixture(match_frame, shots, home_team, away_team)
    return {**forecast.__dict__, "home_scorers": scorer_probabilities(enriched, shots, home_team, forecast.home_xg).head(10).round(4).to_dict(orient="records"), "away_scorers": scorer_probabilities(enriched, shots, away_team, forecast.away_xg).head(10).round(4).to_dict(orient="records")}


@app.get("/teams/{team}/identity")
def team_identity(team: str):
    _, events, _ = datasets()
    identity = team_style_profiles(events).merge(attacking_side_profile(events), on="team", how="left")
    row = identity.loc[identity["team"] == team]
    if row.empty:
        raise HTTPException(404, "Unknown team")
    return row.iloc[0].to_dict()
