from football_analytics.features import add_spatial_features
from football_analytics.metrics import team_match_summary, player_summary
from football_analytics.xg import train_xg, predict_xg
from football_analytics.xt import train_xt, apply_xt


def test_team_and_player_metrics(demo_data):
    _, events, _ = demo_data
    xg = train_xg(events)
    shots = predict_xg(xg.model, events)
    xt = train_xt(events)
    enriched = apply_xt(xt, add_spatial_features(events))
    match_id = events.iloc[0]["match_id"]
    match_events = enriched.loc[enriched["match_id"] == match_id]
    match_shots = shots.loc[shots["match_id"] == match_id]
    teams = team_match_summary(match_events, match_shots)
    players = player_summary(enriched, shots)
    assert len(teams) == 2
    assert not players.empty
    assert teams["xg"].ge(0).all()
