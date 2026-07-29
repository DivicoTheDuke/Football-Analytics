from football_analytics.features import add_spatial_features
from football_analytics.metrics import player_summary
from football_analytics.scouting import player_similarity, percentile_profile
from football_analytics.xg import train_xg, predict_xg
from football_analytics.xt import train_xt, apply_xt


def test_similarity_and_percentiles(demo_data):
    _, events, _ = demo_data
    artifact = train_xg(events)
    shots = predict_xg(artifact.model, events)
    enriched = apply_xt(train_xt(events), add_spatial_features(events))
    players = player_summary(enriched, shots)
    player = players.iloc[0]["player"]
    similar = player_similarity(players, player, 4)
    profile = percentile_profile(players, player)
    assert len(similar) == 4
    assert profile["percentile"].between(0, 100).all()
