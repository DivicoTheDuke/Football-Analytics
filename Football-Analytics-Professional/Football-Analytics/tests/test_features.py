from football_analytics.features import add_spatial_features, shot_frame


def test_spatial_features_are_valid(demo_data):
    _, events, _ = demo_data
    result = add_spatial_features(events)
    assert result["distance_to_goal"].ge(0).all()
    assert result["shot_angle"].ge(0).all()
    assert set(result["progressive"].unique()).issubset({True, False})


def test_shot_frame_only_contains_shots(demo_data):
    _, events, _ = demo_data
    shots = shot_frame(events)
    assert not shots.empty
    assert shots["event_type"].eq("Shot").all()
