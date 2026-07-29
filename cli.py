from football_analytics.xg import train_xg, predict_xg


def test_xg_training_and_prediction(demo_data):
    _, events, _ = demo_data
    artifact = train_xg(events, random_state=7, test_size=0.25)
    predictions = predict_xg(artifact.model, events)
    assert predictions["xg"].between(0, 1).all()
    assert 0 <= artifact.metrics["roc_auc"] <= 1
    assert artifact.metrics["test_matches"] >= 1
