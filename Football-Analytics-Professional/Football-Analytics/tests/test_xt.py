from football_analytics.xt import train_xt, apply_xt


def test_xt_training_and_application(demo_data):
    _, events, _ = demo_data
    model = train_xt(events, x_bins=12, y_bins=8)
    enriched = apply_xt(model, events)
    assert model.grid.shape == (8, 12)
    assert model.grid.min() >= 0
    assert "xt_added" in enriched
