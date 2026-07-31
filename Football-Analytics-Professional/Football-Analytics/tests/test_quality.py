from football_analytics.quality import validate_events


def test_demo_data_passes_quality_gate(demo_data):
    _, events, _ = demo_data
    report = validate_events(events)
    assert report.passed
    assert report.matches == 8
