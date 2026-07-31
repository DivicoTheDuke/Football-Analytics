import json
from pathlib import Path

from football_analytics.providers import normalise_statsbomb_events
from football_analytics.quality import validate_events


def test_statsbomb_adapter_maps_to_canonical_schema():
    fixture = Path(__file__).parent / "fixtures" / "statsbomb_sample.json"
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    events = normalise_statsbomb_events(
        raw,
        match_id="SB-1",
        home_team="North London Red",
        away_team="Manchester Blue",
    )
    assert len(events) == 2
    assert events.iloc[0]["recipient"] == "NLR Striker"
    assert bool(events.iloc[0]["key_pass"])
    assert bool(events.iloc[1]["shot_goal"])
    assert events["x"].between(0, 105).all()
    assert validate_events(events).passed
