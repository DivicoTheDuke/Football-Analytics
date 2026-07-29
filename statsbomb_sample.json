import pandas as pd
import pytest

from football_analytics.demo import generate_demo


@pytest.fixture(scope="session")
def demo_data(tmp_path_factory):
    directory = tmp_path_factory.mktemp("demo")
    matches, events, lineups = generate_demo(directory, matches=8, seed=7)
    return matches, events, lineups
