from football_analytics.networks import passing_network


def test_passing_network(demo_data):
    _, events, _ = demo_data
    team = events.iloc[0]["team"]
    nodes, edges, graph = passing_network(events, team, minimum_connections=1)
    assert not nodes.empty
    assert graph.number_of_nodes() == len(nodes)
