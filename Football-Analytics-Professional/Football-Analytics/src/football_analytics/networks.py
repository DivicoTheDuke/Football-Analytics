from __future__ import annotations

import networkx as nx
import pandas as pd


def passing_network(events: pd.DataFrame, team: str, minimum_connections: int = 2):
    passes = events.loc[
        (events["team"] == team)
        & (events["event_type"] == "Pass")
        & (events["outcome"] == "Complete")
        & events["recipient"].notna()
    ].copy()

    edges = (
        passes.groupby(["player", "recipient"], as_index=False)
        .agg(pass_count=("event_id", "size"), avg_start_x=("x", "mean"), avg_start_y=("y", "mean"))
    )
    edges = edges.loc[edges["pass_count"] >= minimum_connections]

    nodes = passes.groupby("player", as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"), touches=("event_id", "size")
    )

    graph = nx.DiGraph()
    for row in nodes.itertuples(index=False):
        graph.add_node(row.player, x=row.x, y=row.y, touches=row.touches)
    for row in edges.itertuples(index=False):
        graph.add_edge(row.player, row.recipient, weight=row.pass_count)

    centrality = nx.pagerank(graph, weight="weight") if graph.number_of_nodes() else {}
    nodes["network_centrality"] = nodes["player"].map(centrality).fillna(0.0)
    return nodes, edges, graph
