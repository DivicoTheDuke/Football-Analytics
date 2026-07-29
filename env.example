from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from football_analytics.metrics import player_summary, team_match_summary, xg_momentum
from football_analytics.networks import passing_network
from football_analytics.scouting import percentile_profile, player_similarity, cluster_players
from football_analytics.service import datasets, enriched_events, models
from football_analytics.xt import xt_grid_frame

st.set_page_config(page_title="Football Analytics Platform", layout="wide")
st.title("Football Analytics Platform")
st.caption("Decision-support workflows for analysis, coaching and recruitment · synthetic portfolio data")

matches, raw_events, lineups = datasets()
events, shots = enriched_events()
xg_artifact, xt_model = models()

page = st.sidebar.radio(
    "Workflow",
    ["Match Centre", "Shot & xG Lab", "Possession & Threat", "Passing Network", "Player Recruitment", "Model Governance"],
)

match_labels = {
    row.match_id: f"{row.match_date.date()} · {row.home_team} {row.home_goals}-{row.away_goals} {row.away_team}"
    for row in matches.itertuples(index=False)
}
selected_match = st.sidebar.selectbox("Match", list(match_labels), format_func=lambda value: match_labels[value])
match_events = events.loc[events["match_id"] == selected_match].copy()
match_shots = shots.loc[shots["match_id"] == selected_match].copy()
teams = list(match_events["team"].dropna().unique())

if page == "Match Centre":
    st.header("Executive Match Centre")
    summary = team_match_summary(match_events, match_shots)
    c1, c2 = st.columns(2)
    for column, row in zip([c1, c2], summary.itertuples(index=False)):
        with column:
            st.subheader(row.team)
            a, b, c = st.columns(3)
            a.metric("Goals", int(row.goals))
            b.metric("xG", f"{row.xg:.2f}")
            c.metric("Shots", int(row.shots))
            d, e, f = st.columns(3)
            d.metric("Field tilt", f"{row.field_tilt:.1f}%")
            e.metric("PPDA", f"{row.ppda:.1f}")
            f.metric("Box entries", int(row.box_entries))

    momentum = xg_momentum(match_shots)
    fig = px.line(momentum, x="match_minute", y="cumulative_xg", color="team", markers=True,
                  title="Cumulative expected goals")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analyst interpretation")
    st.write(
        "Use the metrics as a structured starting point, then validate the causes with video. "
        "Field tilt describes territorial control, PPDA approximates pressing intensity and xG separates shot volume from shot quality."
    )
    st.dataframe(summary.round(2), use_container_width=True)

elif page == "Shot & xG Lab":
    st.header("Shot and Expected Goals Lab")
    selected_team = st.selectbox("Team", ["All"] + teams)
    view = match_shots if selected_team == "All" else match_shots.loc[match_shots["team"] == selected_team]
    fig = px.scatter(
        view, x="x", y="y", color="team", size="xg", symbol="shot_goal",
        hover_data=["player", "xg", "distance_to_goal", "shot_angle", "body_part", "play_pattern", "outcome"],
        title="Shot map, marker size represents xG"
    )
    fig.update_xaxes(range=[0, 105])
    fig.update_yaxes(range=[0, 68])
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Shot quality by player")
        player_shots = view.groupby(["team", "player"], as_index=False).agg(
            shots=("event_id", "size"), goals=("shot_goal", "sum"), xg=("xg", "sum"), average_xg=("xg", "mean")
        )
        st.dataframe(player_shots.sort_values("xg", ascending=False).round(3), use_container_width=True)
    with right:
        st.subheader("Shot quality distribution")
        st.plotly_chart(px.histogram(view, x="xg", color="team", nbins=15), use_container_width=True)

elif page == "Possession & Threat":
    st.header("Possession and Expected Threat")
    grid = xt_grid_frame(xt_model)
    heatmap = go.Figure(data=go.Heatmap(
        z=xt_model.grid, x=np.linspace(0, 105, xt_model.x_bins), y=np.linspace(0, 68, xt_model.y_bins),
        colorbar_title="xT"
    ))
    heatmap.update_layout(title="Expected threat surface", xaxis_title="Pitch length", yaxis_title="Pitch width")
    st.plotly_chart(heatmap, use_container_width=True)

    team = st.selectbox("Team for action analysis", teams, key="xt_team")
    team_actions = match_events.loc[(match_events["team"] == team) & (match_events["xt_added"] != 0)].copy()
    st.subheader("Highest-value progression actions")
    columns = ["minute", "player", "event_type", "x", "y", "end_x", "end_y", "xt_added"]
    st.dataframe(team_actions.sort_values("xt_added", ascending=False)[columns].head(25).round(4), use_container_width=True)

    player_threat = team_actions.groupby("player", as_index=False)["xt_added"].sum().sort_values("xt_added", ascending=False)
    st.plotly_chart(px.bar(player_threat.head(12), x="player", y="xt_added", title="xT added by player"), use_container_width=True)

elif page == "Passing Network":
    st.header("Passing Network")
    team = st.selectbox("Team", teams, key="network_team")
    nodes, edges, graph = passing_network(match_events, team)
    fig = go.Figure()
    for edge in edges.itertuples(index=False):
        source = nodes.loc[nodes["player"] == edge.player]
        target = nodes.loc[nodes["player"] == edge.recipient]
        if source.empty or target.empty:
            continue
        fig.add_trace(go.Scatter(
            x=[source.iloc[0]["x"], target.iloc[0]["x"]], y=[source.iloc[0]["y"], target.iloc[0]["y"]],
            mode="lines", line={"width": max(1, edge.pass_count / 2)}, hoverinfo="text",
            text=f"{edge.player} → {edge.recipient}: {edge.pass_count}", showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=nodes["x"], y=nodes["y"], mode="markers+text", text=nodes["player"], textposition="top center",
        marker={"size": 14 + nodes["touches"] / max(1, nodes["touches"].max()) * 28},
        customdata=nodes[["touches", "network_centrality"]],
        hovertemplate="%{text}<br>Touches: %{customdata[0]}<br>Centrality: %{customdata[1]:.3f}<extra></extra>"
    ))
    fig.update_layout(title=f"{team} passing network", xaxis={"range": [0, 105]}, yaxis={"range": [0, 68]}, height=650)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(nodes.sort_values("network_centrality", ascending=False).round(3), use_container_width=True)

elif page == "Player Recruitment":
    st.header("Player Recruitment and Similarity")
    players = player_summary(events, shots)
    players = cluster_players(players)
    selected_player = st.selectbox("Reference player", sorted(players["player"].unique()))

    profile = percentile_profile(players, selected_player)
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=profile["percentile"], theta=profile["metric"], fill="toself", name=selected_player))
    radar.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 100]}}, showlegend=False,
                        title="Percentile profile across the demo league")
    st.plotly_chart(radar, use_container_width=True)

    st.subheader("Most similar players")
    similar = player_similarity(players, selected_player, 8)
    st.dataframe(similar.round(3), use_container_width=True)
    st.caption("Similarity uses standardised event metrics and cosine distance. Role, age, competition strength and physical data would be required for real recruitment decisions.")

elif page == "Model Governance":
    st.header("Model Governance")
    metrics = xg_artifact.metrics
    columns = st.columns(4)
    columns[0].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    columns[1].metric("Average precision", f"{metrics['average_precision']:.3f}")
    columns[2].metric("Brier score", f"{metrics['brier_score']:.3f}")
    columns[3].metric("Log loss", f"{metrics['log_loss']:.3f}")

    calibration = xg_artifact.calibration
    fig = px.line(calibration, x="predicted_probability", y="observed_goal_rate", markers=True,
                  title="Calibration curve")
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Governance notes")
    st.markdown(
        """
- Match IDs are used to separate training and test data, reducing leakage between events from the same match.
- Metrics cover ranking quality, probability accuracy and calibration.
- Synthetic data is useful for software testing but not for sporting validation.
- A real deployment needs temporal validation, league-specific calibration, drift monitoring and formal analyst sign-off.
- Model outputs should support, not replace, video analysis and expert judgement.
"""
    )
