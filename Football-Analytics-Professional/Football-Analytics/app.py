from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from football_analytics.config import load_settings
from football_analytics.forecasting import (
    attacking_side_profile,
    forecast_fixture,
    scorer_probabilities,
    season_projection,
    team_style_profiles,
)
from football_analytics.match_forecasting import (
    forecast_from_match_history,
    load_cached_match_history,
    season_projection_from_match_history,
)
from football_analytics.match_models import (
    forecast_with_match_models,
    load_match_model_bundle,
    ml_season_projection,
)
from football_analytics.metrics import (
    player_summary,
    team_match_summary,
    xg_momentum,
)
from football_analytics.networks import passing_network
from football_analytics.scouting import (
    cluster_players,
    percentile_profile,
    player_similarity,
)
from football_analytics.service import (
    datasets,
    enriched_events,
    models,
)
from football_analytics.training import retrain_and_recalculate
from football_analytics.xt import xt_grid_frame


PAGE_LABELS = {
    "platform_overview": "⚽ Platform Overview",
    "season_forecast": "🏆 Season Forecast",
    "fixture_forecast": "🏟️ Fixture Forecast",
    "team_identity": "👕 Team Identity",
    "match_centre": "⚽ Match Centre",
    "shot_xg_lab": "🥅 Shot & xG Lab",
    "possession_threat": "⚽ Possession & Threat",
    "passing_network": "⚽ Passing Network",
    "player_recruitment": "👕 Player Recruitment",
    "model_governance": "🏆 Model Governance",
}


def resolve_page(selected_label: str) -> str:
    """Return the stable internal page identifier for a visible label."""
    for page_key, label in PAGE_LABELS.items():
        if label == selected_label:
            return page_key

    raise ValueError(f"Unknown page label: {selected_label}")


st.set_page_config(
    page_title="Football Analytics Platform",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Analytics Platform")

settings = load_settings()

mode_label = (
    "SYNTHETIC DEMO DATA — NOT A REAL SPORTING FORECAST"
    if settings.is_demo
    else "HISTORICAL PROVIDER DATA — VERIFY LICENCE AND DATA LINEAGE"
)

st.caption(
    "Decision-support workflows for analysis, coaching and recruitment"
    f" · {mode_label}"
)

st.warning(
    "Forecast probabilities are model outputs with uncertainty. "
    "They support analyst review and must not be presented as facts."
)

matches, raw_events, lineups = datasets()
events, shots = enriched_events()
xg_artifact, xt_model = models()

selected_label = st.sidebar.radio(
    "⚽ Football Workflows",
    list(PAGE_LABELS.values()),
)

page = resolve_page(selected_label)

match_labels = {
    row.match_id: (
        f"{row.match_date.date()} · "
        f"{row.home_team} {row.home_goals}-{row.away_goals} "
        f"{row.away_team}"
    )
    for row in matches.itertuples(index=False)
}

# selected_match = st.sidebar.selectbox(
    # "🏟️ Match",
    # list(match_labels),
    # format_func=lambda value: match_labels[value],
# )
selected_match = list(match_labels)[0]

match_events = events.loc[
    events["match_id"] == selected_match
].copy()

match_shots = shots.loc[
    shots["match_id"] == selected_match
].copy()

teams = list(
    match_events["team"]
    .dropna()
    .unique()
)


if page == "platform_overview":
    st.header("⚽ Data and Capability Overview")

    provider_cache = settings.footystats_matches_cache
    model_exists = settings.match_model_bundle.exists()

    cached_matches = (
        load_cached_match_history(provider_cache)
        if provider_cache.exists()
        else None
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "⚽ Cached matches",
        (
            f"{len(cached_matches):,}"
            if cached_matches is not None
            else "0"
        ),
    )

    c2.metric(
        "🏆 Cached seasons",
        (
            str(cached_matches["season"].nunique())
            if cached_matches is not None
            else "0"
        ),
    )

    c3.metric(
        "🥅 Match forecast model",
        "Ready" if model_exists else "Not trained",
    )

    c4.metric(
        "⚽ Synthetic demo events",
        f"{len(events):,}",
    )

    st.subheader("🏟️ Current Platform Capabilities")

    capability_rows = [
        {
            "Capability": "Match outcome probabilities",
            "Status": "⚽ Available",
            "Data basis": "Cached FootyStats match results",
            "Method": (
                "scikit-learn outcome classifier "
                "and goal models"
            ),
        },
        {
            "Capability": "Expected home and away goals",
            "Status": "🥅 Available",
            "Data basis": "Pre-match rolling team form",
            "Method": "Poisson regression",
        },
        {
            "Capability": (
                "Most likely score, BTTS and Over 2.5"
            ),
            "Status": "🥅 Available",
            "Data basis": "ML expected-goal rates",
            "Method": "Normalized Poisson score matrix",
        },
        {
            "Capability": "Next-season table simulation",
            "Status": "🏆 Available with limitations",
            "Data basis": "Teams in latest cached season",
            "Method": (
                "Double round-robin expected-points simulation"
            ),
        },
        {
            "Capability": (
                "Shot maps, xG, xT and passing networks"
            ),
            "Status": "⚽ Synthetic demo only",
            "Data basis": "Synthetic event data",
            "Method": (
                "Event analytics workflow demonstration"
            ),
        },
        {
            "Capability": (
                "Real attacking-side prediction"
            ),
            "Status": "🏟️ Additional data required",
            "Data basis": (
                "Requires licensed event coordinates"
            ),
            "Method": (
                "Can be implemented when suitable "
                "provider data exists"
            ),
        },
        {
            "Capability": "Real formation prediction",
            "Status": "👕 Additional data required",
            "Data basis": (
                "Requires line-ups, positions or tracking data"
            ),
            "Method": (
                "Can be implemented when suitable "
                "provider data exists"
            ),
        },
        {
            "Capability": "Real player goalscorer forecast",
            "Status": "🥅 Additional data required",
            "Data basis": (
                "Requires player shots, minutes, expected "
                "line-ups and availability"
            ),
            "Method": (
                "Can be implemented when player-level "
                "data exists"
            ),
        },
        {
            "Capability": (
                "Pressing height and build-up structure"
            ),
            "Status": "⚽ Additional data required",
            "Data basis": (
                "Requires event sequences or tracking data"
            ),
            "Method": (
                "Can be implemented when detailed "
                "tactical data exists"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(capability_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "⚽ The cached FootyStats file is never refreshed from "
        "this dashboard. Provider access happens only through "
        "the explicit import command. Training and recalculation "
        "use local files only."
    )

    if cached_matches is not None:
        seasons = sorted(
            cached_matches["season"]
            .dropna()
            .astype(str)
            .unique()
        )

        st.write(
            "🏆 Loaded provider seasons: "
            f"**{', '.join(seasons)}**"
        )

        if len(seasons) < 3:
            st.warning(
                "Only a small number of seasons is currently "
                "available. The workflow is technically valid, "
                "but the result should be presented as a model "
                "demonstration rather than a reliable forecast "
                "for the current Premier League."
            )


elif page == "season_forecast":
    st.header("🏆 Next-Season Projection")

    provider_cache = settings.footystats_matches_cache

    if provider_cache.exists():
        history = load_cached_match_history(provider_cache)

        seasons = sorted(
            history["season"]
            .dropna()
            .astype(str)
            .unique()
        )

        st.success(
            f"⚽ Local FootyStats cache: {len(history):,} matches "
            f"· seasons {', '.join(seasons)}. "
            "No provider request is made by this page."
        )

        if st.button(
            "⚽ Retrain models and recalculate locally",
            type="primary",
        ):
            with st.spinner(
                "Building leakage-safe football features, "
                "training models and regenerating forecasts..."
            ):
                bundle, paths = retrain_and_recalculate(
                    history,
                    model_dir=settings.model_dir,
                    report_dir=settings.report_dir,
                    random_state=settings.random_state,
                    test_fraction=settings.test_size,
                )

                st.session_state["match_model_bundle"] = bundle

                st.session_state["ml_projection"] = (
                    ml_season_projection(
                        bundle,
                        history,
                    )
                )

            st.success(
                "⚽ Training completed locally. "
                "FootyStats was not contacted."
            )

            st.json(
                {
                    name: str(path)
                    for name, path in paths.items()
                }
            )

        bundle = st.session_state.get(
            "match_model_bundle"
        )

        if (
            bundle is None
            and settings.match_model_bundle.exists()
        ):
            bundle = load_match_model_bundle(
                settings.match_model_bundle
            )

        if bundle is not None:
            projection = st.session_state.get(
                "ml_projection",
                ml_season_projection(
                    bundle,
                    history,
                ),
            )

            st.caption(
                "⚽ ML mode: pre-match rolling features, "
                "temporal holdout evaluation, logistic "
                "match-outcome classification and Poisson "
                "goal regression."
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "⚽ Outcome log loss",
                f"{bundle.metrics.outcome_log_loss:.3f}",
            )

            m2.metric(
                "🏆 Outcome accuracy",
                f"{bundle.metrics.outcome_accuracy:.1%}",
            )

            m3.metric(
                "🥅 Home goals MAE",
                f"{bundle.metrics.home_goals_mae:.3f}",
            )

            m4.metric(
                "🥅 Away goals MAE",
                f"{bundle.metrics.away_goals_mae:.3f}",
            )

            st.warning(
                "These metrics come from a chronological "
                "holdout. With only one old season, they "
                "validate the pipeline rather than a reliable "
                "forecast for the current Premier League."
            )

        else:
            projection = (
                season_projection_from_match_history(
                    history
                )
            )

            st.warning(
                "No trained ML bundle exists yet. The table "
                "below uses the statistical fallback. Press "
                "the training button to create the local "
                "scikit-learn models."
            )

    else:
        st.warning(
            "No cached FootyStats file was found. This table "
            "uses synthetic demo events and is not a real "
            "Premier League prediction."
        )

        projection = season_projection(
            matches,
            shots,
        )

    st.dataframe(
        projection.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.plotly_chart(
        px.bar(
            projection.head(10),
            x="team",
            y="points",
            title="🏆 Projected Expected Points",
        ),
        use_container_width=True,
    )


elif page == "fixture_forecast":
    st.header("🏟️ Fixture Forecast")

    provider_cache = settings.footystats_matches_cache

    if provider_cache.exists():
        history = load_cached_match_history(
            provider_cache
        )

        latest_season = (
            history.sort_values("match_date")["season"]
            .dropna()
            .astype(str)
            .iloc[-1]
        )

        latest = history.loc[
            history["season"]
            .astype(str)
            .eq(latest_season)
        ]

        all_teams = sorted(
            set(latest["home_team"])
            | set(latest["away_team"])
        )

        c1, c2 = st.columns(2)

        home_team = c1.selectbox(
            "🏟️ Home team",
            all_teams,
        )

        away_team = c2.selectbox(
            "⚽ Away team",
            [
                team
                for team in all_teams
                if team != home_team
            ],
        )

        bundle = st.session_state.get(
            "match_model_bundle"
        )

        if (
            bundle is None
            and settings.match_model_bundle.exists()
        ):
            bundle = load_match_model_bundle(
                settings.match_model_bundle
            )

        if bundle is not None:
            prediction = forecast_with_match_models(
                bundle,
                history,
                home_team,
                away_team,
            )

            home_rate = (
                prediction.home_expected_goals
            )

            away_rate = (
                prediction.away_expected_goals
            )

            st.success(
                "⚽ Using locally trained scikit-learn "
                "match models. No provider request is made."
            )

            st.caption(
                prediction.model_source
            )

        else:
            prediction = forecast_from_match_history(
                history,
                home_team,
                away_team,
            )

            home_rate = prediction.home_rate
            away_rate = prediction.away_rate

            st.warning(
                "No ML bundle was found. The statistical "
                "match-history fallback is being used."
            )

        provider_mode = True

    else:
        all_teams = sorted(
            set(matches["home_team"])
            | set(matches["away_team"])
        )

        c1, c2 = st.columns(2)

        home_team = c1.selectbox(
            "🏟️ Home team",
            all_teams,
        )

        away_team = c2.selectbox(
            "⚽ Away team",
            [
                team
                for team in all_teams
                if team != home_team
            ],
        )

        prediction = forecast_fixture(
            matches,
            shots,
            home_team,
            away_team,
        )

        home_rate = prediction.home_xg
        away_rate = prediction.away_xg
        provider_mode = False

        st.warning(
            "No FootyStats cache was found. This fixture "
            "uses synthetic demo performances."
        )

    a, b, c, d = st.columns(4)

    a.metric(
        "🏟️ Home win",
        f"{prediction.home_win:.1%}",
    )

    b.metric(
        "⚽ Draw",
        f"{prediction.draw:.1%}",
    )

    c.metric(
        "⚽ Away win",
        f"{prediction.away_win:.1%}",
    )

    d.metric(
        "🥅 Most likely score",
        prediction.most_likely_score,
    )

    st.write(
        "🥅 Expected scoring rates: "
        f"**{home_team} {home_rate:.2f} – "
        f"{away_rate:.2f} {away_team}**"
    )

    e, f = st.columns(2)

    e.metric(
        "🥅 Both teams to score",
        f"{prediction.both_teams_to_score:.1%}",
    )

    f.metric(
        "⚽ Over 2.5 goals",
        f"{prediction.over_2_5:.1%}",
    )

    if provider_mode:
        st.subheader(
            "⚽ What This Forecast Knows"
        )

        st.write(
            "Available inputs include historical result form, "
            "home and away scoring, recent goals, pre-match "
            "points rate and probabilistic score outcomes."
        )

        st.subheader(
            "🏟️ What Additional Football Data Could Add"
        )

        unavailable_rows = [
            {
                "Analysis": "Expected line-up impact",
                "Required data": (
                    "Line-ups, player minutes and availability"
                ),
                "Potential extension": (
                    "Adjust team strength for likely starters"
                ),
            },
            {
                "Analysis": "Transfer impact",
                "Required data": (
                    "Current squads, transfers and player ratings"
                ),
                "Potential extension": (
                    "Apply player-strength changes before forecasting"
                ),
            },
            {
                "Analysis": "Attacking-side tendencies",
                "Required data": (
                    "Event coordinates for passes, carries and shots"
                ),
                "Potential extension": (
                    "Estimate left, central and right attack shares"
                ),
            },
            {
                "Analysis": "Formation prediction",
                "Required data": (
                    "Line-ups, player positions or tracking data"
                ),
                "Potential extension": (
                    "Predict likely base and in-possession shapes"
                ),
            },
            {
                "Analysis": "Pressing behaviour",
                "Required data": (
                    "Event sequences or tracking data"
                ),
                "Potential extension": (
                    "Estimate pressing height, PPDA and regains"
                ),
            },
            {
                "Analysis": "Goalscorer prediction",
                "Required data": (
                    "Player shots, xG, minutes and expected line-ups"
                ),
                "Potential extension": (
                    "Calculate player-specific scoring probabilities"
                ),
            },
        ]

        st.dataframe(
            pd.DataFrame(unavailable_rows),
            use_container_width=True,
            hide_index=True,
        )

    else:
        left, right = st.columns(2)

        with left:
            st.subheader(
                f"🥅 Synthetic likely scorers: {home_team}"
            )

            st.dataframe(
                scorer_probabilities(
                    events,
                    shots,
                    home_team,
                    home_rate,
                )
                .head(10)
                .round(3),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.subheader(
                f"🥅 Synthetic likely scorers: {away_team}"
            )

            st.dataframe(
                scorer_probabilities(
                    events,
                    shots,
                    away_team,
                    away_rate,
                )
                .head(10)
                .round(3),
                use_container_width=True,
                hide_index=True,
            )


elif page == "team_identity":
    st.header(
        "👕 Team Identity and Attack-Side Tendencies"
    )

    st.warning(
        "This page currently uses synthetic event coordinates. "
        "Real attacking-side and playing-style conclusions "
        "require licensed event or tracking data."
    )

    side = attacking_side_profile(events)
    styles = team_style_profiles(events)

    identity = styles.merge(
        side,
        on="team",
        how="left",
    )

    selected_team = st.selectbox(
        "👕 Team",
        sorted(identity["team"]),
    )

    row = identity.loc[
        identity["team"] == selected_team
    ].iloc[0]

    st.subheader(
        f"⚽ {selected_team}: {row['style_label']}"
    )

    a, b, c = st.columns(3)

    a.metric(
        "⚽ Left-side share",
        f"{row['left_share']:.1%}",
    )

    b.metric(
        "⚽ Central share",
        f"{row['centre_share']:.1%}",
    )

    c.metric(
        "⚽ Right-side share",
        f"{row['right_share']:.1%}",
    )

    attack_channels = pd.DataFrame(
        {
            "channel": [
                "Left",
                "Centre",
                "Right",
            ],
            "share": [
                row["left_share"],
                row["centre_share"],
                row["right_share"],
            ],
        }
    )

    st.plotly_chart(
        px.bar(
            attack_channels,
            x="channel",
            y="share",
            title="⚽ Attacking Action Distribution",
        ),
        use_container_width=True,
    )

    st.dataframe(
        identity.round(3),
        use_container_width=True,
        hide_index=True,
    )


elif page == "match_centre":
    st.header("⚽ Executive Match Centre")

    summary = team_match_summary(
        match_events,
        match_shots,
    )

    c1, c2 = st.columns(2)

    for column, row in zip(
        [c1, c2],
        summary.itertuples(index=False),
    ):
        with column:
            st.subheader(f"👕 {row.team}")

            a, b, c = st.columns(3)

            a.metric(
                "🥅 Goals",
                int(row.goals),
            )

            b.metric(
                "🥅 xG",
                f"{row.xg:.2f}",
            )

            c.metric(
                "⚽ Shots",
                int(row.shots),
            )

            d, e, f = st.columns(3)

            d.metric(
                "⚽ Field tilt",
                f"{row.field_tilt:.1f}%",
            )

            e.metric(
                "⚽ PPDA",
                f"{row.ppda:.1f}",
            )

            f.metric(
                "🥅 Box entries",
                int(row.box_entries),
            )

    momentum = xg_momentum(match_shots)

    fig = px.line(
        momentum,
        x="match_minute",
        y="cumulative_xg",
        color="team",
        markers=True,
        title="🥅 Cumulative Expected Goals",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("⚽ Analyst Interpretation")

    st.write(
        "Use the metrics as a structured starting point, "
        "then validate the causes with video. Field tilt "
        "describes territorial control, PPDA approximates "
        "pressing intensity and xG separates shot volume "
        "from shot quality."
    )

    st.dataframe(
        summary.round(2),
        use_container_width=True,
    )


elif page == "shot_xg_lab":
    st.header("🥅 Shot and Expected Goals Lab")

    selected_team = st.selectbox(
        "👕 Team",
        ["All"] + teams,
    )

    view = (
        match_shots
        if selected_team == "All"
        else match_shots.loc[
            match_shots["team"] == selected_team
        ]
    )

    fig = px.scatter(
        view,
        x="x",
        y="y",
        color="team",
        size="xg",
        symbol="shot_goal",
        hover_data=[
            "player",
            "xg",
            "distance_to_goal",
            "shot_angle",
            "body_part",
            "play_pattern",
            "outcome",
        ],
        title="🥅 Shot Map — Marker Size Represents xG",
    )

    fig.update_xaxes(
        range=[0, 105]
    )

    fig.update_yaxes(
        range=[0, 68]
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.subheader("🥅 Shot Quality by Player")

        player_shots = (
            view.groupby(
                ["team", "player"],
                as_index=False,
            )
            .agg(
                shots=("event_id", "size"),
                goals=("shot_goal", "sum"),
                xg=("xg", "sum"),
                average_xg=("xg", "mean"),
            )
        )

        st.dataframe(
            player_shots
            .sort_values(
                "xg",
                ascending=False,
            )
            .round(3),
            use_container_width=True,
        )

    with right:
        st.subheader(
            "⚽ Shot Quality Distribution"
        )

        st.plotly_chart(
            px.histogram(
                view,
                x="xg",
                color="team",
                nbins=15,
            ),
            use_container_width=True,
        )


elif page == "possession_threat":
    st.header("⚽ Possession and Expected Threat")

    grid = xt_grid_frame(xt_model)

    heatmap = go.Figure(
        data=go.Heatmap(
            z=xt_model.grid,
            x=np.linspace(
                0,
                105,
                xt_model.x_bins,
            ),
            y=np.linspace(
                0,
                68,
                xt_model.y_bins,
            ),
            colorbar_title="xT",
        )
    )

    heatmap.update_layout(
        title="⚽ Expected Threat Surface",
        xaxis_title="Pitch length",
        yaxis_title="Pitch width",
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
    )

    team = st.selectbox(
        "👕 Team for action analysis",
        teams,
        key="xt_team",
    )

    team_actions = match_events.loc[
        (match_events["team"] == team)
        & (match_events["xt_added"] != 0)
    ].copy()

    st.subheader(
        "⚽ Highest-Value Progression Actions"
    )

    columns = [
        "minute",
        "player",
        "event_type",
        "x",
        "y",
        "end_x",
        "end_y",
        "xt_added",
    ]

    st.dataframe(
        team_actions
        .sort_values(
            "xt_added",
            ascending=False,
        )[columns]
        .head(25)
        .round(4),
        use_container_width=True,
    )

    player_threat = (
        team_actions.groupby(
            "player",
            as_index=False,
        )["xt_added"]
        .sum()
        .sort_values(
            "xt_added",
            ascending=False,
        )
    )

    st.plotly_chart(
        px.bar(
            player_threat.head(12),
            x="player",
            y="xt_added",
            title="⚽ xT Added by Player",
        ),
        use_container_width=True,
    )


elif page == "passing_network":
    st.header("⚽ Passing Network")

    team = st.selectbox(
        "👕 Team",
        teams,
        key="network_team",
    )

    nodes, edges, graph = passing_network(
        match_events,
        team,
    )

    fig = go.Figure()

    for edge in edges.itertuples(index=False):
        source = nodes.loc[
            nodes["player"] == edge.player
        ]

        target = nodes.loc[
            nodes["player"] == edge.recipient
        ]

        if source.empty or target.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=[
                    source.iloc[0]["x"],
                    target.iloc[0]["x"],
                ],
                y=[
                    source.iloc[0]["y"],
                    target.iloc[0]["y"],
                ],
                mode="lines",
                line={
                    "width": max(
                        1,
                        edge.pass_count / 2,
                    )
                },
                hoverinfo="text",
                text=(
                    f"{edge.player} → "
                    f"{edge.recipient}: "
                    f"{edge.pass_count}"
                ),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=nodes["x"],
            y=nodes["y"],
            mode="markers+text",
            text=nodes["player"],
            textposition="top center",
            marker={
                "size": (
                    14
                    + nodes["touches"]
                    / max(
                        1,
                        nodes["touches"].max(),
                    )
                    * 28
                )
            },
            customdata=nodes[
                [
                    "touches",
                    "network_centrality",
                ]
            ],
            hovertemplate=(
                "%{text}"
                "<br>Touches: %{customdata[0]}"
                "<br>Centrality: %{customdata[1]:.3f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"⚽ {team} Passing Network",
        xaxis={"range": [0, 105]},
        yaxis={"range": [0, 68]},
        height=650,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        nodes
        .sort_values(
            "network_centrality",
            ascending=False,
        )
        .round(3),
        use_container_width=True,
    )


elif page == "player_recruitment":
    st.header(
        "👕 Player Recruitment and Similarity"
    )

    players = player_summary(
        events,
        shots,
    )

    players = cluster_players(players)

    selected_player = st.selectbox(
        "👕 Reference player",
        sorted(players["player"].unique()),
    )

    profile = percentile_profile(
        players,
        selected_player,
    )

    radar = go.Figure()

    radar.add_trace(
        go.Scatterpolar(
            r=profile["percentile"],
            theta=profile["metric"],
            fill="toself",
            name=selected_player,
        )
    )

    radar.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
            }
        },
        showlegend=False,
        title=(
            "⚽ Percentile Profile Across "
            "the Synthetic Demo League"
        ),
    )

    st.plotly_chart(
        radar,
        use_container_width=True,
    )

    st.subheader("👕 Most Similar Players")

    similar = player_similarity(
        players,
        selected_player,
        8,
    )

    st.dataframe(
        similar.round(3),
        use_container_width=True,
    )

    st.caption(
        "Similarity uses standardised synthetic event "
        "metrics and cosine distance. Role, age, "
        "competition strength, current availability and "
        "physical data would be required for real "
        "recruitment decisions."
    )


elif page == "model_governance":
    st.header("🏆 Model Governance")

    metrics = xg_artifact.metrics

    columns = st.columns(4)

    roc_auc = metrics.get("roc_auc")
    average_precision = metrics.get(
        "average_precision"
    )
    brier_score = metrics.get("brier_score")
    log_loss = metrics.get("log_loss")

    columns[0].metric(
        "⚽ ROC AUC",
        (
            f"{roc_auc:.3f}"
            if roc_auc is not None
            else "Not defined"
        ),
    )

    columns[1].metric(
        "🥅 Average precision",
        (
            f"{average_precision:.3f}"
            if average_precision is not None
            else "Not defined"
        ),
    )

    columns[2].metric(
        "⚽ Brier score",
        (
            f"{brier_score:.3f}"
            if brier_score is not None
            else "Not defined"
        ),
    )

    columns[3].metric(
        "🏆 Log loss",
        (
            f"{log_loss:.3f}"
            if log_loss is not None
            else "Not defined"
        ),
    )

    calibration = xg_artifact.calibration

    fig = px.line(
        calibration,
        x="predicted_probability",
        y="observed_goal_rate",
        markers=True,
        title="🥅 xG Calibration Curve",
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("⚽ Governance Notes")

    st.markdown(
        """
- Match IDs are used to separate training and test data, reducing leakage between events from the same match.
- Match forecasting features are created chronologically and use only information available before each fixture.
- Metrics cover ranking quality, probability accuracy, goal error and calibration.
- Synthetic data is useful for software testing and workflow demonstrations, but not for real sporting validation.
- A real deployment needs multiple recent seasons, temporal validation, league-specific calibration and drift monitoring.
- Current-team forecasts should account for promoted clubs, transfers, injuries, expected line-ups and coaching changes.
- Real tactical conclusions require licensed event or tracking data.
- Model outputs should support, not replace, video analysis and expert football judgement.
"""
    )