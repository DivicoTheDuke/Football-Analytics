from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCOUTING_FEATURES = [
    "pass_completion", "progressive_passes", "progressive_carries", "final_third_entries",
    "box_entries", "key_passes", "xa", "defensive_actions", "xt_added", "shots", "xg"
]


def player_similarity(players: pd.DataFrame, player: str, n_neighbors: int = 6) -> pd.DataFrame:
    if player not in set(players["player"]):
        raise KeyError(f"Unknown player: {player}")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("neighbors", NearestNeighbors(metric="cosine")),
    ])
    X = players[SCOUTING_FEATURES]
    transformed = model.named_steps["scale"].fit_transform(model.named_steps["imputer"].fit_transform(X))
    neighbors = model.named_steps["neighbors"]
    neighbors.fit(transformed)
    index = players.index[players["player"] == player][0]
    distances, indices = neighbors.kneighbors(transformed[[index]], n_neighbors=min(n_neighbors, len(players)))
    result = players.iloc[indices[0]].copy()
    result["similarity"] = 1 - distances[0]
    return result[["team", "player", "similarity"] + SCOUTING_FEATURES]


def cluster_players(players: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> pd.DataFrame:
    n_clusters = min(n_clusters, max(2, len(players)))
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("cluster", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)),
    ])
    result = players.copy()
    result["role_cluster"] = pipeline.fit_predict(players[SCOUTING_FEATURES])
    return result


def percentile_profile(players: pd.DataFrame, player: str) -> pd.DataFrame:
    row = players.loc[players["player"] == player]
    if row.empty:
        raise KeyError(f"Unknown player: {player}")
    values = []
    for feature in SCOUTING_FEATURES:
        percentile = float(players[feature].rank(pct=True).loc[row.index[0]] * 100)
        values.append({"metric": feature, "value": float(row.iloc[0][feature]), "percentile": percentile})
    return pd.DataFrame(values)
