from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

MOVE_TYPES = {"Pass", "Carry"}


@dataclass
class XTModel:
    grid: np.ndarray
    x_bins: int
    y_bins: int
    iterations: int
    convergence: float

    def zone(self, x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        xi = np.clip((x.to_numpy(float) / 105 * self.x_bins).astype(int), 0, self.x_bins - 1)
        yi = np.clip((y.to_numpy(float) / 68 * self.y_bins).astype(int), 0, self.y_bins - 1)
        return xi, yi


def _zone_indices(frame: pd.DataFrame, x_col: str, y_col: str, x_bins: int, y_bins: int):
    xi = np.clip((frame[x_col].to_numpy(float) / 105 * x_bins).astype(int), 0, x_bins - 1)
    yi = np.clip((frame[y_col].to_numpy(float) / 68 * y_bins).astype(int), 0, y_bins - 1)
    return xi, yi


def train_xt(events: pd.DataFrame, x_bins: int = 16, y_bins: int = 12, max_iter: int = 200, tol: float = 1e-6) -> XTModel:
    valid = events.copy()
    valid["complete_move"] = valid["event_type"].isin(MOVE_TYPES) & (valid["outcome"] == "Complete")
    valid["is_shot"] = valid["event_type"] == "Shot"

    sx, sy = _zone_indices(valid, "x", "y", x_bins, y_bins)
    ex, ey = _zone_indices(valid, "end_x", "end_y", x_bins, y_bins)

    total = np.zeros((y_bins, x_bins), dtype=float)
    shots = np.zeros_like(total)
    goals = np.zeros_like(total)
    moves = np.zeros_like(total)
    transition = np.zeros((y_bins, x_bins, y_bins, x_bins), dtype=float)

    for index in range(len(valid)):
        total[sy[index], sx[index]] += 1
        if valid.iloc[index]["is_shot"]:
            shots[sy[index], sx[index]] += 1
            goals[sy[index], sx[index]] += int(valid.iloc[index]["shot_goal"])
        elif valid.iloc[index]["complete_move"]:
            moves[sy[index], sx[index]] += 1
            transition[sy[index], sx[index], ey[index], ex[index]] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        p_shot = np.divide(shots, total, out=np.zeros_like(shots), where=total > 0)
        p_move = np.divide(moves, total, out=np.zeros_like(moves), where=total > 0)
        p_goal = np.divide(goals, shots, out=np.zeros_like(goals), where=shots > 0)
        transition_prob = np.divide(
            transition,
            moves[:, :, None, None],
            out=np.zeros_like(transition),
            where=moves[:, :, None, None] > 0,
        )

    values = p_shot * p_goal
    convergence = float("inf")
    iterations = 0
    for iterations in range(1, max_iter + 1):
        continuation = np.einsum("abij,ij->ab", transition_prob, values)
        updated = p_shot * p_goal + p_move * continuation
        convergence = float(np.max(np.abs(updated - values)))
        values = updated
        if convergence < tol:
            break

    return XTModel(values, x_bins, y_bins, iterations, convergence)


def apply_xt(model: XTModel, events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    sx, sy = model.zone(result["x"], result["y"])
    ex, ey = model.zone(result["end_x"], result["end_y"])
    result["start_xt"] = model.grid[sy, sx]
    result["end_xt"] = model.grid[ey, ex]
    eligible = result["event_type"].isin(MOVE_TYPES) & (result["outcome"] == "Complete")
    result["xt_added"] = np.where(eligible, result["end_xt"] - result["start_xt"], 0.0)
    return result


def xt_grid_frame(model: XTModel) -> pd.DataFrame:
    rows = []
    for y in range(model.y_bins):
        for x in range(model.x_bins):
            rows.append({"x_bin": x, "y_bin": y, "x_start": x * 105 / model.x_bins,
                         "y_start": y * 68 / model.y_bins, "xt": model.grid[y, x]})
    return pd.DataFrame(rows)
