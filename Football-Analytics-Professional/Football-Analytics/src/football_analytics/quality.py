from __future__ import annotations

from dataclasses import dataclass, asdict
import pandas as pd

REQUIRED_EVENT_COLUMNS = {
    "event_id", "match_id", "team", "opponent", "period", "minute", "second",
    "timestamp_seconds", "possession_id", "player", "event_type", "outcome",
    "x", "y", "end_x", "end_y", "under_pressure", "shot_goal"
}

ALLOWED_EVENT_TYPES = {
    "Pass", "Carry", "Shot", "Duel", "Interception", "Recovery", "Pressure", "Foul"
}


@dataclass
class QualityIssue:
    severity: str
    check: str
    count: int
    message: str


@dataclass
class QualityReport:
    rows: int
    matches: int
    issues: list[QualityIssue]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "matches": self.matches,
            "passed": self.passed,
            "issues": [asdict(issue) for issue in self.issues],
        }


def validate_events(events: pd.DataFrame) -> QualityReport:
    issues: list[QualityIssue] = []
    missing = REQUIRED_EVENT_COLUMNS.difference(events.columns)
    if missing:
        issues.append(QualityIssue("error", "required_columns", len(missing), f"Missing: {sorted(missing)}"))
        return QualityReport(len(events), events.get("match_id", pd.Series(dtype=str)).nunique(), issues)

    duplicate_count = int(events["event_id"].duplicated().sum())
    if duplicate_count:
        issues.append(QualityIssue("error", "unique_event_id", duplicate_count, "Duplicate event IDs"))

    null_identity = int(events[["match_id", "team", "player", "event_type"]].isna().any(axis=1).sum())
    if null_identity:
        issues.append(QualityIssue("error", "identity_fields", null_identity, "Missing identity fields"))

    invalid_x = int((~events["x"].between(0, 105)).sum())
    invalid_y = int((~events["y"].between(0, 68)).sum())
    invalid_end_x = int((~events["end_x"].between(0, 105)).sum())
    invalid_end_y = int((~events["end_y"].between(0, 68)).sum())
    for check, count in [
        ("x_bounds", invalid_x), ("y_bounds", invalid_y),
        ("end_x_bounds", invalid_end_x), ("end_y_bounds", invalid_end_y),
    ]:
        if count:
            issues.append(QualityIssue("error", check, count, "Coordinates outside pitch bounds"))

    unknown_types = set(events["event_type"].dropna().unique()).difference(ALLOWED_EVENT_TYPES)
    if unknown_types:
        issues.append(QualityIssue("warning", "event_types", len(unknown_types), f"Unknown: {sorted(unknown_types)}"))

    impossible_goals = int(((events["shot_goal"]) & (events["event_type"] != "Shot")).sum())
    if impossible_goals:
        issues.append(QualityIssue("error", "goal_event_consistency", impossible_goals, "Goals outside shot events"))

    time_issues = 0
    for _, frame in events.groupby("match_id", sort=False):
        ordered = frame.sort_values(["period", "timestamp_seconds"])
        if not ordered["timestamp_seconds"].is_monotonic_increasing:
            time_issues += 1
    if time_issues:
        issues.append(QualityIssue("warning", "event_order", time_issues, "Matches with non-monotonic timestamps"))

    return QualityReport(len(events), events["match_id"].nunique(), issues)
