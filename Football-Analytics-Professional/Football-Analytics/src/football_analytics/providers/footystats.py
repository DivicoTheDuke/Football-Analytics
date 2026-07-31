from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://api.football-data-api.com"


@dataclass(frozen=True)
class FootyStatsImportResult:
    raw_path: Path
    matches_path: Path
    seasons: tuple[str, ...]
    match_count: int
    request_count: int


def _request_json(url: str, timeout_seconds: int = 30) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "football-analytics-portfolio/2.2",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - fixed provider URL
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FootyStats returned a non-object JSON response")
    return payload


def _response_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data", [])
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("matches", "league_matches", "results"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _first(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return default


def _normalise_date(row: dict[str, Any]) -> str:
    unix_value = _first(row, "date_unix", "timestamp", "unix")
    if unix_value is not None:
        try:
            return datetime.fromtimestamp(int(unix_value), tz=timezone.utc).date().isoformat()
        except (TypeError, ValueError, OSError):
            pass

    value = _first(row, "date", "match_date", "date_GMT", "dateGMT")
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def normalise_footystats_matches(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Map FootyStats match objects to the project's cached match-history schema."""
    output: list[dict[str, Any]] = []
    for row in rows:
        match_id = _first(row, "id", "match_id")
        home_team = _first(row, "home_name", "homeTeam", "home_team_name")
        away_team = _first(row, "away_name", "awayTeam", "away_team_name")
        if match_id is None or not home_team or not away_team:
            continue

        home_goals = _first(row, "homeGoalCount", "home_goals", "team_a_score")
        away_goals = _first(row, "awayGoalCount", "away_goals", "team_b_score")
        home_xg = _first(row, "team_a_xg", "home_xg", "homeExpectedGoals")
        away_xg = _first(row, "team_b_xg", "away_xg", "awayExpectedGoals")

        output.append(
            {
                "match_id": f"FS-{match_id}",
                "provider_match_id": str(match_id),
                "competition": "Premier League",
                "season": str(_first(row, "season", default="unknown")),
                "match_date": _normalise_date(row),
                "home_team": str(home_team),
                "away_team": str(away_team),
                "home_goals": pd.to_numeric(home_goals, errors="coerce"),
                "away_goals": pd.to_numeric(away_goals, errors="coerce"),
                "home_xg": pd.to_numeric(home_xg, errors="coerce"),
                "away_xg": pd.to_numeric(away_xg, errors="coerce"),
                "status": str(_first(row, "status", default="unknown")),
                "game_week": pd.to_numeric(
                    _first(row, "game_week", "revised_game_week"), errors="coerce"
                ),
                "provider": "FootyStats",
                "synthetic_data": False,
            }
        )

    frame = pd.DataFrame(output)
    if frame.empty:
        raise ValueError("No usable FootyStats match records were returned")

    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    frame = frame.drop_duplicates("provider_match_id", keep="last")
    frame = frame.sort_values(["match_date", "provider_match_id"]).reset_index(drop=True)
    return frame


def fetch_premier_league_once(
    *,
    api_key: str,
    league_ids: list[int],
    raw_path: str | Path,
    matches_path: str | Path,
    force: bool = False,
    timeout_seconds: int = 30,
    max_pages: int = 100,
) -> FootyStatsImportResult:
    """Run one guarded import job and cache every paginated provider response.

    "Once" means one explicit ingestion run. FootyStats pagination can require
    multiple HTTP requests to retrieve a full season. The function refuses to
    contact the provider again while the raw cache exists unless ``force=True``.
    """
    if not api_key:
        raise ValueError("A FootyStats API key is required")
    if not league_ids:
        raise ValueError("At least one FootyStats Premier League league_id is required")

    raw_path = Path(raw_path)
    matches_path = Path(matches_path)
    if raw_path.exists() and not force:
        raise FileExistsError(
            f"FootyStats cache already exists at {raw_path}. "
            "No provider request was made. Use the cached data or pass --force deliberately."
        )

    all_rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    request_count = 0

    for league_id in league_ids:
        seen_ids: set[str] = set()
        for page in range(1, max_pages + 1):
            query = urlencode({"key": api_key, "league_id": league_id, "page": page})
            url = f"{BASE_URL}/league-matches?{query}"
            payload = _request_json(url, timeout_seconds=timeout_seconds)
            request_count += 1
            rows = _response_rows(payload)
            if not rows:
                break

            new_rows = []
            for row in rows:
                record_id = str(_first(row, "id", "match_id", default=""))
                if record_id and record_id in seen_ids:
                    continue
                if record_id:
                    seen_ids.add(record_id)
                new_rows.append(row)

            if not new_rows:
                break

            all_rows.extend(new_rows)
            pages.append({"league_id": league_id, "page": page, "response": payload})

            # The documented EPL season contains 380 matches. Stop once the
            # response metadata says there is no next page, when available.
            pager = payload.get("pager") or payload.get("pagination") or {}
            if isinstance(pager, dict):
                current = pager.get("current_page") or pager.get("page")
                last = pager.get("max_page") or pager.get("last_page") or pager.get("total_pages")
                if current is not None and last is not None and int(current) >= int(last):
                    break

    if not all_rows:
        raise ValueError("FootyStats returned no match rows")

    raw_document = {
        "provider": "FootyStats",
        "endpoint": "league-matches",
        "competition": "Premier League",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "league_ids": league_ids,
        "request_count": request_count,
        "pages": pages,
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw_document, indent=2), encoding="utf-8")

    matches = normalise_footystats_matches(all_rows)
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(matches_path, index=False)

    return FootyStatsImportResult(
        raw_path=raw_path,
        matches_path=matches_path,
        seasons=tuple(sorted(matches["season"].dropna().astype(str).unique())),
        match_count=len(matches),
        request_count=request_count,
    )
