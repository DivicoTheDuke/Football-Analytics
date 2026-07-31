from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math
import random
import urllib.error
import urllib.request

import pandas as pd


FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
DEFAULT_CACHE_PATH = Path("data/reference/fpl_bootstrap_static.json")
COMPETITION = "Premier League"
SEASON = "2026/27"

# FPL position IDs are stable:
# 1 Goalkeeper, 2 Defender, 3 Midfielder, 4 Forward.
FPL_POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

POSITION_START = {
    "GK": (8, 34),
    "RB": (30, 10),
    "CB": (25, 27),
    "LB": (30, 58),
    "DM": (45, 34),
    "CM": (55, 30),
    "AM": (70, 35),
    "RW": (72, 10),
    "LW": (72, 58),
    "ST": (82, 34),
}


@dataclass(frozen=True)
class Player:
    name: str
    broad_position: str
    position: str


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalise_name(first_name: str, second_name: str, web_name: str) -> str:
    full_name = " ".join(
        part.strip() for part in (first_name, second_name) if part and part.strip()
    )
    return full_name or web_name.strip()


def _download_fpl_bootstrap(timeout: int = 30) -> dict:
    request = urllib.request.Request(
        FPL_BOOTSTRAP_URL,
        headers={
            "User-Agent": (
                "Football-Analytics portfolio project "
                "(educational, non-commercial use)"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_fpl_bootstrap(
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    *,
    refresh: bool = True,
) -> dict:
    """
    Load the current official Fantasy Premier League bootstrap dataset.

    When refresh=True, the current dataset is downloaded and cached locally.
    If the request fails, an existing cache is used. This keeps demo generation
    reproducible while allowing the club and player list to stay current.
    """
    cache_path = Path(cache_path)

    if refresh:
        try:
            data = _download_fpl_bootstrap()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if not cache_path.exists():
                raise RuntimeError(
                    "Could not download the current Premier League squads and "
                    f"no cache exists at {cache_path}. Connect to the internet "
                    "and run the command again."
                ) from exc

    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached FPL dataset found at {cache_path}. "
            "Run once with internet access and refresh=True."
        )

    return json.loads(cache_path.read_text(encoding="utf-8"))


def _assign_detailed_positions(players: list[tuple[str, str]]) -> list[Player]:
    """
    Convert broad FPL positions into balanced football-analysis positions.

    FPL does not distinguish centre-backs from full-backs or central
    midfielders from wingers. The detailed labels here are therefore synthetic
    role assignments used only by the demo event generator.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, broad_position in players:
        grouped[broad_position].append(name)

    assigned: list[Player] = []

    for index, name in enumerate(grouped["GK"]):
        assigned.append(Player(name, "GK", "GK"))

    defender_cycle = ["RB", "CB", "CB", "LB", "CB", "RB", "LB"]
    for index, name in enumerate(grouped["DEF"]):
        assigned.append(
            Player(name, "DEF", defender_cycle[index % len(defender_cycle)])
        )

    midfielder_cycle = ["DM", "CM", "AM", "RW", "LW", "CM", "AM", "DM"]
    for index, name in enumerate(grouped["MID"]):
        assigned.append(
            Player(name, "MID", midfielder_cycle[index % len(midfielder_cycle)])
        )

    forward_cycle = ["ST", "RW", "LW", "ST"]
    for index, name in enumerate(grouped["FWD"]):
        assigned.append(
            Player(name, "FWD", forward_cycle[index % len(forward_cycle)])
        )

    return assigned


def load_current_clubs(
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    *,
    refresh: bool = True,
) -> dict[str, list[Player]]:
    """
    Return all current Premier League clubs and FPL-listed first-team players.

    Players marked as removed are excluded. The exact squad reflects the
    official FPL data available when this function is run.
    """
    bootstrap = load_fpl_bootstrap(cache_path, refresh=refresh)

    team_names = {
        int(team["id"]): str(team["name"])
        for team in bootstrap.get("teams", [])
    }
    clubs: dict[str, list[tuple[str, str]]] = {
        team_name: [] for team_name in team_names.values()
    }

    for element in bootstrap.get("elements", []):
        if element.get("removed", False):
            continue

        team_id = int(element["team"])
        team_name = team_names.get(team_id)
        if not team_name:
            continue

        broad_position = FPL_POSITION_MAP.get(int(element["element_type"]))
        if broad_position is None:
            continue

        player_name = _normalise_name(
            str(element.get("first_name", "")),
            str(element.get("second_name", "")),
            str(element.get("web_name", "")),
        )
        clubs[team_name].append((player_name, broad_position))

    detailed_clubs = {
        team: _assign_detailed_positions(players)
        for team, players in sorted(clubs.items())
        if players
    }

    if len(detailed_clubs) != 20:
        raise ValueError(
            "Expected 20 Premier League clubs, but found "
            f"{len(detailed_clubs)}. The FPL season may not yet be initialised."
        )

    return detailed_clubs


def _select_starting_xi(
    squad: list[Player],
    rng: random.Random,
) -> list[Player]:
    """
    Select a balanced 4-5-1 starting XI from the complete current squad.

    The event generator needs exactly eleven active players per match. The
    complete squad is still written to lineups.csv, with substitutes receiving
    zero minutes.
    """
    by_broad: dict[str, list[Player]] = defaultdict(list)
    for player in squad:
        by_broad[player.broad_position].append(player)

    for players in by_broad.values():
        rng.shuffle(players)

    required = {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1}
    selected: list[Player] = []

    for broad_position, count in required.items():
        selected.extend(by_broad[broad_position][:count])

    if len(selected) < 11:
        selected_names = {player.name for player in selected}
        remaining = [
            player for player in squad if player.name not in selected_names
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: 11 - len(selected)])

    if len(selected) != 11:
        raise ValueError("A club does not contain enough players for a starting XI.")

    return selected


def _pick_player(
    players: list[Player],
    rng: random.Random,
    *,
    attacking: bool = False,
) -> Player:
    weights = []
    for player in players:
        position = player.position
        if attacking:
            weights.append(
                {
                    "ST": 5.0,
                    "RW": 4.0,
                    "LW": 4.0,
                    "AM": 3.0,
                    "CM": 2.0,
                }.get(position, 0.5)
            )
        else:
            weights.append(
                {
                    "GK": 0.4,
                    "CB": 2.0,
                    "RB": 1.5,
                    "LB": 1.5,
                    "DM": 2.0,
                    "CM": 2.0,
                    "AM": 1.5,
                    "RW": 1.0,
                    "LW": 1.0,
                    "ST": 0.8,
                }.get(position, 1.0)
            )

    return rng.choices(players, weights=weights, k=1)[0]


def _write_squad_reference(
    clubs: dict[str, list[Player]],
    output_dir: Path,
) -> None:
    rows = []
    for team, squad in clubs.items():
        for player in squad:
            rows.append(
                {
                    "team": team,
                    "player": player.name,
                    "fpl_position": player.broad_position,
                    "synthetic_demo_position": player.position,
                    "source": "Official Fantasy Premier League bootstrap-static API",
                    "season": SEASON,
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "squads.csv", index=False)


def generate_demo(
    output_dir: str | Path = "data/demo",
    matches: int = 38,
    seed: int = 42,
    *,
    refresh_squads: bool = True,
    squad_cache_path: str | Path = DEFAULT_CACHE_PATH,
):
    """
    Generate synthetic event data using the current Premier League clubs and squads.

    Club and player identities come from the official FPL API. Every event,
    score, lineup decision and performance value is synthetic and must not be
    interpreted as real player or club performance data.
    """
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clubs = load_current_clubs(
        cache_path=squad_cache_path,
        refresh=refresh_squads,
    )
    club_names = list(clubs)

    match_rows = []
    event_rows = []
    lineup_rows = []
    event_counter = 1

    for match_number in range(1, matches + 1):
        match_id = f"M{match_number:04d}"
        home, away = rng.sample(club_names, 2)
        date = pd.Timestamp("2026-08-21") + pd.Timedelta(days=match_number * 5)
        home_goals = 0
        away_goals = 0

        starting_xi = {
            home: _select_starting_xi(clubs[home], rng),
            away: _select_starting_xi(clubs[away], rng),
        }

        for team in (home, away):
            starter_names = {player.name for player in starting_xi[team]}
            for player in clubs[team]:
                is_starter = player.name in starter_names
                lineup_rows.append(
                    {
                        "match_id": match_id,
                        "team": team,
                        "player": player.name,
                        "position": player.position,
                        "fpl_position": player.broad_position,
                        "starter": is_starter,
                        "minutes": 90 if is_starter else 0,
                    }
                )

        timestamp = 0.0
        possession_id = 0
        current_team = home if rng.random() < 0.5 else away

        for _ in range(rng.randint(185, 225)):
            possession_id += 1
            current_team = (
                current_team
                if rng.random() < 0.18
                else (away if current_team == home else home)
            )
            opponent = away if current_team == home else home
            pattern = rng.choices(
                ["Open Play", "Counter", "Corner", "Free Kick", "Throw In"],
                [0.68, 0.12, 0.07, 0.06, 0.07],
                k=1,
            )[0]
            x = rng.uniform(5, 35) if pattern != "Counter" else rng.uniform(25, 50)
            y = rng.uniform(10, 58)
            possession_events = rng.randint(3, 10)

            for step in range(possession_events):
                timestamp += rng.uniform(2.0, 7.0)
                if timestamp > 90 * 60:
                    break

                period = 1 if timestamp < 45 * 60 else 2
                minute = int(timestamp // 60)
                second = int(timestamp % 60)

                player = _pick_player(
                    starting_xi[current_team],
                    rng,
                    attacking=x > 65,
                )
                position = player.position

                shot_probability = max(0.0, (x - 65) / 160) + (
                    0.06 if step >= 4 else 0
                )
                turnover_probability = 0.06 + max(0, (x - 80) / 180)

                if x > 72 and rng.random() < shot_probability:
                    event_type = "Shot"
                    end_x = 105.0
                    end_y = rng.uniform(30.5, 37.5)
                    distance = math.sqrt((105 - x) ** 2 + (34 - y) ** 2)
                    centrality = max(0, 1 - abs(34 - y) / 34)
                    under_pressure = rng.random() < 0.47
                    first_time = rng.random() < 0.18
                    assisted = step > 0 and rng.random() < 0.78
                    body_part = rng.choices(
                        ["Right Foot", "Left Foot", "Head", "Other"],
                        [0.44, 0.34, 0.18, 0.04],
                        k=1,
                    )[0]

                    logit = (
                        -2.8
                        + 0.12 * (22 - distance)
                        + 0.75 * centrality
                        + 0.30 * assisted
                        + 0.20 * first_time
                        - 0.35 * under_pressure
                    )
                    if body_part == "Head":
                        logit -= 0.35
                    if pattern in {"Corner", "Free Kick"}:
                        logit -= 0.15

                    goal_probability = 1 / (1 + math.exp(-logit))
                    shot_goal = rng.random() < goal_probability
                    outcome = (
                        "Goal"
                        if shot_goal
                        else rng.choices(
                            ["Saved", "Off Target", "Blocked"],
                            [0.46, 0.34, 0.20],
                            k=1,
                        )[0]
                    )
                    recipient = ""
                    key_pass = False

                    if shot_goal:
                        if current_team == home:
                            home_goals += 1
                        else:
                            away_goals += 1
                else:
                    under_pressure = rng.random() < (
                        0.25 + 0.15 * (x > 70)
                    )
                    first_time = False
                    assisted = False
                    body_part = ""
                    shot_goal = False
                    key_pass = x > 70 and rng.random() < 0.10

                    event_type = rng.choices(
                        [
                            "Pass",
                            "Carry",
                            "Duel",
                            "Pressure",
                            "Interception",
                            "Recovery",
                            "Foul",
                        ],
                        [0.56, 0.17, 0.08, 0.08, 0.04, 0.05, 0.02],
                        k=1,
                    )[0]

                    if event_type == "Pass":
                        progress = rng.gauss(11 if x < 70 else 6, 10)
                        lateral = rng.gauss(0, 13)
                        end_x = _clip(x + progress, 0, 105)
                        end_y = _clip(y + lateral, 0, 68)
                        complete_probability = (
                            0.91
                            - 0.0025 * max(0, end_x - x)
                            - (0.08 if under_pressure else 0)
                        )
                        complete = rng.random() < complete_probability
                        outcome = "Complete" if complete else "Incomplete"
                        recipient_player = _pick_player(
                            starting_xi[current_team],
                            rng,
                            attacking=end_x > 65,
                        )
                        recipient = recipient_player.name
                    elif event_type == "Carry":
                        end_x = _clip(x + rng.uniform(3, 15), 0, 105)
                        end_y = _clip(y + rng.gauss(0, 7), 0, 68)
                        outcome = "Complete"
                        recipient = ""
                    else:
                        end_x = x
                        end_y = y
                        outcome = (
                            rng.choice(["Won", "Lost"])
                            if event_type == "Duel"
                            else "Complete"
                        )
                        recipient = ""

                event_rows.append(
                    {
                        "event_id": f"E{event_counter:08d}",
                        "match_id": match_id,
                        "competition": COMPETITION,
                        "season": SEASON,
                        "match_date": date.date().isoformat(),
                        "home_team": home,
                        "away_team": away,
                        "team": current_team,
                        "opponent": opponent,
                        "period": period,
                        "minute": minute,
                        "second": second,
                        "timestamp_seconds": round(timestamp, 2),
                        "possession_id": possession_id,
                        "player": player.name,
                        "position": position,
                        "recipient": recipient,
                        "event_type": event_type,
                        "outcome": outcome,
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "end_x": round(end_x, 2),
                        "end_y": round(end_y, 2),
                        "body_part": body_part,
                        "play_pattern": pattern,
                        "under_pressure": under_pressure,
                        "first_time": first_time,
                        "assisted": assisted,
                        "key_pass": key_pass,
                        "shot_goal": shot_goal,
                        "synthetic_data": True,
                    }
                )
                event_counter += 1

                if (
                    event_type == "Shot"
                    or outcome in {"Incomplete", "Lost"}
                    or rng.random() < turnover_probability
                ):
                    break

                x, y = end_x, end_y

            if timestamp > 90 * 60:
                break

        match_rows.append(
            {
                "match_id": match_id,
                "competition": COMPETITION,
                "season": SEASON,
                "match_date": date.date().isoformat(),
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "synthetic_data": True,
            }
        )

    matches_df = pd.DataFrame(match_rows)
    events_df = pd.DataFrame(event_rows)
    lineups_df = pd.DataFrame(lineup_rows)

    matches_df.to_csv(output_dir / "matches.csv", index=False)
    events_df.to_csv(output_dir / "events.csv", index=False)
    lineups_df.to_csv(output_dir / "lineups.csv", index=False)
    _write_squad_reference(clubs, output_dir)

    return matches_df, events_df, lineups_df