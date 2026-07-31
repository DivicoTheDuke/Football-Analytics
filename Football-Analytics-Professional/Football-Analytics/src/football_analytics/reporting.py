from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px

from .metrics import team_match_summary, xg_momentum


def build_match_report(events: pd.DataFrame, shots: pd.DataFrame, match_id: str, output: str | Path) -> Path:
    match_events = events.loc[events["match_id"] == match_id].copy()
    match_shots = shots.loc[shots["match_id"] == match_id].copy()
    if match_events.empty:
        raise KeyError(f"Unknown match_id: {match_id}")

    summary = team_match_summary(match_events, match_shots)
    momentum = xg_momentum(match_shots)
    shot_map = px.scatter(
        match_shots, x="x", y="y", color="team", size="xg", symbol="shot_goal",
        hover_data=["player", "xg", "outcome"], title="Shot Map"
    )
    momentum_chart = px.line(
        momentum, x="match_minute", y="cumulative_xg", color="team",
        markers=True, title="Cumulative Expected Goals"
    )

    title = f"Match Analysis Report – {match_id}"
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>
    <style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:40px auto;line-height:1.5}}
    table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border:1px solid #ddd;text-align:right}}
    th:first-child,td:first-child{{text-align:left}}.note{{background:#f4f4f4;padding:16px}}</style></head><body>
    <h1>{title}</h1>
    <p class='note'>Portfolio demonstration using synthetic data. Validate all definitions and models before operational use.</p>
    <h2>Executive summary</h2>{summary.round(3).to_html(index=False)}
    <h2>Shot locations</h2>{shot_map.to_html(full_html=False, include_plotlyjs='cdn')}
    <h2>Match momentum</h2>{momentum_chart.to_html(full_html=False, include_plotlyjs=False)}
    <h2>Analyst interpretation checklist</h2>
    <ul><li>Compare shot quantity and shot quality.</li><li>Review whether field tilt translated into box entries.</li>
    <li>Identify high-value possessions and recurring progression routes.</li><li>Validate observations against video.</li></ul>
    </body></html>"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
