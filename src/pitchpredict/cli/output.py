# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from pitchpredict.types.api import PredictPitcherResponse
    from pitchpredict.types.api import PredictBatterResponse

console = Console()


def _get_player_name(record: dict[str, Any]) -> str:
    """Extract player name from a record, handling various field formats."""
    # Try combined name fields first
    if name := record.get("name_display_first_last"):
        return name
    if name := record.get("name_last_first"):
        return name

    # Build from first/last name fields
    first = record.get("name_first", "")
    last = record.get("name_last", "")
    if first or last:
        # Capitalize properly
        first = first.title() if first else ""
        last = last.title() if last else ""
        return f"{first} {last}".strip()

    return "Unknown"


def _get_player_id(record: dict[str, Any]) -> str:
    """Extract MLBAM ID from a record."""
    if player_id := record.get("player_id"):
        return str(player_id)
    if key_mlbam := record.get("key_mlbam"):
        return str(key_mlbam)
    return "N/A"


def _spacer(compact: bool) -> None:
    if not compact:
        console.print()


def _table_kwargs(compact: bool) -> dict[str, Any]:
    if not compact:
        return {}
    return {
        "padding": (0, 0),
        "pad_edge": False,
        "collapse_padding": True,
    }


def _panel_kwargs(compact: bool) -> dict[str, Any]:
    if not compact:
        return {}
    return {"padding": (0, 1)}


def _format_stats_table(
    data: dict[str, Any],
    title: str,
    key_style: str = "cyan",
    value_style: str = "white",
    compact: bool = True,
) -> Table:
    """Format a flat stats dictionary as a Rich table."""
    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        **_table_kwargs(compact),
    )
    table.add_column("Metric", style=key_style)
    table.add_column("Value", justify="right", style=value_style)

    for key, value in data.items():
        if isinstance(value, dict):
            # Skip nested dicts - they'll be handled separately
            continue
        elif isinstance(value, float):
            if "prob" in key.lower() or "pctg" in key.lower():
                table.add_row(key, f"{value:.1%}")
            elif abs(value) < 0.01 and value != 0:
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, f"{value:.2f}")
        else:
            table.add_row(key, str(value))

    return table


def _format_probs_table(
    probs: dict[str, float],
    title: str,
    key_header: str = "Type",
    key_style: str = "cyan",
    compact: bool = True,
) -> Table:
    """Format a probability dictionary as a sorted Rich table."""
    table = Table(title=title, **_table_kwargs(compact))
    table.add_column(key_header, style=key_style)
    table.add_column("Probability", justify="right", style="green")

    for key, prob in sorted(probs.items(), key=lambda x: -x[1]):
        table.add_row(key, f"{prob:.1%}")

    return table


def format_pitcher_prediction(
    response: "PredictPitcherResponse",
    format_type: str = "rich",
    verbose: bool = False,
    compact: bool = True,
) -> None:
    """
    Format and display pitcher prediction results.

    Args:
        response: The prediction response from the API
        format_type: Output format ('rich' or 'json')
        verbose: Whether to show detailed output
    """
    if format_type == "json":
        print(json.dumps(response.model_dump(), indent=2, default=str))
        return

    basic = response.basic_pitch_data
    probs = basic.get("pitch_type_probs", {})

    # Main pitch type probabilities table
    table = _format_probs_table(
        probs, "Pitch Type Probabilities", "Pitch Type", compact=compact
    )
    console.print(table)

    # Velocity and location summary
    _spacer(compact)
    if speed_mean := basic.get("pitch_speed_mean"):
        speed_std = basic.get("pitch_speed_std", 0)
        console.print(
            f"[bold]Expected Velocity:[/bold] {speed_mean:.1f} mph "
            f"[dim](std: {speed_std:.1f})[/dim]"
        )

    if basic.get("pitch_x_mean") is not None and basic.get("pitch_z_mean") is not None:
        x_mean = basic["pitch_x_mean"]
        z_mean = basic["pitch_z_mean"]
        console.print(f"[bold]Expected Location:[/bold] ({x_mean:.2f}, {z_mean:.2f})")

    # Basic outcome summary
    if response.basic_outcome_data:
        outcome_probs = response.basic_outcome_data.get("outcome_probs", {})
        if outcome_probs:
            _spacer(compact)
            table = _format_probs_table(
                outcome_probs,
                "Outcome Probabilities",
                "Outcome",
                "yellow",
                compact=compact,
            )
            console.print(table)

    if verbose:
        _format_pitcher_verbose(response, compact=compact)


def _format_pitcher_verbose(
    response: "PredictPitcherResponse", compact: bool = True
) -> None:
    """Format verbose output for pitcher predictions."""
    detailed = response.detailed_pitch_data or {}
    outcome = response.detailed_outcome_data or {}

    # Fastball vs Offspeed breakdown
    if "pitch_prob_fastball" in detailed:
        _spacer(compact)
        console.print(
            Panel(
                f"[bold]Fastball:[/bold] {detailed['pitch_prob_fastball']:.1%}    "
                f"[bold]Offspeed:[/bold] {detailed['pitch_prob_offspeed']:.1%}",
                title="[cyan]Pitch Category Breakdown[/cyan]",
                border_style="blue",
                **_panel_kwargs(compact),
            )
        )

    # Fastball details
    if fastball_data := detailed.get("pitch_data_fastballs"):
        _spacer(compact)
        if fb_probs := fastball_data.get("pitch_type_probs"):
            console.print(
                _format_probs_table(fb_probs, "Fastball Types", "Type", compact=compact)
            )

        fb_stats = {
            k: v
            for k, v in fastball_data.items()
            if k != "pitch_type_probs" and not isinstance(v, dict)
        }
        if fb_stats:
            console.print(
                _format_stats_table(fb_stats, "Fastball Metrics", compact=compact)
            )

    # Offspeed details
    if offspeed_data := detailed.get("pitch_data_offspeed"):
        _spacer(compact)
        if os_probs := offspeed_data.get("pitch_type_probs"):
            console.print(
                _format_probs_table(
                    os_probs, "Offspeed Types", "Type", compact=compact
                )
            )

        os_stats = {
            k: v
            for k, v in offspeed_data.items()
            if k != "pitch_type_probs" and not isinstance(v, dict)
        }
        if os_stats:
            console.print(
                _format_stats_table(os_stats, "Offspeed Metrics", compact=compact)
            )

    # Swing data
    if swing_data := outcome.get("swing_data"):
        _spacer(compact)
        console.print(_format_stats_table(swing_data, "Swing Metrics", compact=compact))

    # Contact data
    if contact_data := outcome.get("contact_data"):
        _spacer(compact)
        if bb_probs := contact_data.get("bb_type_probs"):
            console.print(
                _format_probs_table(
                    bb_probs, "Batted Ball Types", "Type", "yellow", compact=compact
                )
            )

        contact_stats = {
            k: v
            for k, v in contact_data.items()
            if k != "bb_type_probs" and not isinstance(v, dict)
        }
        if contact_stats:
            console.print(
                _format_stats_table(contact_stats, "Contact Metrics", compact=compact)
            )

    # Metadata
    if response.prediction_metadata:
        _spacer(compact)
        meta = response.prediction_metadata
        console.print(
            Panel(
                f"[bold]Duration:[/bold] {meta.get('duration', 0):.3f}s\n"
                f"[bold]Pitches Sampled:[/bold] {meta.get('n_pitches_sampled', 'N/A')} "
                f"/ {meta.get('n_pitches_total', 'N/A')} "
                f"({meta.get('sample_pctg', 0):.1%})",
                title="[dim]Prediction Metadata[/dim]",
                border_style="dim",
                **_panel_kwargs(compact),
            )
        )


def format_batter_prediction(
    response: dict[str, Any] | "PredictBatterResponse",
    format_type: str = "rich",
    verbose: bool = False,
    compact: bool = True,
) -> None:
    """
    Format and display batter prediction results.
    """
    data = response
    if hasattr(response, "model_dump"):
        data = response.model_dump()

    if format_type == "json":
        print(json.dumps(data, indent=2, default=str))
        return

    basic = data.get("basic_outcome_data", {})
    outcome_probs = basic.get("outcome_probs", {})

    if outcome_probs:
        table = _format_probs_table(
            outcome_probs,
            "Batter Outcome Probabilities",
            "Outcome",
            "yellow",
            compact=compact,
        )
        console.print(table)

    if verbose and (detailed := data.get("detailed_outcome_data")):
        _format_detailed_outcome(detailed, compact=compact)


def format_batted_ball_prediction(
    response: dict[str, Any],
    format_type: str = "rich",
    verbose: bool = False,
    compact: bool = True,
) -> None:
    """
    Format and display batted ball prediction results.
    """
    if format_type == "json":
        print(json.dumps(response, indent=2, default=str))
        return

    basic = response.get("basic_outcome_data", {})
    outcome_probs = basic.get("outcome_probs", {})

    if outcome_probs:
        table = _format_probs_table(
            outcome_probs,
            "Batted Ball Outcome Probabilities",
            "Outcome",
            "yellow",
            compact=compact,
        )
        console.print(table)

    if verbose and (detailed := response.get("detailed_outcome_data")):
        _format_detailed_outcome(detailed, compact=compact)


def _format_detailed_outcome(
    detailed: dict[str, Any], compact: bool = True
) -> None:
    """Format detailed outcome data with proper handling of nested structures."""
    for section_name, section_data in detailed.items():
        _spacer(compact)
        if isinstance(section_data, dict):
            # Check for probability sub-dicts
            prob_keys = [k for k in section_data if "prob" in k.lower()]
            for prob_key in prob_keys:
                if isinstance(section_data[prob_key], dict):
                    title = (
                        f"{section_name.replace('_', ' ').title()} - "
                        f"{prob_key.replace('_', ' ').title()}"
                    )
                    console.print(
                        _format_probs_table(
                            section_data[prob_key], title, "Type", compact=compact
                        )
                    )

            # Format remaining stats
            stats = {
                k: v
                for k, v in section_data.items()
                if not isinstance(v, dict) and k not in prob_keys
            }
            if stats:
                title = section_name.replace("_", " ").title()
                console.print(_format_stats_table(stats, title, compact=compact))
        elif isinstance(section_data, float):
            console.print(f"[bold]{section_name}:[/bold] {section_data:.3f}")
        else:
            console.print(f"[bold]{section_name}:[/bold] {section_data}")


def format_player_results(
    results: list[dict[str, Any]],
    format_type: str = "rich",
    compact: bool = True,
) -> None:
    """
    Format and display player lookup results.
    """
    if format_type == "json":
        print(json.dumps(results, indent=2, default=str))
        return

    if not results:
        console.print("[yellow]No players found.[/yellow]")
        return

    table = Table(title="Player Search Results", **_table_kwargs(compact))
    table.add_column("Name", style="cyan")
    table.add_column("MLBAM ID", justify="right", style="green")
    table.add_column("Years Active", style="yellow")

    for player in results:
        name = _get_player_name(player)
        player_id = _get_player_id(player)

        # Build years active string
        first_year = player.get("mlb_played_first")
        last_year = player.get("mlb_played_last")
        if first_year and last_year:
            years = f"{int(first_year)}-{int(last_year)}"
        elif first_year:
            years = f"{int(first_year)}-"
        elif last_year:
            years = f"-{int(last_year)}"
        else:
            years = "N/A"

        table.add_row(name, player_id, years)

    console.print(table)


def format_player_info(
    record: dict[str, Any],
    mlbam_id: int,
    format_type: str = "rich",
    compact: bool = True,
) -> None:
    """
    Format and display detailed player information.
    """
    if format_type == "json":
        print(
            json.dumps({"mlbam_id": mlbam_id, "record": record}, indent=2, default=str)
        )
        return

    name = _get_player_name(record)

    panel_content = []
    panel_content.append(f"[bold]MLBAM ID:[/bold] {mlbam_id}")

    # Years active
    first_year = record.get("mlb_played_first")
    last_year = record.get("mlb_played_last")
    if first_year or last_year:
        first = int(first_year) if first_year else "?"
        last = int(last_year) if last_year else "?"
        panel_content.append(f"[bold]Years Active:[/bold] {first}-{last}")

    # External IDs
    if bbref := record.get("key_bbref"):
        panel_content.append(f"[bold]Baseball-Reference:[/bold] {bbref}")

    if fangraphs := record.get("key_fangraphs"):
        panel_content.append(f"[bold]FanGraphs ID:[/bold] {fangraphs}")

    if retro := record.get("key_retro"):
        panel_content.append(f"[bold]Retrosheet ID:[/bold] {retro}")

    # Legacy fields (may be present in some data sources)
    if position := record.get("primary_position") or record.get("position"):
        panel_content.append(f"[bold]Position:[/bold] {position}")

    if team := record.get("team_abbrev") or record.get("team_code"):
        panel_content.append(f"[bold]Team:[/bold] {team}")

    if bats := record.get("bats"):
        panel_content.append(f"[bold]Bats:[/bold] {bats}")

    if throws := record.get("throws"):
        panel_content.append(f"[bold]Throws:[/bold] {throws}")

    if birth_date := record.get("birth_date"):
        panel_content.append(f"[bold]Birth Date:[/bold] {birth_date}")

    panel = Panel(
        "\n".join(panel_content),
        title=f"[cyan]{name}[/cyan]",
        border_style="blue",
        **_panel_kwargs(compact),
    )
    console.print(panel)


def format_cache_status(
    cache_dir: str,
    stats: dict[str, Any],
    format_type: str = "rich",
    compact: bool = True,
) -> None:
    """
    Format and display cache status information.
    """
    if format_type == "json":
        print(json.dumps({"cache_dir": cache_dir, **stats}, indent=2, default=str))
        return

    console.print(
        Panel.fit(
            f"[bold]Location:[/bold] {cache_dir}\n"
            f"[bold]Total Size:[/bold] {_format_size(stats.get('total_size', 0))}",
            title="[cyan]PitchPredict Cache Status[/cyan]",
            border_style="blue",
            **_panel_kwargs(compact),
        )
    )

    table = Table(**_table_kwargs(compact))
    table.add_column("Category", style="cyan")
    table.add_column("Files", justify="right", style="green")
    table.add_column("Size", justify="right", style="yellow")

    for category, info in stats.get("categories", {}).items():
        table.add_row(
            category.title(),
            str(info.get("files", 0)),
            _format_size(info.get("size", 0)),
        )

    console.print(table)


def _format_size(size_bytes: int) -> str:
    """Format a size in bytes to a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.2f} TB"
