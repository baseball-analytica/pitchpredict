# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import argparse
import shutil
import sys
from pathlib import Path

from pitchpredict.cli.utils import console, error_console


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'cache' subcommand group to the CLI.
    """
    cache_parser = subparsers.add_parser(
        "cache",
        help="Cache management commands",
        description="Manage the PitchPredict local cache.",
    )
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_command",
        required=True,
        help="Cache command",
    )

    # cache status
    _add_status_parser(cache_subparsers)

    # cache clear
    _add_clear_parser(cache_subparsers)

    # cache warm
    _add_warm_parser(cache_subparsers)


def _add_status_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'cache status' subcommand.
    """
    parser = subparsers.add_parser(
        "status",
        help="Show cache statistics",
        description="Display cache location, size, and contents summary.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".pitchpredict_cache",
        help="Cache directory path (default: .pitchpredict_cache)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["rich", "json"],
        default="rich",
        help="Output format (default: rich)",
    )
    parser.add_argument(
        "--no-compact",
        action="store_false",
        help="Do not compact the console output",
    )
    parser.set_defaults(func=handle_cache_status)


def _add_clear_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'cache clear' subcommand.
    """
    parser = subparsers.add_parser(
        "clear",
        help="Clear the cache",
        description="Remove all cached data.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".pitchpredict_cache",
        help="Cache directory path (default: .pitchpredict_cache)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--category",
        choices=["pitcher", "batter", "batted_ball", "players", "all"],
        default="all",
        help="Category to clear (default: all)",
    )
    parser.set_defaults(func=handle_cache_clear)


def _add_warm_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'cache warm' subcommand.
    """
    parser = subparsers.add_parser(
        "warm",
        help="Pre-populate cache for a player or batted ball data",
        description="Fetch and cache data to speed up future predictions. "
        "For player data (pitcher/batter), provide a player name or ID. "
        "For batted ball data, use --type batted-ball (no player required).",
    )
    parser.add_argument(
        "player",
        nargs="?",
        default=None,
        help="Player name or MLBAM ID (required for pitcher/batter, not used for batted-ball)",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["pitcher", "batter", "both", "batted-ball"],
        default="both",
        help="Data type to cache (default: both). Use 'batted-ball' for league-wide batted ball data.",
    )
    parser.add_argument(
        "--seasons",
        "-n",
        type=int,
        default=3,
        help="Number of seasons to fetch for batted ball data (default: 3)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".pitchpredict_cache",
        help="Cache directory path (default: .pitchpredict_cache)",
    )
    parser.set_defaults(func=handle_cache_warm)


def _get_dir_stats(path: Path) -> dict:
    """
    Get file count and total size for a directory.
    """
    if not path.exists():
        return {"files": 0, "size": 0}

    files = 0
    size = 0
    for item in path.iterdir():
        if item.is_file():
            files += 1
            size += item.stat().st_size
        elif item.is_dir():
            sub_stats = _get_dir_stats(item)
            files += sub_stats["files"]
            size += sub_stats["size"]
    return {"files": files, "size": size}


def handle_cache_status(args: argparse.Namespace) -> None:
    """
    Handle the 'cache status' command.
    """
    from pitchpredict.cli.output import format_cache_status

    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        console.print(f"[yellow]Cache directory does not exist:[/yellow] {cache_dir}")
        return

    categories = {}
    total_size = 0

    for category in ["pitcher", "batter", "batted_ball", "players"]:
        cat_path = cache_dir / category
        stats = _get_dir_stats(cat_path)
        categories[category] = stats
        total_size += stats["size"]

    stats = {
        "total_size": total_size,
        "categories": categories,
    }

    format_cache_status(
        str(cache_dir),
        stats,
        format_type=args.format,
        compact=not args.no_compact,
    )


def handle_cache_clear(args: argparse.Namespace) -> None:
    """
    Handle the 'cache clear' command.
    """
    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        console.print(f"[yellow]Cache directory does not exist:[/yellow] {cache_dir}")
        return

    # Determine what to clear
    if args.category == "all":
        targets = [cache_dir]
        target_desc = "entire cache"
    else:
        targets = [cache_dir / args.category]
        target_desc = f"{args.category} cache"

    # Check if targets exist
    existing_targets = [t for t in targets if t.exists()]
    if not existing_targets:
        console.print(f"[yellow]No {target_desc} to clear.[/yellow]")
        return

    # Calculate size
    total_size = sum(_get_dir_stats(t)["size"] for t in existing_targets)

    # Confirm
    if not args.confirm:
        from rich.prompt import Confirm

        console.print(f"[bold]About to clear:[/bold] {target_desc}")
        console.print(f"[bold]Size:[/bold] {total_size / (1024 * 1024):.2f} MB")

        if not Confirm.ask("Are you sure you want to continue?"):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    # Clear
    try:
        for target in existing_targets:
            if target == cache_dir:
                # Clear all contents but keep the directory
                for item in cache_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            else:
                shutil.rmtree(target)

        console.print(f"[green]Successfully cleared {target_desc}.[/green]")
    except Exception as e:
        error_console.print(f"[red]Error clearing cache:[/red] {e}")
        sys.exit(1)


def handle_cache_warm(args: argparse.Namespace) -> None:
    """
    Handle the 'cache warm' command.
    """
    from datetime import date

    from fastapi import HTTPException
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from pitchpredict.cli.utils import get_api, resolve_player, run_async

    # Handle batted-ball warming (doesn't require a player)
    if args.type == "batted-ball":
        _warm_batted_ball_cache(args)
        return

    # For player-specific warming, require a player argument
    if args.player is None:
        error_console.print(
            "[red]Error:[/red] Player name or ID is required for pitcher/batter cache warming.\n"
            "Use --type batted-ball if you want to warm the batted ball cache."
        )
        sys.exit(1)

    api = get_api()

    async def _warm_player():
        # First resolve the player
        player_id = await resolve_player(args.player, api)
        console.print(f"[cyan]Resolved player ID:[/cyan] {player_id}")

        # Get today's date for the fetch
        today = date.today().isoformat()

        if args.type in ("pitcher", "both"):
            console.print("[cyan]Warming pitcher cache...[/cyan]")
            # Trigger a fetch that will populate the cache
            # We do this by making a prediction request with minimal context
            # The API will fetch and cache the pitcher's data
            try:
                # Use a dummy batter ID for warming
                await api.predict_pitcher(
                    pitcher_id=player_id,
                    batter_id=660271,  # Shohei Ohtani as a common batter
                    game_date=today,
                    algorithm="similarity",
                )
                console.print("[green]Pitcher cache warmed.[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning during pitcher warm:[/yellow] {e}")

        if args.type in ("batter", "both"):
            console.print("[cyan]Warming batter cache...[/cyan]")
            try:
                # Use a dummy pitcher and simulate a pitch
                await api.predict_batter(
                    batter_name=str(player_id),
                    pitcher_name="592789",  # Corbin Burnes as a common pitcher
                    balls=0,
                    strikes=0,
                    score_bat=0,
                    score_fld=0,
                    game_date=today,
                    pitch_type="FF",
                    pitch_speed=95.0,
                    pitch_x=0.0,
                    pitch_z=2.5,
                    algorithm="similarity",
                )
                console.print("[green]Batter cache warmed.[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning during batter warm:[/yellow] {e}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Warming cache...", total=None)
            run_async(_warm_player())

        console.print("[green]Cache warming complete.[/green]")
    except HTTPException as e:
        error_console.print(f"[red]Error:[/red] {e.detail}")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def _warm_batted_ball_cache(args: argparse.Namespace) -> None:
    """
    Warm the batted ball cache by fetching league-wide batted ball data.
    """
    from datetime import datetime

    from fastapi import HTTPException
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from pitchpredict.backend.caching import PitchPredictCache
    from pitchpredict.backend.fetching import get_all_batted_balls
    from pitchpredict.cli.utils import run_async

    cache = PitchPredictCache(cache_dir=args.cache_dir)

    # Calculate date range
    current_year = datetime.now().year
    start_year = current_year - args.seasons
    start_date = f"{start_year}-03-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    console.print("[cyan]Warming batted ball cache...[/cyan]")
    console.print(
        f"[dim]Date range: {start_date} to {end_date} ({args.seasons} seasons)[/dim]"
    )

    async def _fetch():
        df = await get_all_batted_balls(
            start_date=start_date,
            end_date=end_date,
            n_seasons=args.seasons,
            cache=cache,
        )
        return len(df)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(
                description="Fetching batted ball data (this may take a while)...",
                total=None,
            )
            count = run_async(_fetch())

        console.print(
            f"[green]Batted ball cache warmed.[/green] ({count:,} events cached)"
        )
    except HTTPException as e:
        error_console.print(f"[red]Error:[/red] {e.detail}")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
