# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import argparse
import sys

from pitchpredict.cli.utils import error_console


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'player' subcommand group to the CLI.
    """
    player_parser = subparsers.add_parser(
        "player",
        help="Player lookup commands",
        description="Search for players and retrieve player information.",
    )
    player_subparsers = player_parser.add_subparsers(
        dest="player_command",
        required=True,
        help="Player command",
    )

    # player lookup
    _add_lookup_parser(player_subparsers)

    # player info
    _add_info_parser(player_subparsers)


def _add_lookup_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'player lookup' subcommand.
    """
    parser = subparsers.add_parser(
        "lookup",
        help="Search player by name",
        description="Search for players by name using fuzzy matching.",
    )
    parser.add_argument(
        "name",
        help="Player name to search for",
    )
    parser.add_argument(
        "--exact",
        "-e",
        action="store_true",
        help="Disable fuzzy matching (exact match only)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)",
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
    parser.set_defaults(func=handle_player_lookup)


def _add_info_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'player info' subcommand.
    """
    parser = subparsers.add_parser(
        "info",
        help="Get player info by MLBAM ID",
        description="Retrieve detailed information for a player by their MLBAM ID.",
    )
    parser.add_argument(
        "mlbam_id",
        type=int,
        help="Player MLBAM ID",
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
    parser.set_defaults(func=handle_player_info)


def handle_player_lookup(args: argparse.Namespace) -> None:
    """
    Handle the 'player lookup' command.
    """
    from fastapi import HTTPException

    from pitchpredict.cli.output import format_player_results
    from pitchpredict.cli.utils import get_api, run_async

    api = get_api()
    fuzzy = not args.exact

    async def _run():
        results = await api.get_player_records_from_name(
            player_name=args.name,
            fuzzy_lookup=fuzzy,
            limit=args.limit,
        )
        return results

    try:
        results = run_async(_run())
        format_player_results(
            results,
            format_type=args.format,
            compact=not args.no_compact,
        )
    except HTTPException as e:
        error_console.print(f"[red]Error:[/red] {e.detail}")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def handle_player_info(args: argparse.Namespace) -> None:
    """
    Handle the 'player info' command.
    """
    from fastapi import HTTPException

    from pitchpredict.cli.output import format_player_info
    from pitchpredict.cli.utils import get_api, run_async

    api = get_api()

    async def _run():
        record = await api.get_player_record_from_id(mlbam_id=args.mlbam_id)
        return record

    try:
        record = run_async(_run())
        format_player_info(
            record,
            args.mlbam_id,
            format_type=args.format,
            compact=not args.no_compact,
        )
    except HTTPException as e:
        error_console.print(f"[red]Error:[/red] {e.detail}")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
