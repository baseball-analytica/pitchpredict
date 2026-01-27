# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import argparse
import sys

from pitchpredict.cli.utils import error_console


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'predict' subcommand group to the CLI.
    """
    predict_parser = subparsers.add_parser(
        "predict",
        help="Prediction commands",
        description="Run predictions for pitchers, batters, and batted balls.",
    )
    predict_subparsers = predict_parser.add_subparsers(
        dest="predict_command",
        required=True,
        help="Prediction type",
    )

    # predict pitcher
    _add_pitcher_parser(predict_subparsers)

    # predict batter
    _add_batter_parser(predict_subparsers)

    # predict batted-ball
    _add_batted_ball_parser(predict_subparsers)


def _add_pitcher_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'predict pitcher' subcommand.
    """
    parser = subparsers.add_parser(
        "pitcher",
        help="Predict next pitch",
        description="Predict the next pitch type, location, and outcome for a pitcher/batter matchup.",
    )
    parser.add_argument(
        "pitcher",
        help="Pitcher name or MLBAM ID",
    )
    parser.add_argument(
        "batter",
        help="Batter name or MLBAM ID",
    )
    parser.add_argument(
        "--balls",
        "-b",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Ball count (0-3, default: 0)",
    )
    parser.add_argument(
        "--strikes",
        "-s",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Strike count (0-2, default: 0)",
    )
    parser.add_argument(
        "--outs",
        "-o",
        type=int,
        choices=[0, 1, 2],
        help="Number of outs (0-2)",
    )
    parser.add_argument(
        "--inning",
        "-i",
        type=int,
        help="Inning number",
    )
    parser.add_argument(
        "--date",
        "-d",
        help="Game date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        default="similarity",
        help="Algorithm to use (default: similarity)",
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
        action="store_true",
        help="Do not compact the console output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.set_defaults(func=handle_predict_pitcher)


def _add_batter_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'predict batter' subcommand.
    """
    parser = subparsers.add_parser(
        "batter",
        help="Predict batter outcome",
        description="Predict the outcome for a batter given an incoming pitch.",
    )
    parser.add_argument(
        "batter",
        help="Batter name or MLBAM ID",
    )
    parser.add_argument(
        "pitcher",
        help="Pitcher name or MLBAM ID",
    )
    parser.add_argument(
        "type",
        help="Pitch type (e.g., FF, SL, CU)",
    )
    parser.add_argument(
        "speed",
        type=float,
        help="Pitch speed in mph",
    )
    parser.add_argument(
        "release_x",
        type=float,
        help="Pitch horizontal location",
    )
    parser.add_argument(
        "release_z",
        type=float,
        help="Pitch vertical location",
    )
    parser.add_argument(
        "--balls",
        "-b",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Ball count (0-3, default: 0)",
    )
    parser.add_argument(
        "--strikes",
        "-s",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Strike count (0-2, default: 0)",
    )
    parser.add_argument(
        "--score-bat",
        type=int,
        default=0,
        help="Batting team score (default: 0)",
    )
    parser.add_argument(
        "--score-fld",
        type=int,
        default=0,
        help="Fielding team score (default: 0)",
    )
    parser.add_argument(
        "--date",
        "-d",
        help="Game date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        default="similarity",
        help="Algorithm to use (default: similarity)",
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
        action="store_true",
        help="Do not compact the console output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.set_defaults(func=handle_predict_batter)


def _add_batted_ball_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'predict batted-ball' subcommand.
    """
    parser = subparsers.add_parser(
        "batted-ball",
        help="Predict batted ball outcome",
        description="Predict the outcome of a batted ball given launch parameters.",
    )
    parser.add_argument(
        "launch_speed",
        type=float,
        help="Exit velocity in mph",
    )
    parser.add_argument(
        "launch_angle",
        type=float,
        help="Launch angle in degrees",
    )
    parser.add_argument(
        "--spray-angle",
        type=float,
        help="Spray angle in degrees",
    )
    parser.add_argument(
        "--bb-type",
        help="Batted ball type (ground_ball, line_drive, fly_ball, popup)",
    )
    parser.add_argument(
        "--outs",
        "-o",
        type=int,
        choices=[0, 1, 2],
        help="Number of outs (0-2)",
    )
    parser.add_argument(
        "--bases-state",
        type=int,
        help="Base state (0-7, binary encoding of occupied bases)",
    )
    parser.add_argument(
        "--batter-id",
        type=int,
        help="Batter MLBAM ID",
    )
    parser.add_argument(
        "--date",
        "-d",
        help="Game date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        default="similarity",
        help="Algorithm to use (default: similarity)",
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
        action="store_true",
        help="Do not compact the console output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.set_defaults(func=handle_predict_batted_ball)


def handle_predict_pitcher(args: argparse.Namespace) -> None:
    """
    Handle the 'predict pitcher' command.
    """
    from fastapi import HTTPException

    from pitchpredict.cli.output import format_pitcher_prediction
    from pitchpredict.cli.utils import get_api, resolve_player, run_async

    api = get_api()

    async def _run():
        pitcher_id = await resolve_player(args.pitcher, api)
        batter_id = await resolve_player(args.batter, api)

        result = await api.predict_pitcher(
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            count_balls=args.balls,
            count_strikes=args.strikes,
            outs=args.outs,
            inning=args.inning,
            game_date=args.date,
            algorithm=args.algorithm,
        )
        return result

    try:
        result = run_async(_run())
        format_pitcher_prediction(
            result,
            format_type=args.format,
            verbose=args.verbose,
            compact=not args.no_compact,
        )
    except HTTPException as e:
        error_console.print(f"[red]Error:[/red] {e.detail}")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def handle_predict_batter(args: argparse.Namespace) -> None:
    """
    Handle the 'predict batter' command.
    """
    from datetime import date

    from fastapi import HTTPException

    from pitchpredict.cli.output import format_batter_prediction
    from pitchpredict.cli.utils import get_api, run_async

    api = get_api()
    game_date = args.date or date.today().isoformat()

    async def _run():
        result = await api.predict_batter(
            batter_name=args.batter,
            pitcher_name=args.pitcher,
            balls=args.balls,
            strikes=args.strikes,
            score_bat=args.score_bat,
            score_fld=args.score_fld,
            game_date=game_date,
            pitch_type=args.type,
            pitch_speed=args.speed,
            pitch_x=args.release_x,
            pitch_z=args.release_z,
            algorithm=args.algorithm,
        )
        return result

    try:
        result = run_async(_run())
        format_batter_prediction(
            result,
            format_type=args.format,
            verbose=args.verbose,
            compact=not args.no_compact,
        )
    except HTTPException as e:
        error_console.print(f"[red]Error:[/red] {e.detail}")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def handle_predict_batted_ball(args: argparse.Namespace) -> None:
    """
    Handle the 'predict batted-ball' command.
    """
    from fastapi import HTTPException

    from pitchpredict.cli.output import format_batted_ball_prediction
    from pitchpredict.cli.utils import get_api, run_async

    api = get_api()

    async def _run():
        result = await api.predict_batted_ball(
            launch_speed=args.launch_speed,
            launch_angle=args.launch_angle,
            algorithm=args.algorithm,
            spray_angle=args.spray_angle,
            bb_type=args.bb_type,
            outs=args.outs,
            bases_state=args.bases_state,
            batter_id=args.batter_id,
            game_date=args.date,
        )
        return result

    try:
        result = run_async(_run())
        format_batted_ball_prediction(
            result,
            format_type=args.format,
            verbose=args.verbose,
            compact=not args.no_compact,
        )
    except HTTPException as e:
        error_console.print(f"[red]Error:[/red] {e.detail}")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
