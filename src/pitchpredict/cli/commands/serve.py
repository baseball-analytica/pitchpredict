# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'serve' subcommand to the CLI.
    """
    parser = subparsers.add_parser(
        "serve",
        help="Start the PitchPredict FastAPI server",
        description="Start the PitchPredict server for API access to predictions.",
    )
    parser.add_argument(
        "--host",
        "-H",
        default="0.0.0.0",
        help="The host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8056,
        help="The port to bind the server to (default: 8056)",
    )
    parser.add_argument(
        "--reload",
        "-r",
        action="store_true",
        help="Reload the server when code changes are detected",
    )
    parser.set_defaults(func=handle_serve)


def handle_serve(args: argparse.Namespace) -> None:
    """
    Handle the 'serve' command.
    """
    from pitchpredict.server import run_server

    run_server(host=args.host, port=args.port, reload=args.reload)
