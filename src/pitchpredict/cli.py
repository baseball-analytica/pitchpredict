# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import argparse

from pitchpredict.server import run_server


def handle_subcommand(args: argparse.Namespace):
    """
    Handle the subcommand specified by the user with the given arguments.
    """
    match args.subcommand:
        case "serve":
            run_server(host=args.host, port=args.port, reload=args.reload)
        case _:
            raise ValueError(f"Invalid subcommand: {args.subcommand}")


def run_cli():
    """
    Run the PitchPredict CLI.
    """
    parser = argparse.ArgumentParser(
        description="Predict MLB pitcher/batter behavior and outcomes using a given context",
        epilog="https://github.com/baseball-analytica/pitchpredict"
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # command 'serve'
    parser_serve = subparsers.add_parser("serve", help="start the PitchPredict server")
    parser_serve.add_argument(
        "--host", "-H", default="0.0.0.0", help="the host to bind the server to"
    )
    parser_serve.add_argument(
        "--port", "-p", type=int, default=8056, help="the port to bind the server to"
    )
    parser_serve.add_argument(
        "--reload",
        "-r",
        action="store_true",
        help="reload the server when code changes are detected",
    )

    args = parser.parse_args()

    handle_subcommand(args)


if __name__ == "__main__":
    run_cli()
