# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import argparse

from pitchpredict.cli.commands import serve, version, predict, player, cache


def build_parser() -> argparse.ArgumentParser:
    """
    Build the main argument parser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="pitchpredict",
        description="Predict MLB pitcher/batter behavior and outcomes",
        epilog="https://github.com/baseball-analytica/pitchpredict",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Top-level commands
    serve.add_parser(subparsers)
    version.add_parser(subparsers)

    # Nested command groups
    predict.add_parser(subparsers)
    player.add_parser(subparsers)
    cache.add_parser(subparsers)

    return parser


def run_cli():
    """
    Run the PitchPredict CLI.
    """
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    run_cli()
