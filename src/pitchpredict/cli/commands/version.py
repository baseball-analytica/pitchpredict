# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """
    Add the 'version' subcommand to the CLI.
    """
    parser = subparsers.add_parser(
        "version",
        help="Show PitchPredict version information",
        description="Display the current version of PitchPredict.",
    )
    parser.set_defaults(func=handle_version)


def handle_version(args: argparse.Namespace) -> None:
    """
    Handle the 'version' command.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    try:
        from pitchpredict.utils.version import get_version

        version = get_version()
    except Exception:
        # Fallback if package isn't installed
        version = "unknown"

    panel = Panel(
        f"[bold cyan]PitchPredict[/bold cyan] version [green]{version}[/green]\n\n"
        "[dim]Predict MLB pitcher/batter behavior and outcomes[/dim]\n"
        "[dim]https://github.com/baseball-analytica/pitchpredict[/dim]",
        title="[bold]PitchPredict[/bold]",
        border_style="blue",
    )
    console.print(panel)
