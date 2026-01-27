# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

"""
PitchPredict CLI entry point.

This module delegates to the cli package for the full CLI implementation.
It exists for backwards compatibility.
"""

from pitchpredict.cli import run_cli

if __name__ == "__main__":
    run_cli()
