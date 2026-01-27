# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from pitchpredict.api import PitchPredict

console = Console()
error_console = Console(stderr=True)

_api: "PitchPredict | None" = None


def get_api() -> "PitchPredict":
    """
    Get or create shared PitchPredict instance.

    This creates a single instance that can be reused across CLI commands
    to avoid repeated initialization.
    """
    global _api
    if _api is None:
        from pitchpredict.api import PitchPredict

        _api = PitchPredict(
            enable_logging=False,
            log_level_console="WARNING",
        )
    return _api


def run_async(coro):
    """
    Run an async coroutine synchronously.

    This allows CLI commands to call async API methods.
    """
    return asyncio.run(coro)


async def resolve_player(ref: str, api: "PitchPredict") -> int:
    """
    Resolve a player name or ID to an MLBAM ID.

    If the reference is a valid integer, it's returned as-is.
    Otherwise, it's treated as a player name and resolved via the API.

    Args:
        ref: Player name or MLBAM ID as a string
        api: PitchPredict API instance

    Returns:
        The MLBAM ID as an integer

    Raises:
        HTTPException: If the player cannot be found
    """
    try:
        return int(ref)
    except ValueError:
        return await api.get_player_id_from_name(ref)
