# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from importlib.metadata import version

def get_version() -> str:
    return version("pitchpredict")