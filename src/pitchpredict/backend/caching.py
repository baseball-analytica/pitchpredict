# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Addison Kline

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path

import pandas as pd


class PitchPredictCache:
    """
    A local cache for PitchPredict data.
    """

    def __init__(
        self,
        cache_dir: str = ".pitchpredict_cache",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.logger = logging.getLogger("pitchpredict.backend.caching")
        self._pitcher_dir = self.cache_dir / "pitcher"
        self._batter_dir = self.cache_dir / "batter"
        self._player_dir = self.cache_dir / "players"
        self._player_index_path = self._player_dir / "name_to_id.json"
        self._player_index: dict[str, int] | None = None
        self.__post_init__()

    def __post_init__(self) -> None:
        """
        Perform post-initialization tasks, including validation.
        """
        self.logger.debug("running post-initialization tasks for cache")

        for path in (self.cache_dir, self._pitcher_dir, self._batter_dir, self._player_dir):
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info("created cache directory at %s", path)
            else:
                self.logger.debug("cache directory already exists at %s", path)

        self.logger.debug("cache initialized")

    def _normalize_end_date(self, end_date: str | None) -> str:
        if end_date is None:
            return datetime.now().strftime("%Y-%m-%d")
        try:
            return pd.Timestamp(end_date).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            self.logger.warning("invalid end_date %s; using current date", end_date)
            return datetime.now().strftime("%Y-%m-%d")

    def _normalize_player_key(self, player_name: str, fuzzy_lookup: bool) -> str:
        normalized = " ".join(player_name.strip().lower().split())
        flag = "fuzzy" if fuzzy_lookup else "exact"
        return f"{normalized}|{flag}"

    def _pitcher_path(self, pitcher_id: int) -> Path:
        return self._pitcher_dir / f"{pitcher_id}.parquet"

    def _pitcher_meta_path(self, pitcher_id: int) -> Path:
        return self._pitcher_dir / f"{pitcher_id}.meta.json"

    def _batter_path(self, batter_id: int) -> Path:
        return self._batter_dir / f"{batter_id}.parquet"

    def _batter_meta_path(self, batter_id: int) -> Path:
        return self._batter_dir / f"{batter_id}.meta.json"

    def _ensure_game_date_dt(self, pitches: pd.DataFrame) -> pd.DataFrame:
        if "game_date_dt" in pitches.columns or "game_date" not in pitches.columns:
            return pitches
        pitches = pitches.copy(deep=False)
        pitches["game_date_dt"] = pd.to_datetime(pitches["game_date"], errors="coerce")
        return pitches

    def _filter_pitches_by_date(self, pitches: pd.DataFrame, end_date: str) -> pd.DataFrame:
        if "game_date_dt" not in pitches.columns:
            pitches = self._ensure_game_date_dt(pitches)
        if "game_date_dt" not in pitches.columns:
            return pitches.copy(deep=False)
        end_ts = pd.Timestamp(end_date)
        mask = pitches["game_date_dt"] <= end_ts
        return pitches.loc[mask].copy(deep=False)

    def _read_dataframe(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            self.logger.warning(f"failed to read cache file {path}: {exc}")
            return None

    def _write_dataframe(self, path: Path, data: pd.DataFrame) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.parent / f"{path.name}.tmp"
            data.to_parquet(tmp_path)
            os.replace(tmp_path, path)
        except Exception as exc:
            self.logger.warning(f"failed to write cache file {path}: {exc}")

    def _read_end_date_meta(self, path: Path) -> pd.Timestamp | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            end_date = payload.get("end_date")
            if end_date is None:
                return None
            return pd.Timestamp(end_date)
        except Exception as exc:
            self.logger.warning(f"failed to read cache metadata {path}: {exc}")
            return None

    def _write_end_date_meta(self, path: Path, end_date: str) -> None:
        try:
            tmp_path = path.parent / f"{path.name}.tmp"
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump({"end_date": end_date}, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        except Exception as exc:
            self.logger.warning(f"failed to write cache metadata {path}: {exc}")

    def get_pitcher_pitches(self, pitcher_id: int, end_date: str | None) -> pd.DataFrame | None:
        normalized_end_date = self._normalize_end_date(end_date)
        path = self._pitcher_path(pitcher_id)
        meta_path = self._pitcher_meta_path(pitcher_id)
        cached_end_date = self._read_end_date_meta(meta_path)
        if cached_end_date is not None:
            if pd.Timestamp(normalized_end_date) <= cached_end_date:
                data = self._read_dataframe(path)
                if data is not None:
                    self.logger.debug(f"cache hit for pitcher {pitcher_id} at {path}")
                    return self._filter_pitches_by_date(data, normalized_end_date)
            return None
        data = self._read_dataframe(path)
        if data is None:
            return None
        data = self._ensure_game_date_dt(data)
        if "game_date_dt" in data.columns:
            max_date = data["game_date_dt"].max()
            if pd.notna(max_date) and pd.Timestamp(normalized_end_date) <= pd.Timestamp(max_date).normalize():
                self.logger.debug(f"cache hit for pitcher {pitcher_id} at {path}")
                return self._filter_pitches_by_date(data, normalized_end_date)
        return None

    def set_pitcher_pitches(self, pitcher_id: int, end_date: str | None, pitches: pd.DataFrame) -> None:
        normalized_end_date = self._normalize_end_date(end_date)
        meta_path = self._pitcher_meta_path(pitcher_id)
        cached_end_date = self._read_end_date_meta(meta_path)
        if cached_end_date is not None and pd.Timestamp(normalized_end_date) <= cached_end_date:
            return
        path = self._pitcher_path(pitcher_id)
        self._write_dataframe(path, pitches)
        self._write_end_date_meta(meta_path, normalized_end_date)
        self.logger.debug(f"cached pitcher pitches at {path}")

    def get_batter_pitches(self, batter_id: int, end_date: str | None) -> pd.DataFrame | None:
        normalized_end_date = self._normalize_end_date(end_date)
        path = self._batter_path(batter_id)
        meta_path = self._batter_meta_path(batter_id)
        cached_end_date = self._read_end_date_meta(meta_path)
        if cached_end_date is not None:
            if pd.Timestamp(normalized_end_date) <= cached_end_date:
                data = self._read_dataframe(path)
                if data is not None:
                    self.logger.debug(f"cache hit for batter {batter_id} at {path}")
                    return self._filter_pitches_by_date(data, normalized_end_date)
            return None
        data = self._read_dataframe(path)
        if data is None:
            return None
        data = self._ensure_game_date_dt(data)
        if "game_date_dt" in data.columns:
            max_date = data["game_date_dt"].max()
            if pd.notna(max_date) and pd.Timestamp(normalized_end_date) <= pd.Timestamp(max_date).normalize():
                self.logger.debug(f"cache hit for batter {batter_id} at {path}")
                return self._filter_pitches_by_date(data, normalized_end_date)
        return None

    def set_batter_pitches(self, batter_id: int, end_date: str | None, pitches: pd.DataFrame) -> None:
        normalized_end_date = self._normalize_end_date(end_date)
        meta_path = self._batter_meta_path(batter_id)
        cached_end_date = self._read_end_date_meta(meta_path)
        if cached_end_date is not None and pd.Timestamp(normalized_end_date) <= cached_end_date:
            return
        path = self._batter_path(batter_id)
        self._write_dataframe(path, pitches)
        self._write_end_date_meta(meta_path, normalized_end_date)
        self.logger.debug(f"cached batter pitches at {path}")

    def _load_player_index(self) -> dict[str, int]:
        if self._player_index is not None:
            return self._player_index
        if not self._player_index_path.exists():
            self._player_index = {}
            return self._player_index
        try:
            with self._player_index_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                self._player_index = {str(key): int(value) for key, value in data.items()}
            else:
                self._player_index = {}
        except Exception as exc:
            self.logger.warning(f"failed to read player cache index: {exc}")
            self._player_index = {}
        return self._player_index

    def _write_player_index(self, data: dict[str, int]) -> None:
        tmp_path = self._player_index_path.parent / f"{self._player_index_path.name}.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self._player_index_path)

    def get_player_id(self, player_name: str, fuzzy_lookup: bool = True) -> int | None:
        key = self._normalize_player_key(player_name, fuzzy_lookup)
        data = self._load_player_index()
        player_id = data.get(key)
        if player_id is not None:
            self.logger.debug(f"cache hit for player {player_name}")
        return player_id

    def set_player_id(self, player_name: str, player_id: int, fuzzy_lookup: bool = True) -> None:
        key = self._normalize_player_key(player_name, fuzzy_lookup)
        data = self._load_player_index()
        data[key] = int(player_id)
        self._write_player_index(data)
        self._player_index = data
        self.logger.debug(f"cached player id for {player_name}")
