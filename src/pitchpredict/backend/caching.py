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

    Stores Parquet datasets and small JSON metadata files for quick coverage checks.
    """

    def __init__(
        self,
        cache_dir: str = ".pitchpredict_cache",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.logger = logging.getLogger("pitchpredict.backend.caching")
        self._pitcher_dir = self.cache_dir / "pitcher"
        self._batter_dir = self.cache_dir / "batter"
        self._batted_ball_dir = self.cache_dir / "batted_ball"
        self._batted_ball_path = self._batted_ball_dir / "batted_balls.parquet"
        self._batted_ball_meta_path = self._batted_ball_dir / "batted_balls.meta.json"
        self._player_dir = self.cache_dir / "players"
        self._player_index_path = self._player_dir / "name_to_id.json"
        self._player_records_path = self._player_dir / "name_to_records.json"
        self._player_id_records_path = self._player_dir / "id_to_record.json"
        self._player_index: dict[str, int] | None = None
        self._player_records_index: dict[str, list[dict[str, object]]] | None = None
        self._player_id_records: dict[str, dict[str, object]] | None = None
        self.__post_init__()

    def __post_init__(self) -> None:
        """
        Perform post-initialization tasks, including validation.
        """
        self.logger.debug("running post-initialization tasks for cache")

        for path in (self.cache_dir, self._pitcher_dir, self._batter_dir, self._batted_ball_dir, self._player_dir):
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info("created cache directory at %s", path)
            else:
                self.logger.debug("cache directory already exists at %s", path)

        self.logger.debug("cache initialized")

    def _normalize_end_date(self, end_date: str | None) -> str:
        """Normalize end dates to YYYY-MM-DD with a safe fallback."""
        if end_date is None:
            return datetime.now().strftime("%Y-%m-%d")
        try:
            return pd.Timestamp(end_date).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            self.logger.warning("invalid end_date %s; using current date", end_date)
            return datetime.now().strftime("%Y-%m-%d")

    def _normalize_player_key(self, player_name: str, fuzzy_lookup: bool) -> str:
        """Normalize player names to a stable cache key."""
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
        """Ensure a parsed game_date_dt column exists for filtering."""
        if "game_date_dt" in pitches.columns or "game_date" not in pitches.columns:
            return pitches
        pitches = pitches.copy(deep=False)
        pitches["game_date_dt"] = pd.to_datetime(pitches["game_date"], errors="coerce")
        return pitches

    def _filter_pitches_by_date(self, pitches: pd.DataFrame, end_date: str) -> pd.DataFrame:
        """Filter pitches with game_date_dt <= end_date."""
        if "game_date_dt" not in pitches.columns:
            pitches = self._ensure_game_date_dt(pitches)
        if "game_date_dt" not in pitches.columns:
            return pitches.copy(deep=False)
        end_ts = pd.Timestamp(end_date)
        mask = pitches["game_date_dt"] <= end_ts
        return pitches.loc[mask].copy(deep=False)

    def _filter_pitches_by_range(self, pitches: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter pitches with start_date <= game_date_dt <= end_date."""
        if "game_date_dt" not in pitches.columns:
            pitches = self._ensure_game_date_dt(pitches)
        if "game_date_dt" not in pitches.columns:
            return pitches.copy(deep=False)
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        mask = (pitches["game_date_dt"] >= start_ts) & (pitches["game_date_dt"] <= end_ts)
        return pitches.loc[mask].copy(deep=False)

    def _read_dataframe(self, path: Path) -> pd.DataFrame | None:
        """Read a cached Parquet file if it exists."""
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            self.logger.warning(f"failed to read cache file {path}: {exc}")
            return None

    def _write_dataframe(self, path: Path, data: pd.DataFrame) -> None:
        """Write a Parquet file atomically to avoid partial writes."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.parent / f"{path.name}.tmp"
            data.to_parquet(tmp_path)
            os.replace(tmp_path, path)
        except Exception as exc:
            self.logger.warning(f"failed to write cache file {path}: {exc}")

    def _read_end_date_meta(self, path: Path) -> pd.Timestamp | None:
        """Read end_date metadata used to validate cache coverage."""
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
        """Write end_date metadata atomically."""
        try:
            tmp_path = path.parent / f"{path.name}.tmp"
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump({"end_date": end_date}, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        except Exception as exc:
            self.logger.warning(f"failed to write cache metadata {path}: {exc}")

    def _read_range_meta(self, path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Read cached start/end coverage range metadata."""
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            start_date = payload.get("start_date")
            end_date = payload.get("end_date")
            if start_date is None or end_date is None:
                return None
            return pd.Timestamp(start_date), pd.Timestamp(end_date)
        except Exception as exc:
            self.logger.warning(f"failed to read cache metadata {path}: {exc}")
            return None

    def _write_range_meta(self, path: Path, start_date: str, end_date: str) -> None:
        """Write cached start/end coverage range metadata."""
        try:
            tmp_path = path.parent / f"{path.name}.tmp"
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump({"start_date": start_date, "end_date": end_date}, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        except Exception as exc:
            self.logger.warning(f"failed to write cache metadata {path}: {exc}")

    def get_pitcher_pitches(self, pitcher_id: int, end_date: str | None) -> pd.DataFrame | None:
        """Return cached pitcher pitches if coverage includes end_date."""
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

    def get_pitcher_cache_state(self, pitcher_id: int) -> tuple[pd.DataFrame, str] | None:
        """Return cached pitcher data with its coverage end date."""
        path = self._pitcher_path(pitcher_id)
        data = self._read_dataframe(path)
        if data is None:
            return None
        meta_path = self._pitcher_meta_path(pitcher_id)
        cached_end_date = self._read_end_date_meta(meta_path)
        if cached_end_date is None:
            data = self._ensure_game_date_dt(data)
            if "game_date_dt" not in data.columns:
                return None
            max_date = data["game_date_dt"].max()
            if pd.isna(max_date):
                return None
            cached_end_date = pd.Timestamp(max_date).normalize()
        return data, cached_end_date.strftime("%Y-%m-%d")

    def set_pitcher_pitches(self, pitcher_id: int, end_date: str | None, pitches: pd.DataFrame) -> None:
        """Persist pitcher pitches if they extend current coverage."""
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
        """Return cached batter pitches if coverage includes end_date."""
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

    def get_batter_cache_state(self, batter_id: int) -> tuple[pd.DataFrame, str] | None:
        """Return cached batter data with its coverage end date."""
        path = self._batter_path(batter_id)
        data = self._read_dataframe(path)
        if data is None:
            return None
        meta_path = self._batter_meta_path(batter_id)
        cached_end_date = self._read_end_date_meta(meta_path)
        if cached_end_date is None:
            data = self._ensure_game_date_dt(data)
            if "game_date_dt" not in data.columns:
                return None
            max_date = data["game_date_dt"].max()
            if pd.isna(max_date):
                return None
            cached_end_date = pd.Timestamp(max_date).normalize()
        return data, cached_end_date.strftime("%Y-%m-%d")

    def set_batter_pitches(self, batter_id: int, end_date: str | None, pitches: pd.DataFrame) -> None:
        """Persist batter pitches if they extend current coverage."""
        normalized_end_date = self._normalize_end_date(end_date)
        meta_path = self._batter_meta_path(batter_id)
        cached_end_date = self._read_end_date_meta(meta_path)
        if cached_end_date is not None and pd.Timestamp(normalized_end_date) <= cached_end_date:
            return
        path = self._batter_path(batter_id)
        self._write_dataframe(path, pitches)
        self._write_end_date_meta(meta_path, normalized_end_date)
        self.logger.debug(f"cached batter pitches at {path}")

    def get_batted_balls(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        """Return cached batted balls if the date range is covered."""
        try:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
        except (TypeError, ValueError):
            self.logger.warning(f"invalid date range for batted balls cache: {start_date} - {end_date}")
            return None

        cached_range = self._read_range_meta(self._batted_ball_meta_path)
        if cached_range is not None:
            cached_start, cached_end = cached_range
            if start_ts >= cached_start and end_ts <= cached_end:
                data = self._read_dataframe(self._batted_ball_path)
                if data is not None:
                    self.logger.debug(f"cache hit for batted balls at {self._batted_ball_path}")
                    return self._filter_pitches_by_range(data, start_date, end_date)
                return None

        data = self._read_dataframe(self._batted_ball_path)
        if data is None:
            return None
        data = self._ensure_game_date_dt(data)
        if "game_date_dt" not in data.columns:
            return None
        cached_start = data["game_date_dt"].min()
        cached_end = data["game_date_dt"].max()
        if pd.notna(cached_start) and pd.notna(cached_end):
            if start_ts >= cached_start and end_ts <= cached_end:
                self.logger.debug(f"cache hit for batted balls at {self._batted_ball_path}")
                return self._filter_pitches_by_range(data, start_date, end_date)
        return None

    def set_batted_balls(self, start_date: str, end_date: str, batted_balls: pd.DataFrame) -> None:
        """Persist batted balls if they extend the cached range."""
        try:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
        except (TypeError, ValueError):
            self.logger.warning(f"invalid date range for batted balls cache: {start_date} - {end_date}")
            return

        cached_range = self._read_range_meta(self._batted_ball_meta_path)
        if cached_range is not None:
            cached_start, cached_end = cached_range
            if start_ts >= cached_start and end_ts <= cached_end:
                return

        self._write_dataframe(self._batted_ball_path, batted_balls)
        self._write_range_meta(self._batted_ball_meta_path, start_date, end_date)
        self.logger.debug(f"cached batted balls at {self._batted_ball_path}")

    def get_batted_balls_cache_state(self) -> tuple[pd.DataFrame, str, str] | None:
        """Return cached batted balls with coverage start/end."""
        data = self._read_dataframe(self._batted_ball_path)
        if data is None:
            return None
        cached_range = self._read_range_meta(self._batted_ball_meta_path)
        if cached_range is None:
            data = self._ensure_game_date_dt(data)
            if "game_date_dt" not in data.columns:
                return None
            cached_start = data["game_date_dt"].min()
            cached_end = data["game_date_dt"].max()
            if pd.isna(cached_start) or pd.isna(cached_end):
                return None
            cached_range = (pd.Timestamp(cached_start), pd.Timestamp(cached_end))
        cached_start, cached_end = cached_range
        return data, cached_start.strftime("%Y-%m-%d"), cached_end.strftime("%Y-%m-%d")

    def _load_player_index(self) -> dict[str, int]:
        """Load the player name -> ID map from disk."""
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
        """Persist the player name -> ID map atomically."""
        tmp_path = self._player_index_path.parent / f"{self._player_index_path.name}.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self._player_index_path)

    def _load_player_records_index(self) -> dict[str, list[dict[str, object]]]:
        if self._player_records_index is not None:
            return self._player_records_index
        if not self._player_records_path.exists():
            self._player_records_index = {}
            return self._player_records_index
        try:
            with self._player_records_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                cleaned: dict[str, list[dict[str, object]]] = {}
                for key, value in data.items():
                    if isinstance(value, list):
                        cleaned[key] = [record for record in value if isinstance(record, dict)]
                self._player_records_index = cleaned
            else:
                self._player_records_index = {}
        except Exception as exc:
            self.logger.warning(f"failed to read player records cache: {exc}")
            self._player_records_index = {}
        return self._player_records_index

    def _write_player_records_index(self, data: dict[str, list[dict[str, object]]]) -> None:
        tmp_path = self._player_records_path.parent / f"{self._player_records_path.name}.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self._player_records_path)

    def _load_player_id_records(self) -> dict[str, dict[str, object]]:
        if self._player_id_records is not None:
            return self._player_id_records
        if not self._player_id_records_path.exists():
            self._player_id_records = {}
            return self._player_id_records
        try:
            with self._player_id_records_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                cleaned = {
                    str(key): value for key, value in data.items() if isinstance(value, dict)
                }
                self._player_id_records = cleaned
            else:
                self._player_id_records = {}
        except Exception as exc:
            self.logger.warning(f"failed to read player id cache: {exc}")
            self._player_id_records = {}
        return self._player_id_records

    def _write_player_id_records(self, data: dict[str, dict[str, object]]) -> None:
        tmp_path = self._player_id_records_path.parent / f"{self._player_id_records_path.name}.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self._player_id_records_path)

    def _extract_mlbam_id(self, record: dict[str, object]) -> int | None:
        for key in ("key_mlbam", "mlbam_id"):
            value = record.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    def get_player_id(self, player_name: str, fuzzy_lookup: bool = True) -> int | None:
        """Lookup a cached player ID by normalized name."""
        key = self._normalize_player_key(player_name, fuzzy_lookup)
        data = self._load_player_index()
        player_id = data.get(key)
        if player_id is None:
            records = self._load_player_records_index().get(key)
            if records:
                player_id = self._extract_mlbam_id(records[0])
        if player_id is not None:
            self.logger.debug(f"cache hit for player {player_name}")
        return player_id

    def set_player_id(self, player_name: str, player_id: int, fuzzy_lookup: bool = True) -> None:
        """Store a player ID for future lookups."""
        key = self._normalize_player_key(player_name, fuzzy_lookup)
        data = self._load_player_index()
        data[key] = int(player_id)
        self._write_player_index(data)
        self._player_index = data
        self.logger.debug(f"cached player id for {player_name}")

    def get_player_records(self, player_name: str, fuzzy_lookup: bool = True) -> list[dict[str, object]] | None:
        """Lookup cached player record list by name."""
        key = self._normalize_player_key(player_name, fuzzy_lookup)
        data = self._load_player_records_index()
        records = data.get(key)
        if records is not None:
            self.logger.debug(f"cache hit for player records {player_name}")
        return records

    def set_player_records(
        self,
        player_name: str,
        fuzzy_lookup: bool,
        records: list[dict[str, object]],
    ) -> None:
        """Store a list of player records for a name lookup."""
        key = self._normalize_player_key(player_name, fuzzy_lookup)
        data = self._load_player_records_index()
        data[key] = records
        self._write_player_records_index(data)
        self._player_records_index = data

        if records:
            mlbam_id = self._extract_mlbam_id(records[0])
            if mlbam_id is not None:
                self.set_player_id(player_name=player_name, player_id=mlbam_id, fuzzy_lookup=fuzzy_lookup)

        id_records = self._load_player_id_records()
        for record in records:
            mlbam_id = self._extract_mlbam_id(record)
            if mlbam_id is not None:
                id_records[str(mlbam_id)] = record
        if records:
            self._write_player_id_records(id_records)
            self._player_id_records = id_records
        self.logger.debug(f"cached player records for {player_name}")

    def get_player_record_by_id(self, mlbam_id: int) -> dict[str, object] | None:
        """Lookup cached player record by MLBAM ID."""
        data = self._load_player_id_records()
        record = data.get(str(mlbam_id))
        if record is not None:
            self.logger.debug(f"cache hit for player id {mlbam_id}")
        return record

    def set_player_record_by_id(self, mlbam_id: int, record: dict[str, object]) -> None:
        """Store a player record for reverse ID lookup."""
        data = self._load_player_id_records()
        data[str(mlbam_id)] = record
        self._write_player_id_records(data)
        self._player_id_records = data
        self.logger.debug(f"cached player record for id {mlbam_id}")
