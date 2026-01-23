from typing import NamedTuple, cast
from torch.utils.data import Dataset
from collections import namedtuple
import numpy as np
import os
import torch
from pitchpredict.backend.algs.deep.nn import (
    _CONTEXT_FIELD_SPECS,
    _context_field_path,
    TOKEN_DTYPE,
)


class PackedPitchChunk(NamedTuple):  # keep parallel with nn._CONTEXT_FIELD_SPECS
    x: torch.LongTensor
    y: torch.LongTensor
    pitcher_id: torch.IntTensor
    batter_id: torch.IntTensor
    pitcher_age: torch.FloatTensor
    pitcher_throws: torch.IntTensor
    batter_age: torch.FloatTensor
    batter_hits: torch.IntTensor
    count_balls: torch.IntTensor
    count_strikes: torch.IntTensor
    outs: torch.LongTensor
    bases_state: torch.IntTensor
    score_bat: torch.FloatTensor
    score_fld: torch.FloatTensor
    inning: torch.IntTensor
    pitch_number: torch.FloatTensor
    number_through_order: torch.IntTensor
    game_date: torch.FloatTensor
    fielder_2_id: torch.IntTensor
    fielder_3_id: torch.IntTensor
    fielder_4_id: torch.IntTensor
    fielder_5_id: torch.IntTensor
    fielder_6_id: torch.IntTensor
    fielder_7_id: torch.IntTensor
    fielder_8_id: torch.IntTensor
    fielder_9_id: torch.IntTensor
    batter_days_since_prev_game: torch.IntTensor
    pitcher_days_since_prev_game: torch.IntTensor
    strike_zone_top: torch.FloatTensor
    strike_zone_bottom: torch.FloatTensor


class PackedPitchContext(NamedTuple):
    pitcher_id: torch.LongTensor
    batter_id: torch.LongTensor
    pitcher_age: torch.FloatTensor
    pitcher_throws: torch.IntTensor
    batter_age: torch.FloatTensor
    batter_hits: torch.IntTensor
    count_balls: torch.IntTensor
    count_strikes: torch.IntTensor
    outs: torch.LongTensor
    bases_state: torch.IntTensor
    score_bat: torch.FloatTensor
    score_fld: torch.FloatTensor
    inning: torch.IntTensor
    pitch_number: torch.FloatTensor
    number_through_order: torch.IntTensor
    game_date: torch.FloatTensor
    fielder_2_id: torch.IntTensor
    fielder_3_id: torch.IntTensor
    fielder_4_id: torch.IntTensor
    fielder_5_id: torch.IntTensor
    fielder_6_id: torch.IntTensor
    fielder_7_id: torch.IntTensor
    fielder_8_id: torch.IntTensor
    fielder_9_id: torch.IntTensor
    batter_days_since_prev_game: torch.IntTensor
    pitcher_days_since_prev_game: torch.IntTensor
    strike_zone_top: torch.FloatTensor
    strike_zone_bottom: torch.FloatTensor


def chunk_to_context(
    chunk: PackedPitchChunk, device: torch.device
) -> PackedPitchContext:
    """
    Moves chunk data to device and enforces strict type casting:
    - IDs -> torch.long
    - Stats -> torch.float
    """
    return PackedPitchContext(
        # Categorical / IDs (Must be Long for Embeddings)
        pitcher_id=chunk.pitcher_id.to(device, dtype=torch.long, non_blocking=True),  # type: ignore
        batter_id=chunk.batter_id.to(device, dtype=torch.long, non_blocking=True),  # type: ignore
        pitcher_throws=chunk.pitcher_throws.to(
            device, dtype=torch.int, non_blocking=True
        ),  # type: ignore
        batter_hits=chunk.batter_hits.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        count_balls=chunk.count_balls.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        count_strikes=chunk.count_strikes.to(
            device, dtype=torch.int, non_blocking=True
        ),  # type: ignore
        outs=chunk.outs.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        bases_state=chunk.bases_state.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        inning=chunk.inning.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        number_through_order=chunk.number_through_order.to(
            device, dtype=torch.int, non_blocking=True
        ),  # type: ignore
        fielder_2_id=chunk.fielder_2_id.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        fielder_3_id=chunk.fielder_3_id.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        fielder_4_id=chunk.fielder_4_id.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        fielder_5_id=chunk.fielder_5_id.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        fielder_6_id=chunk.fielder_6_id.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        fielder_7_id=chunk.fielder_7_id.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        fielder_8_id=chunk.fielder_8_id.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        fielder_9_id=chunk.fielder_9_id.to(device, dtype=torch.int, non_blocking=True),  # type: ignore
        batter_days_since_prev_game=chunk.batter_days_since_prev_game.to(
            device, dtype=torch.int, non_blocking=True
        ),  # type: ignore
        pitcher_days_since_prev_game=chunk.pitcher_days_since_prev_game.to(
            device, dtype=torch.int, non_blocking=True
        ),  # type: ignore
        # Continuous / Stats (Must be Float for Linear layers)
        pitcher_age=chunk.pitcher_age.to(device, dtype=torch.float, non_blocking=True),  # type: ignore
        batter_age=chunk.batter_age.to(device, dtype=torch.float, non_blocking=True),  # type: ignore
        score_bat=chunk.score_bat.to(device, dtype=torch.float, non_blocking=True),  # type: ignore
        score_fld=chunk.score_fld.to(device, dtype=torch.float, non_blocking=True),  # type: ignore
        pitch_number=chunk.pitch_number.to(
            device, dtype=torch.float, non_blocking=True
        ),  # type: ignore
        game_date=chunk.game_date.to(device, dtype=torch.float, non_blocking=True),  # type: ignore
        strike_zone_top=chunk.strike_zone_top.to(
            device, dtype=torch.float, non_blocking=True
        ),  # type: ignore
        strike_zone_bottom=chunk.strike_zone_bottom.to(
            device, dtype=torch.float, non_blocking=True
        ),  # type: ignore
    )


class PackedPitchDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        tokens_file: str = "pitch_seq.bin",
        context_prefix: str = "pitch_context",
    ):
        self.contexts = {}
        self.tokens = np.memmap(
            os.path.join(data_dir, tokens_file), dtype=TOKEN_DTYPE, mode="r"
        )
        self.L = int(seq_len)
        context_prefix = os.path.join(data_dir, context_prefix)
        for field_name, spec in _CONTEXT_FIELD_SPECS.items():
            path = _context_field_path(context_prefix, field_name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"context field file not found: {path}")
            file_size = os.path.getsize(path)
            expected_size = self.tokens.size * spec.dtype.itemsize
            if file_size != expected_size:
                raise ValueError(
                    f"{path} has {file_size} bytes but expected {expected_size} for {self.tokens.size} samples"
                )
            self.contexts[field_name] = np.memmap(path, dtype=spec.dtype, mode="r")
        self.total_tokens = int(self.tokens.size)
        self.cycle = max(1, self.total_tokens - self.L)
        self.offset = 0
        self.num_chunks = max(0, (self.total_tokens - 1) // self.L)

    def set_offset(self, offset: int) -> None:
        if self.L <= 0:
            self.offset = 0
            return
        self.offset = int(offset) % self.L

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, index: int) -> PackedPitchChunk:
        if index < 0 or index >= self.num_chunks:
            raise IndexError(
                f"index {index} out of range for PackedPitchDataset with {self.num_chunks} chunks"
            )
        start = (self.offset + index * self.L) % self.cycle
        end = start + self.L + 1
        chunk_tok = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        x = chunk_tok[:-1]
        y = chunk_tok[1:]
        return PackedPitchChunk(
            x=x,  # type: ignore
            y=y,  # type: ignore
            pitcher_id=torch.from_numpy(
                self.contexts["pitcher_id"][start : end - 1].astype(np.int64)
            ),  # type: ignore
            batter_id=torch.from_numpy(
                self.contexts["batter_id"][start : end - 1].astype(np.int64)
            ),  # type: ignore
            pitcher_age=torch.from_numpy(
                self.contexts["pitcher_age"][start : end - 1].astype(np.float32)
            ),  # type: ignore
            pitcher_throws=torch.from_numpy(
                self.contexts["pitcher_throws"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            batter_age=torch.from_numpy(
                self.contexts["batter_age"][start : end - 1].astype(np.float32)
            ),  # type: ignore
            batter_hits=torch.from_numpy(
                self.contexts["batter_hits"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            count_balls=torch.from_numpy(
                self.contexts["count_balls"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            count_strikes=torch.from_numpy(
                self.contexts["count_strikes"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            outs=torch.from_numpy(
                self.contexts["outs"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            bases_state=torch.from_numpy(
                self.contexts["bases_state"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            score_bat=torch.from_numpy(
                self.contexts["score_bat"][start : end - 1].astype(np.float32)
            ),  # type: ignore
            score_fld=torch.from_numpy(
                self.contexts["score_fld"][start : end - 1].astype(np.float32)
            ),  # type: ignore
            inning=torch.from_numpy(
                self.contexts["inning"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            pitch_number=torch.from_numpy(
                self.contexts["pitch_number"][start : end - 1].astype(np.float32)
            ),  # type: ignore
            number_through_order=torch.from_numpy(
                self.contexts["number_through_order"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            game_date=torch.from_numpy(
                self.contexts["game_date"][start : end - 1].astype(np.float32)
            ),  # type: ignore
            fielder_2_id=torch.from_numpy(
                self.contexts["fielder_2_id"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            fielder_3_id=torch.from_numpy(
                self.contexts["fielder_3_id"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            fielder_4_id=torch.from_numpy(
                self.contexts["fielder_4_id"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            fielder_5_id=torch.from_numpy(
                self.contexts["fielder_5_id"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            fielder_6_id=torch.from_numpy(
                self.contexts["fielder_6_id"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            fielder_7_id=torch.from_numpy(
                self.contexts["fielder_7_id"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            fielder_8_id=torch.from_numpy(
                self.contexts["fielder_8_id"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            fielder_9_id=torch.from_numpy(
                self.contexts["fielder_9_id"][start : end - 1].astype(np.int32)
            ),  # type: ignore
            batter_days_since_prev_game=torch.from_numpy(
                self.contexts["batter_days_since_prev_game"][start : end - 1].astype(
                    np.int32
                )
            ),  # type: ignore
            pitcher_days_since_prev_game=torch.from_numpy(
                self.contexts["pitcher_days_since_prev_game"][start : end - 1].astype(
                    np.int32
                )
            ),  # type: ignore
            strike_zone_top=torch.from_numpy(
                self.contexts["strike_zone_top"][start : end - 1].astype(np.float32)
            ),  # type: ignore
            strike_zone_bottom=torch.from_numpy(
                self.contexts["strike_zone_bottom"][start : end - 1].astype(np.float32)
            ),  # type: ignore
        )
