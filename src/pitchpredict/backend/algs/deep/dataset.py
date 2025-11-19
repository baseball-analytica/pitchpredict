from typing import NamedTuple, cast
from torch.utils.data import Dataset
from collections import namedtuple
import numpy as np
import os
import torch
from pitchpredict.backend.algs.deep.nn import _CONTEXT_FIELD_SPECS, _context_field_path, TOKEN_DTYPE

class PackedPitchChunk(NamedTuple): # keep parallel with nn._CONTEXT_FIELD_SPECS
    x: torch.LongTensor
    y: torch.LongTensor
    pitcher_id: torch.LongTensor
    batter_id: torch.LongTensor
    pitcher_age: torch.LongTensor
    pitcher_throws: torch.LongTensor
    batter_age: torch.LongTensor
    batter_hits: torch.LongTensor
    count_balls: torch.LongTensor
    count_strikes: torch.LongTensor
    outs: torch.LongTensor
    bases_state: torch.LongTensor
    score_bat: torch.LongTensor
    score_fld: torch.LongTensor
    inning: torch.LongTensor
    pitch_number: torch.LongTensor
    number_through_order: torch.LongTensor
    game_date: torch.LongTensor


class PackedPitchDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int, tokens_file: str = "pitch_seq.bin", context_prefix: str = "pitch_context"):
        self.contexts = {}
        self.tokens = np.memmap(os.path.join(data_dir, tokens_file), dtype=TOKEN_DTYPE, mode="r")
        self.L = int(seq_len)
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
        self.offset = 0
        self.num_chunks = max(0, (self.tokens.size -self.offset - 1) // self.L)

    def set_offset(self, offset: int) -> None:
        self.offset = int(offset) % self.L
        self.num_chunks = max(0, (self.tokens.size - self.offset - 1) // self.L)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, index: int) -> PackedPitchChunk:
        start = self.offset + index * self.L
        end = start + self.L + 1
        chunk_tok = torch.from_numpy(self.tokens[start:end].astype(TOKEN_DTYPE))
        x = chunk_tok[:-1]
        y = chunk_tok[1:]
        context_fields = {field_name: cast(torch.LongTensor, torch.from_numpy(self.contexts[field_name][start:end-1].astype(spec.dtype))) for field_name, spec in _CONTEXT_FIELD_SPECS.items()}
        return PackedPitchChunk(
            x=cast(torch.LongTensor, x),
            y=cast(torch.LongTensor, y),
            **context_fields,
        )