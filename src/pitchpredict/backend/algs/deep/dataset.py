from typing import NamedTuple, cast
from torch.utils.data import Dataset
from collections import namedtuple
import numpy as np
import os
import torch
from pitchpredict.backend.algs.deep.nn import _CONTEXT_FIELD_SPECS, _context_field_path, TOKEN_DTYPE
from pitchpredict.backend.algs.deep.types import PitchToken

# Token value for SESSION_START (used for session-boundary-aligned sampling)
SESSION_START_TOKEN = PitchToken.SESSION_START.value

class PackedPitchChunk(NamedTuple): # keep parallel with nn._CONTEXT_FIELD_SPECS
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

def chunk_to_context(chunk: PackedPitchChunk, device: torch.device) -> PackedPitchContext:
    """
    Moves chunk data to device and enforces strict type casting:
    - IDs -> torch.long
    - Stats -> torch.float
    """
    return PackedPitchContext(
        # Categorical / IDs (Must be Long for Embeddings)
        pitcher_id=chunk.pitcher_id.to(device, dtype=torch.long, non_blocking=True), # type: ignore
        batter_id=chunk.batter_id.to(device, dtype=torch.long, non_blocking=True), # type: ignore
        pitcher_throws=chunk.pitcher_throws.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        batter_hits=chunk.batter_hits.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        count_balls=chunk.count_balls.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        count_strikes=chunk.count_strikes.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        outs=chunk.outs.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        bases_state=chunk.bases_state.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        inning=chunk.inning.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        number_through_order=chunk.number_through_order.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        fielder_2_id=chunk.fielder_2_id.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        fielder_3_id=chunk.fielder_3_id.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        fielder_4_id=chunk.fielder_4_id.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        fielder_5_id=chunk.fielder_5_id.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        fielder_6_id=chunk.fielder_6_id.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        fielder_7_id=chunk.fielder_7_id.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        fielder_8_id=chunk.fielder_8_id.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        fielder_9_id=chunk.fielder_9_id.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        batter_days_since_prev_game=chunk.batter_days_since_prev_game.to(device, dtype=torch.int, non_blocking=True), # type: ignore
        pitcher_days_since_prev_game=chunk.pitcher_days_since_prev_game.to(device, dtype=torch.int, non_blocking=True), # type: ignore

        # Continuous / Stats (Must be Float for Linear layers)
        pitcher_age=chunk.pitcher_age.to(device, dtype=torch.float, non_blocking=True), # type: ignore
        batter_age=chunk.batter_age.to(device, dtype=torch.float, non_blocking=True), # type: ignore
        score_bat=chunk.score_bat.to(device, dtype=torch.float, non_blocking=True), # type: ignore
        score_fld=chunk.score_fld.to(device, dtype=torch.float, non_blocking=True), # type: ignore
        pitch_number=chunk.pitch_number.to(device, dtype=torch.float, non_blocking=True), # type: ignore
        game_date=chunk.game_date.to(device, dtype=torch.float, non_blocking=True), # type: ignore
        strike_zone_top=chunk.strike_zone_top.to(device, dtype=torch.float, non_blocking=True), # type: ignore
        strike_zone_bottom=chunk.strike_zone_bottom.to(device, dtype=torch.float, non_blocking=True), # type: ignore
    )


class PackedPitchDataset(Dataset):
    """
    Dataset that samples fixed-length chunks aligned to session boundaries.

    Each chunk starts at a SESSION_START token, ensuring that when indices are
    shuffled, we shuffle which session to start from rather than arbitrary
    token positions. This preserves the semantic structure of the data.
    """

    def __init__(self, data_dir: str, seq_len: int, tokens_file: str = "pitch_seq.bin", context_prefix: str = "pitch_context"):
        self.contexts = {}
        self.tokens = np.memmap(os.path.join(data_dir, tokens_file), dtype=TOKEN_DTYPE, mode="r")
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

        # Find all SESSION_START positions for session-boundary-aligned sampling
        self.session_starts = np.where(self.tokens == SESSION_START_TOKEN)[0]

        if len(self.session_starts) == 0:
            # Fallback to old behavior if no session tokens found (legacy dataset)
            self.session_starts = np.arange(0, max(1, self.total_tokens - self.L), self.L)

        self.num_sessions = len(self.session_starts)

    def set_offset(self, _offset: int) -> None:
        # No longer used for session-aligned sampling, kept for API compatibility
        pass

    def __len__(self) -> int:
        return self.num_sessions

    def __getitem__(self, index: int) -> PackedPitchChunk:
        if index < 0 or index >= self.num_sessions:
            raise IndexError(f"index {index} out of range for PackedPitchDataset with {self.num_sessions} sessions")

        # Start at the session boundary
        start = int(self.session_starts[index])
        end = min(start + self.L + 1, self.total_tokens)

        # Handle case where remaining tokens are less than seq_len + 1
        actual_len = end - start
        if actual_len < 2:
            # Need at least 2 tokens for x and y
            # Fall back to a valid session
            start = int(self.session_starts[0])
            end = min(start + self.L + 1, self.total_tokens)

        chunk_tok = torch.from_numpy(self.tokens[start:end].astype(np.int64))

        # Pad if necessary (when near end of dataset)
        if len(chunk_tok) < self.L + 1:
            pad_len = self.L + 1 - len(chunk_tok)
            chunk_tok = torch.cat([chunk_tok, torch.zeros(pad_len, dtype=torch.int64)])

        x = chunk_tok[:-1]
        y = chunk_tok[1:]

        # Get context slice, handling padding when near end of dataset
        ctx_end = min(start + self.L, self.total_tokens)

        def get_ctx(field_name: str, dtype) -> torch.Tensor:
            data = self.contexts[field_name][start:ctx_end]
            tensor = torch.from_numpy(data.astype(dtype))
            if len(tensor) < self.L:
                pad_len = self.L - len(tensor)
                tensor = torch.cat([tensor, torch.zeros(pad_len, dtype=tensor.dtype)])
            return tensor

        return PackedPitchChunk(
            x=x,  # type: ignore
            y=y,  # type: ignore
            pitcher_id=get_ctx('pitcher_id', np.int64),  # type: ignore
            batter_id=get_ctx('batter_id', np.int64),  # type: ignore
            pitcher_age=get_ctx('pitcher_age', np.float32),  # type: ignore
            pitcher_throws=get_ctx('pitcher_throws', np.int32),  # type: ignore
            batter_age=get_ctx('batter_age', np.float32),  # type: ignore
            batter_hits=get_ctx('batter_hits', np.int32),  # type: ignore
            count_balls=get_ctx('count_balls', np.int32),  # type: ignore
            count_strikes=get_ctx('count_strikes', np.int32),  # type: ignore
            outs=get_ctx('outs', np.int32),  # type: ignore
            bases_state=get_ctx('bases_state', np.int32),  # type: ignore
            score_bat=get_ctx('score_bat', np.float32),  # type: ignore
            score_fld=get_ctx('score_fld', np.float32),  # type: ignore
            inning=get_ctx('inning', np.int32),  # type: ignore
            pitch_number=get_ctx('pitch_number', np.float32),  # type: ignore
            number_through_order=get_ctx('number_through_order', np.int32),  # type: ignore
            game_date=get_ctx('game_date', np.float32),  # type: ignore
            fielder_2_id=get_ctx('fielder_2_id', np.int32),  # type: ignore
            fielder_3_id=get_ctx('fielder_3_id', np.int32),  # type: ignore
            fielder_4_id=get_ctx('fielder_4_id', np.int32),  # type: ignore
            fielder_5_id=get_ctx('fielder_5_id', np.int32),  # type: ignore
            fielder_6_id=get_ctx('fielder_6_id', np.int32),  # type: ignore
            fielder_7_id=get_ctx('fielder_7_id', np.int32),  # type: ignore
            fielder_8_id=get_ctx('fielder_8_id', np.int32),  # type: ignore
            fielder_9_id=get_ctx('fielder_9_id', np.int32),  # type: ignore
            batter_days_since_prev_game=get_ctx('batter_days_since_prev_game', np.int32),  # type: ignore
            pitcher_days_since_prev_game=get_ctx('pitcher_days_since_prev_game', np.int32),  # type: ignore
            strike_zone_top=get_ctx('strike_zone_top', np.float32),  # type: ignore
            strike_zone_bottom=get_ctx('strike_zone_bottom', np.float32),  # type: ignore
        )