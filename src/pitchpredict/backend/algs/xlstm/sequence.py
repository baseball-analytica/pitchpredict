# SPDX-License-Identifier: MIT
"""Build tokenized sequences from API requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from pitchpredict.backend.algs.xlstm.tokens import PitchToken
from pitchpredict.backend.algs.xlstm.encoding import encode_pitch_from_dict
from pitchpredict.backend.algs.xlstm.model import PackedPitchContext


# Results that count as balls
BALL_RESULTS = frozenset([
    "ball", "ball_in_dirt", "blocked_ball", "automatic_ball",
    "intentional_ball", "intent_ball", "pitchout",
])

# Results that count as strikes (even with 2 strikes)
STRIKE_RESULTS = frozenset([
    "called_strike", "swinging_strike", "swinging_strike_blocked",
    "swinging_pitchout", "foul_tip", "automatic_strike",
])

# Results that count as strikes only if strikes < 2
FOUL_RESULTS = frozenset([
    "foul", "foul_bunt", "bunt_foul_tip", "foul_pitchout",
])

# Results that end the plate appearance
PA_ENDING_RESULTS = frozenset([
    "hit_into_play", "hit_by_pitch", "missed_bunt",
])


@dataclass
class ContextDefaults:
    """Default context values from the request."""
    pitcher_id: int
    batter_id: int
    pitcher_age: int = 28
    pitcher_throws: Literal["L", "R"] = "R"
    batter_age: int = 28
    batter_hits: Literal["L", "R"] = "R"
    count_balls: int = 0
    count_strikes: int = 0
    outs: int = 0
    bases_state: int = 0
    score_bat: int = 0
    score_fld: int = 0
    inning: int = 1
    pitch_number: int = 1
    number_through_order: int = 1
    game_date: int = 2024  # Year as int
    fielder_2_id: int = 0
    fielder_3_id: int = 0
    fielder_4_id: int = 0
    fielder_5_id: int = 0
    fielder_6_id: int = 0
    fielder_7_id: int = 0
    fielder_8_id: int = 0
    fielder_9_id: int = 0
    batter_days_since_prev_game: int = 1
    pitcher_days_since_prev_game: int = 1
    strike_zone_top: float = 3.5
    strike_zone_bottom: float = 1.5


@dataclass
class PerPitchContext:
    """Context for a single pitch position."""
    pitcher_id: int
    batter_id: int
    pitcher_age: float
    pitcher_throws: int  # 0=unknown, 1=L, 2=R
    batter_age: float
    batter_hits: int  # 0=unknown, 1=L, 2=R, 3=S (switch)
    count_balls: int
    count_strikes: int
    outs: int
    bases_state: int
    score_bat: float
    score_fld: float
    inning: int
    pitch_number: float
    number_through_order: int
    game_date: float
    fielder_2_id: int
    fielder_3_id: int
    fielder_4_id: int
    fielder_5_id: int
    fielder_6_id: int
    fielder_7_id: int
    fielder_8_id: int
    fielder_9_id: int
    batter_days_since_prev_game: int
    pitcher_days_since_prev_game: int
    strike_zone_top: float
    strike_zone_bottom: float


def _throws_to_int(throws: str | None) -> int:
    """Convert throws string to int encoding."""
    if throws is None:
        return 0
    return 1 if throws.upper() == "L" else 2


def _hits_to_int(hits: str | None) -> int:
    """Convert batter stance to int encoding."""
    if hits is None:
        return 0
    h = hits.upper()
    if h == "L":
        return 1
    if h == "R":
        return 2
    if h == "S":
        return 3
    return 0


def _merge_context(defaults: ContextDefaults, pitch: dict[str, Any], pitch_number: int, balls: int, strikes: int) -> PerPitchContext:
    """Merge defaults with per-pitch overrides."""
    return PerPitchContext(
        pitcher_id=defaults.pitcher_id,
        batter_id=pitch.get("batter_id") or defaults.batter_id,
        pitcher_age=float(defaults.pitcher_age),
        pitcher_throws=_throws_to_int(defaults.pitcher_throws),
        batter_age=float(pitch.get("batter_age") or defaults.batter_age),
        batter_hits=_hits_to_int(pitch.get("batter_hits") or defaults.batter_hits),
        count_balls=pitch.get("count_balls") if pitch.get("count_balls") is not None else balls,
        count_strikes=pitch.get("count_strikes") if pitch.get("count_strikes") is not None else strikes,
        outs=pitch.get("outs") if pitch.get("outs") is not None else defaults.outs,
        bases_state=pitch.get("bases_state") if pitch.get("bases_state") is not None else defaults.bases_state,
        score_bat=float(pitch.get("score_bat") if pitch.get("score_bat") is not None else defaults.score_bat),
        score_fld=float(pitch.get("score_fld") if pitch.get("score_fld") is not None else defaults.score_fld),
        inning=pitch.get("inning") if pitch.get("inning") is not None else defaults.inning,
        pitch_number=float(pitch.get("pitch_number") if pitch.get("pitch_number") is not None else pitch_number),
        number_through_order=pitch.get("number_through_order") if pitch.get("number_through_order") is not None else defaults.number_through_order,
        game_date=float(defaults.game_date),
        fielder_2_id=defaults.fielder_2_id,
        fielder_3_id=defaults.fielder_3_id,
        fielder_4_id=defaults.fielder_4_id,
        fielder_5_id=defaults.fielder_5_id,
        fielder_6_id=defaults.fielder_6_id,
        fielder_7_id=defaults.fielder_7_id,
        fielder_8_id=defaults.fielder_8_id,
        fielder_9_id=defaults.fielder_9_id,
        batter_days_since_prev_game=defaults.batter_days_since_prev_game,
        pitcher_days_since_prev_game=defaults.pitcher_days_since_prev_game,
        strike_zone_top=defaults.strike_zone_top,
        strike_zone_bottom=defaults.strike_zone_bottom,
    )


def infer_count_after_pitch(result: str, balls: int, strikes: int) -> tuple[int, int, bool]:
    """Infer the count after a pitch result.

    Returns:
        Tuple of (new_balls, new_strikes, pa_ended)
    """
    result_lower = result.lower()

    if result_lower in PA_ENDING_RESULTS:
        return balls, strikes, True

    if result_lower in BALL_RESULTS:
        new_balls = min(balls + 1, 4)
        # Walk ends PA
        if new_balls == 4:
            return new_balls, strikes, True
        return new_balls, strikes, False

    if result_lower in STRIKE_RESULTS:
        new_strikes = min(strikes + 1, 3)
        # Strikeout ends PA
        if new_strikes == 3:
            return balls, new_strikes, True
        return balls, new_strikes, False

    if result_lower in FOUL_RESULTS:
        # Fouls only add strikes if < 2
        if strikes < 2:
            return balls, strikes + 1, False
        return balls, strikes, False

    # Unknown result - don't modify count
    return balls, strikes, False


@dataclass
class SequenceBuilderResult:
    """Result of building a token sequence."""
    tokens: list[int]
    contexts: list[PerPitchContext]
    last_pa_id: int | None = None
    last_balls: int = 0
    last_strikes: int = 0
    pa_open: bool = False  # True if the last PA is still in progress


def build_history_sequence(
    prev_pitches: list[dict[str, Any]],
    defaults: ContextDefaults,
) -> SequenceBuilderResult:
    """Build token sequence from previous pitches.

    Args:
        prev_pitches: List of pitch dictionaries with pa_id and pitch attributes
        defaults: Default context values from the request

    Returns:
        SequenceBuilderResult with tokens and contexts
    """
    tokens: list[int] = []
    contexts: list[PerPitchContext] = []

    if not prev_pitches:
        # Cold start: just SESSION_START + PA_START
        cold_ctx = _merge_context(defaults, {}, pitch_number=1, balls=0, strikes=0)
        tokens.append(PitchToken.SESSION_START.value)
        contexts.append(cold_ctx)
        tokens.append(PitchToken.PA_START.value)
        contexts.append(cold_ctx)
        return SequenceBuilderResult(tokens=tokens, contexts=contexts, last_pa_id=1, last_balls=0, last_strikes=0)

    seen_pa_ids: set[int] = set()
    current_pa_id: int | None = None
    balls = 0
    strikes = 0
    pitch_num_in_pa = 1
    last_ctx: PerPitchContext | None = None
    last_pa_ended = False

    for pitch in prev_pitches:
        pa_id = pitch.get("pa_id")
        if pa_id is None or pa_id <= 0:
            raise ValueError(f"Invalid or missing pa_id in pitch: {pitch}")

        if current_pa_id is None or pa_id != current_pa_id:
            # Close previous PA if needed
            if current_pa_id is not None:
                if last_ctx is None:
                    raise ValueError("Missing context for PA_END")
                tokens.append(PitchToken.PA_END.value)
                contexts.append(last_ctx)

            if pa_id in seen_pa_ids:
                raise ValueError(f"Non-contiguous pa_id detected: {pa_id}")
            seen_pa_ids.add(pa_id)

            # Start new PA
            current_pa_id = pa_id
            balls = 0
            strikes = 0
            pitch_num_in_pa = 1

            cur_balls = pitch.get("count_balls") if pitch.get("count_balls") is not None else balls
            cur_strikes = pitch.get("count_strikes") if pitch.get("count_strikes") is not None else strikes
            ctx = _merge_context(defaults, pitch, pitch_number=pitch_num_in_pa, balls=cur_balls, strikes=cur_strikes)

            if not tokens:
                tokens.append(PitchToken.SESSION_START.value)
                contexts.append(ctx)

            tokens.append(PitchToken.PA_START.value)
            contexts.append(ctx)
        else:
            cur_balls = pitch.get("count_balls") if pitch.get("count_balls") is not None else balls
            cur_strikes = pitch.get("count_strikes") if pitch.get("count_strikes") is not None else strikes
            ctx = _merge_context(defaults, pitch, pitch_number=pitch_num_in_pa, balls=cur_balls, strikes=cur_strikes)

        # Encode pitch to 16 tokens
        pitch_tokens = encode_pitch_from_dict(pitch)

        # Add tokens and contexts
        for t in pitch_tokens:
            tokens.append(t)
            contexts.append(ctx)

        # Update count based on result
        result = pitch.get("result", "")
        balls, strikes, pa_ended = infer_count_after_pitch(result, cur_balls, cur_strikes)

        last_ctx = ctx
        last_pa_ended = pa_ended
        pitch_num_in_pa += 1

    # Close final PA and session only if the last PA actually ended
    if last_ctx is None:
        raise ValueError("No valid pitches in history")

    if last_pa_ended:
        tokens.append(PitchToken.PA_END.value)
        contexts.append(last_ctx)
        tokens.append(PitchToken.SESSION_END.value)
        contexts.append(last_ctx)
        pa_open = False
    else:
        # PA still in progress — leave sequence open (no PA_END or SESSION_END)
        pa_open = True

    return SequenceBuilderResult(
        tokens=tokens,
        contexts=contexts,
        last_pa_id=current_pa_id,
        last_balls=balls,
        last_strikes=strikes,
        pa_open=pa_open,
    )


def contexts_to_packed(
    contexts: list[PerPitchContext],
    device: torch.device,
) -> PackedPitchContext:
    """Convert list of contexts to PackedPitchContext tensors.

    Args:
        contexts: List of per-position contexts
        device: Target device for tensors

    Returns:
        PackedPitchContext with tensors of shape [1, seq_len]
    """
    seq_len = len(contexts)

    # Pre-allocate lists
    pitcher_ids = []
    batter_ids = []
    pitcher_ages = []
    pitcher_throws = []
    batter_ages = []
    batter_hits = []
    count_balls = []
    count_strikes = []
    outs = []
    bases_states = []
    score_bats = []
    score_flds = []
    innings = []
    pitch_numbers = []
    numbers_through_order = []
    game_dates = []
    fielder_2_ids = []
    fielder_3_ids = []
    fielder_4_ids = []
    fielder_5_ids = []
    fielder_6_ids = []
    fielder_7_ids = []
    fielder_8_ids = []
    fielder_9_ids = []
    batter_days = []
    pitcher_days = []
    sz_tops = []
    sz_bottoms = []

    for ctx in contexts:
        pitcher_ids.append(ctx.pitcher_id)
        batter_ids.append(ctx.batter_id)
        pitcher_ages.append(ctx.pitcher_age)
        pitcher_throws.append(ctx.pitcher_throws)
        batter_ages.append(ctx.batter_age)
        batter_hits.append(ctx.batter_hits)
        count_balls.append(ctx.count_balls)
        count_strikes.append(ctx.count_strikes)
        outs.append(ctx.outs)
        bases_states.append(ctx.bases_state)
        score_bats.append(ctx.score_bat)
        score_flds.append(ctx.score_fld)
        innings.append(ctx.inning)
        pitch_numbers.append(ctx.pitch_number)
        numbers_through_order.append(ctx.number_through_order)
        game_dates.append(ctx.game_date)
        fielder_2_ids.append(ctx.fielder_2_id)
        fielder_3_ids.append(ctx.fielder_3_id)
        fielder_4_ids.append(ctx.fielder_4_id)
        fielder_5_ids.append(ctx.fielder_5_id)
        fielder_6_ids.append(ctx.fielder_6_id)
        fielder_7_ids.append(ctx.fielder_7_id)
        fielder_8_ids.append(ctx.fielder_8_id)
        fielder_9_ids.append(ctx.fielder_9_id)
        batter_days.append(ctx.batter_days_since_prev_game)
        pitcher_days.append(ctx.pitcher_days_since_prev_game)
        sz_tops.append(ctx.strike_zone_top)
        sz_bottoms.append(ctx.strike_zone_bottom)

    return PackedPitchContext(
        pitcher_id=torch.tensor([pitcher_ids], dtype=torch.long, device=device),
        batter_id=torch.tensor([batter_ids], dtype=torch.long, device=device),
        pitcher_age=torch.tensor([pitcher_ages], dtype=torch.float, device=device),
        pitcher_throws=torch.tensor([pitcher_throws], dtype=torch.int, device=device),
        batter_age=torch.tensor([batter_ages], dtype=torch.float, device=device),
        batter_hits=torch.tensor([batter_hits], dtype=torch.int, device=device),
        count_balls=torch.tensor([count_balls], dtype=torch.int, device=device),
        count_strikes=torch.tensor([count_strikes], dtype=torch.int, device=device),
        outs=torch.tensor([outs], dtype=torch.long, device=device),
        bases_state=torch.tensor([bases_states], dtype=torch.int, device=device),
        score_bat=torch.tensor([score_bats], dtype=torch.float, device=device),
        score_fld=torch.tensor([score_flds], dtype=torch.float, device=device),
        inning=torch.tensor([innings], dtype=torch.int, device=device),
        pitch_number=torch.tensor([pitch_numbers], dtype=torch.float, device=device),
        number_through_order=torch.tensor([numbers_through_order], dtype=torch.int, device=device),
        game_date=torch.tensor([game_dates], dtype=torch.float, device=device),
        fielder_2_id=torch.tensor([fielder_2_ids], dtype=torch.int, device=device),
        fielder_3_id=torch.tensor([fielder_3_ids], dtype=torch.int, device=device),
        fielder_4_id=torch.tensor([fielder_4_ids], dtype=torch.int, device=device),
        fielder_5_id=torch.tensor([fielder_5_ids], dtype=torch.int, device=device),
        fielder_6_id=torch.tensor([fielder_6_ids], dtype=torch.int, device=device),
        fielder_7_id=torch.tensor([fielder_7_ids], dtype=torch.int, device=device),
        fielder_8_id=torch.tensor([fielder_8_ids], dtype=torch.int, device=device),
        fielder_9_id=torch.tensor([fielder_9_ids], dtype=torch.int, device=device),
        batter_days_since_prev_game=torch.tensor([batter_days], dtype=torch.int, device=device),
        pitcher_days_since_prev_game=torch.tensor([pitcher_days], dtype=torch.int, device=device),
        strike_zone_top=torch.tensor([sz_tops], dtype=torch.float, device=device),
        strike_zone_bottom=torch.tensor([sz_bottoms], dtype=torch.float, device=device),
    )
