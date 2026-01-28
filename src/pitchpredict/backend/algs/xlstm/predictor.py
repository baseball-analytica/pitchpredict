# SPDX-License-Identifier: MIT
"""Autoregressive pitch generation with grammar-constrained sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from pitchpredict.backend.algs.xlstm.tokens import (
    PitchToken,
    TokenCategory,
    valid_next_token_ids,
    get_tokens_in_category,
    TOKENS_PER_PITCH,
    VOCAB_SIZE,
)
from pitchpredict.backend.algs.xlstm.model import BaseballxLSTM, PackedPitchContext
from pitchpredict.backend.algs.xlstm.decoding import decode_pitch, PitchTypeProbabilities


@dataclass
class GenerationConfig:
    """Configuration for pitch generation."""
    sample_size: int = 10
    temperature: float = 1.0
    top_k: int = 5
    top_p: float | None = None  # Nucleus sampling (not used if None)


def create_grammar_mask(
    last_token_id: int,
    vocab_size: int,
    device: torch.device,
    force_category: TokenCategory | None = None,
) -> torch.Tensor:
    """Create a mask that allows only grammatically valid next tokens.

    Args:
        last_token_id: The last token in the sequence
        vocab_size: Size of the vocabulary
        device: Target device

    Returns:
        Boolean tensor of shape [vocab_size] where True = valid token
    """
    if force_category is not None:
        valid_ids = [
            t.value for t in get_tokens_in_category(force_category) if t.value < vocab_size
        ]
    else:
        valid_ids = [v for v in valid_next_token_ids(last_token_id) if v < vocab_size]
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if valid_ids:
        mask[valid_ids] = True
    return mask


def apply_grammar_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply grammar mask to logits, setting invalid tokens to -inf.

    Args:
        logits: Logits tensor of shape [..., vocab_size]
        mask: Boolean mask of shape [vocab_size] where True = valid

    Returns:
        Masked logits with invalid positions set to -inf
    """
    masked = logits.clone()
    masked[..., ~mask] = float("-inf")
    return masked


def sample_with_top_k(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    """Sample from logits using top-k sampling.

    Args:
        logits: Logits tensor of shape [batch, vocab_size]
        k: Number of top tokens to consider
        temperature: Sampling temperature

    Returns:
        Sampled token IDs of shape [batch]
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Get top-k values and indices
    top_values, top_indices = torch.topk(logits, min(k, logits.size(-1)), dim=-1)

    # Sample from top-k
    probs = F.softmax(top_values, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Map back to vocabulary indices
    return top_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)


def expand_context_for_batch(ctx: PackedPitchContext, batch_size: int) -> PackedPitchContext:
    """Expand context tensors from [1, seq_len] to [batch_size, seq_len]."""
    return PackedPitchContext(
        pitcher_id=ctx.pitcher_id.expand(batch_size, -1),
        batter_id=ctx.batter_id.expand(batch_size, -1),
        pitcher_age=ctx.pitcher_age.expand(batch_size, -1),
        pitcher_throws=ctx.pitcher_throws.expand(batch_size, -1),
        batter_age=ctx.batter_age.expand(batch_size, -1),
        batter_hits=ctx.batter_hits.expand(batch_size, -1),
        count_balls=ctx.count_balls.expand(batch_size, -1),
        count_strikes=ctx.count_strikes.expand(batch_size, -1),
        outs=ctx.outs.expand(batch_size, -1),
        bases_state=ctx.bases_state.expand(batch_size, -1),
        score_bat=ctx.score_bat.expand(batch_size, -1),
        score_fld=ctx.score_fld.expand(batch_size, -1),
        inning=ctx.inning.expand(batch_size, -1),
        pitch_number=ctx.pitch_number.expand(batch_size, -1),
        number_through_order=ctx.number_through_order.expand(batch_size, -1),
        game_date=ctx.game_date.expand(batch_size, -1),
        fielder_2_id=ctx.fielder_2_id.expand(batch_size, -1),
        fielder_3_id=ctx.fielder_3_id.expand(batch_size, -1),
        fielder_4_id=ctx.fielder_4_id.expand(batch_size, -1),
        fielder_5_id=ctx.fielder_5_id.expand(batch_size, -1),
        fielder_6_id=ctx.fielder_6_id.expand(batch_size, -1),
        fielder_7_id=ctx.fielder_7_id.expand(batch_size, -1),
        fielder_8_id=ctx.fielder_8_id.expand(batch_size, -1),
        fielder_9_id=ctx.fielder_9_id.expand(batch_size, -1),
        batter_days_since_prev_game=ctx.batter_days_since_prev_game.expand(batch_size, -1),
        pitcher_days_since_prev_game=ctx.pitcher_days_since_prev_game.expand(batch_size, -1),
        strike_zone_top=ctx.strike_zone_top.expand(batch_size, -1),
        strike_zone_bottom=ctx.strike_zone_bottom.expand(batch_size, -1),
    )


def append_token_to_context(ctx: PackedPitchContext, device: torch.device) -> PackedPitchContext:
    """Append one position to context tensors, repeating the last values."""
    batch_size, seq_len = ctx.pitcher_id.shape

    def _append_last(t: torch.Tensor) -> torch.Tensor:
        last = t[:, -1:].clone()
        return torch.cat([t, last], dim=1)

    return PackedPitchContext(
        pitcher_id=_append_last(ctx.pitcher_id),
        batter_id=_append_last(ctx.batter_id),
        pitcher_age=_append_last(ctx.pitcher_age),
        pitcher_throws=_append_last(ctx.pitcher_throws),
        batter_age=_append_last(ctx.batter_age),
        batter_hits=_append_last(ctx.batter_hits),
        count_balls=_append_last(ctx.count_balls),
        count_strikes=_append_last(ctx.count_strikes),
        outs=_append_last(ctx.outs),
        bases_state=_append_last(ctx.bases_state),
        score_bat=_append_last(ctx.score_bat),
        score_fld=_append_last(ctx.score_fld),
        inning=_append_last(ctx.inning),
        pitch_number=_append_last(ctx.pitch_number),
        number_through_order=_append_last(ctx.number_through_order),
        game_date=_append_last(ctx.game_date),
        fielder_2_id=_append_last(ctx.fielder_2_id),
        fielder_3_id=_append_last(ctx.fielder_3_id),
        fielder_4_id=_append_last(ctx.fielder_4_id),
        fielder_5_id=_append_last(ctx.fielder_5_id),
        fielder_6_id=_append_last(ctx.fielder_6_id),
        fielder_7_id=_append_last(ctx.fielder_7_id),
        fielder_8_id=_append_last(ctx.fielder_8_id),
        fielder_9_id=_append_last(ctx.fielder_9_id),
        batter_days_since_prev_game=_append_last(ctx.batter_days_since_prev_game),
        pitcher_days_since_prev_game=_append_last(ctx.pitcher_days_since_prev_game),
        strike_zone_top=_append_last(ctx.strike_zone_top),
        strike_zone_bottom=_append_last(ctx.strike_zone_bottom),
    )


@dataclass
class GenerationResult:
    """Result of pitch generation."""
    generated_tokens: list[list[int]]  # [sample_size, 16] - raw token IDs per sample
    decoded_pitches: list[dict[str, Any]]  # Decoded pitch dictionaries
    pitch_type_probs: dict[str, float]  # From step-0 logits
    step0_logits: torch.Tensor | None = None  # Optional raw logits


@torch.no_grad()
def generate_pitches(
    model: BaseballxLSTM,
    history_tokens: list[int],
    history_context: PackedPitchContext,
    config: GenerationConfig,
    device: torch.device,
    return_logits: bool = False,
    force_first_category: TokenCategory | None = None,
) -> GenerationResult:
    """Generate pitch samples using autoregressive decoding.

    Args:
        model: The xLSTM model in eval mode
        history_tokens: Token IDs for history (including SESSION_START, PA_START, etc.)
        history_context: Context tensors for history [1, history_len]
        config: Generation configuration
        device: Target device
        force_first_category: If set, force step-0 sampling to use this token category

    Returns:
        GenerationResult with generated pitches and statistics
    """
    model.eval()
    batch_size = config.sample_size
    vocab_size = model.vocab_size

    # Expand history for batch
    # history_tokens: [history_len] -> [batch_size, history_len]
    tokens = torch.tensor(history_tokens, dtype=torch.long, device=device).unsqueeze(0)
    tokens = tokens.expand(batch_size, -1).clone()  # Clone to make contiguous

    # Expand context for batch
    ctx = expand_context_for_batch(history_context, batch_size)

    # Get the last token to determine what comes next
    last_token_id = history_tokens[-1]

    # Storage for generated tokens per sample
    generated_per_sample: list[list[int]] = [[] for _ in range(batch_size)]
    step0_logits_saved: torch.Tensor | None = None

    # Generate 16 tokens (one full pitch)
    for step in range(TOKENS_PER_PITCH):
        # Forward pass
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(tokens, ctx)

        # Get logits for the last position
        last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # Save step-0 logits for pitch type probabilities
        if step == 0:
            step0_logits_saved = last_logits.clone() if return_logits else None

            # Calculate pitch type probs from first sample's logits
            pitch_type_probs_obj = PitchTypeProbabilities.from_logits(
                last_logits[0], temperature=config.temperature
            )
            pitch_type_probs = pitch_type_probs_obj.probs

        # Create and apply grammar mask
        # All samples have the same last token at each step, so use the first sample's
        current_last_token = int(tokens[0, -1].item())
        grammar_mask = create_grammar_mask(
            current_last_token,
            vocab_size,
            device,
            force_category=force_first_category if step == 0 else None,
        )
        masked_logits = apply_grammar_mask(last_logits, grammar_mask)

        # Sample next token
        next_tokens = sample_with_top_k(masked_logits, config.top_k, config.temperature)

        # Store generated tokens
        for i in range(batch_size):
            generated_per_sample[i].append(int(next_tokens[i].item()))

        # Append to sequence
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)

        # Expand context for new position
        ctx = append_token_to_context(ctx, device)

    # Decode generated pitches
    decoded_pitches = []
    for sample_tokens in generated_per_sample:
        try:
            pitch_dict = decode_pitch(sample_tokens)
            decoded_pitches.append(pitch_dict)
        except Exception as e:
            # If decoding fails, skip this sample
            pass

    return GenerationResult(
        generated_tokens=generated_per_sample,
        decoded_pitches=decoded_pitches,
        pitch_type_probs=pitch_type_probs,
        step0_logits=step0_logits_saved,
    )
