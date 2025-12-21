#!/usr/bin/env python3
"""
Detailed evaluation script for a single xlstm checkpoint.

Evaluates on a per-session basis using sliding windows, ensuring each token
gets maximum available context. Optionally saves per-token predictions for
downstream analysis (confusion matrices, calibration, etc.).

Usage:
    uv run python scripts/detailed_eval_xlstm.py --ckpt /path/to/checkpoint.pt

    # With prediction saving:
    uv run python scripts/detailed_eval_xlstm.py --ckpt /path/to/checkpoint.pt \
        --save_logits predictions.npz
"""

import argparse
import math
import os
import re
from typing import Optional

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from pitchpredict.backend.algs.deep.dataset import (
    PackedPitchContext,
    SESSION_START_TOKEN,
)
from pitchpredict.backend.algs.deep.nn import _CONTEXT_FIELD_SPECS, _context_field_path, TOKEN_DTYPE
from pitchpredict.backend.xlstm import (
    BaseballxLSTM,
    Config,
    set_seed,
)


def nll_to_bpb(nll: float) -> float:
    """Convert natural-log NLL (per token) to bits-per-byte (base-2)."""
    return float(nll / math.log(2.0))


def build_model(cfg: Config, device: torch.device) -> BaseballxLSTM:
    """Instantiate the xLSTM model."""
    model = BaseballxLSTM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        dqk_factor=cfg.dqk_factor,
        denom_floor=cfg.denom_floor,
        gate_softcap=cfg.gate_softcap,
        dropout=cfg.dropout,
        detach_interval=cfg.tbptt,
        act_ckpt=False,
        tie_weights=cfg.tie_weights,
        logits_softcap=cfg.logits_softcap,
        eod_id=cfg.eod_id,
        num_pitchers=cfg.num_pitchers,
        num_batters=cfg.num_batters,
        num_fielders=cfg.num_fielders,
    )
    torch.cuda.set_device(device)
    model.to(device)
    return model


def strip_compiled_prefix(state_dict: dict) -> dict:
    """Strip '_orig_mod.' prefix from compiled model state dict keys."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def load_checkpoint(ckpt_path: str, device: torch.device) -> tuple[Config, dict]:
    """Load checkpoint and return config + state dict."""
    map_location = {"cuda:%d" % 0: "cuda:%d" % device.index}
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    if "model" in state:
        state["model"] = strip_compiled_prefix(state["model"])

    ckpt_config = state.get("config", {})
    cfg = Config(**ckpt_config)

    return cfg, state


def extract_step_from_path(path: str) -> int:
    """Extract step number from checkpoint filename."""
    basename = os.path.basename(path)
    match = re.search(r"step_(\d+)", basename)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", basename)
    if numbers:
        return int(numbers[-1])
    return 0


class SessionDataLoader:
    """
    Loads tokens and context data, providing access to individual sessions.
    """

    def __init__(self, data_dir: str, context_prefix: str = "pitch_context"):
        self.data_dir = data_dir
        tokens_path = os.path.join(data_dir, "pitch_seq.bin")
        self.tokens = np.memmap(tokens_path, dtype=TOKEN_DTYPE, mode="r")
        self.total_tokens = len(self.tokens)

        # Load context fields
        context_prefix_path = os.path.join(data_dir, context_prefix)
        self.contexts: dict[str, np.memmap] = {}
        for field_name, spec in _CONTEXT_FIELD_SPECS.items():
            path = _context_field_path(context_prefix_path, field_name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"context field file not found: {path}")
            self.contexts[field_name] = np.memmap(path, dtype=spec.dtype, mode="r")

        # Find session boundaries
        self.session_starts = np.where(self.tokens == SESSION_START_TOKEN)[0]
        self.num_sessions = len(self.session_starts)

        # Compute session ends
        self.session_ends = np.zeros(self.num_sessions, dtype=np.int64)
        for i in range(self.num_sessions - 1):
            self.session_ends[i] = self.session_starts[i + 1]
        self.session_ends[-1] = self.total_tokens

    def get_session(self, idx: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Get tokens and context for a session."""
        start = int(self.session_starts[idx])
        end = int(self.session_ends[idx])
        tokens = np.array(self.tokens[start:end], dtype=np.int64)
        context = {
            field_name: np.array(self.contexts[field_name][start:end])
            for field_name in _CONTEXT_FIELD_SPECS
        }
        return tokens, context

    def __len__(self) -> int:
        return self.num_sessions


def make_context_tensor(
    context: dict[str, np.ndarray],
    start: int,
    end: int,
    device: torch.device,
) -> PackedPitchContext:
    """Create a PackedPitchContext from a context slice."""
    def get_field(name: str, dtype: torch.dtype) -> torch.Tensor:
        arr = context[name][start:end]
        return torch.from_numpy(arr).unsqueeze(0).to(device, dtype=dtype, non_blocking=True)

    return PackedPitchContext(
        pitcher_id=get_field("pitcher_id", torch.long),
        batter_id=get_field("batter_id", torch.long),
        pitcher_age=get_field("pitcher_age", torch.float),
        pitcher_throws=get_field("pitcher_throws", torch.int),
        batter_age=get_field("batter_age", torch.float),
        batter_hits=get_field("batter_hits", torch.int),
        count_balls=get_field("count_balls", torch.int),
        count_strikes=get_field("count_strikes", torch.int),
        outs=get_field("outs", torch.int),
        bases_state=get_field("bases_state", torch.int),
        score_bat=get_field("score_bat", torch.float),
        score_fld=get_field("score_fld", torch.float),
        inning=get_field("inning", torch.int),
        pitch_number=get_field("pitch_number", torch.float),
        number_through_order=get_field("number_through_order", torch.int),
        game_date=get_field("game_date", torch.float),
        fielder_2_id=get_field("fielder_2_id", torch.int),
        fielder_3_id=get_field("fielder_3_id", torch.int),
        fielder_4_id=get_field("fielder_4_id", torch.int),
        fielder_5_id=get_field("fielder_5_id", torch.int),
        fielder_6_id=get_field("fielder_6_id", torch.int),
        fielder_7_id=get_field("fielder_7_id", torch.int),
        fielder_8_id=get_field("fielder_8_id", torch.int),
        fielder_9_id=get_field("fielder_9_id", torch.int),
        batter_days_since_prev_game=get_field("batter_days_since_prev_game", torch.int),
        pitcher_days_since_prev_game=get_field("pitcher_days_since_prev_game", torch.int),
        strike_zone_top=get_field("strike_zone_top", torch.float),
        strike_zone_bottom=get_field("strike_zone_bottom", torch.float),
    )


def evaluate_session(
    model: torch.nn.Module,
    tokens: np.ndarray,
    context: dict[str, np.ndarray],
    seq_len: int,
    stride: int,
    device: torch.device,
    amp_dtype: str,
    vocab_size: int,
    save_predictions: bool = False,
) -> tuple[float, int, Optional[dict]]:
    """
    Evaluate one session with sliding windows.

    Returns:
        (total_loss, total_tokens, predictions_dict or None)
    """
    session_len = len(tokens)
    if session_len < 2:
        return 0.0, 0, None

    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    total_loss = 0.0
    total_tokens = 0

    # For saving predictions
    all_targets = [] if save_predictions else None
    all_predictions = [] if save_predictions else None
    all_top_k_indices = [] if save_predictions else None
    all_top_k_probs = [] if save_predictions else None

    # Case 1: Session fits in one window
    if session_len <= seq_len + 1:
        x = torch.from_numpy(tokens[:-1]).unsqueeze(0).to(device, dtype=torch.long)
        y = torch.from_numpy(tokens[1:]).to(device, dtype=torch.long)
        ctx = make_context_tensor(context, 0, len(tokens) - 1, device)

        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            logits = model(x, ctx)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1), reduction="sum"
            )

        total_loss = loss.item()
        total_tokens = len(y)

        if save_predictions:
            probs = F.softmax(logits.squeeze(0).float(), dim=-1)
            top_k = torch.topk(probs, k=5, dim=-1)
            all_targets.append(y.cpu().numpy())
            all_predictions.append(probs.argmax(dim=-1).cpu().numpy())
            all_top_k_indices.append(top_k.indices.cpu().numpy())
            all_top_k_probs.append(top_k.values.cpu().numpy())

    else:
        # Case 2: Sliding window for long sessions
        pos = 0
        counted_up_to = 0  # Track which positions we've already counted

        while pos < session_len - 1:
            # Window end (exclusive), ensuring we have at least one target
            window_end = min(pos + seq_len + 1, session_len)
            window_len = window_end - pos

            if window_len < 2:
                break

            # x = tokens[pos:window_end-1], y = tokens[pos+1:window_end]
            x = torch.from_numpy(tokens[pos:window_end - 1]).unsqueeze(0).to(device, dtype=torch.long)
            y = torch.from_numpy(tokens[pos + 1:window_end]).to(device, dtype=torch.long)
            ctx = make_context_tensor(context, pos, window_end - 1, device)

            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                logits = model(x, ctx)

            # Determine which positions to count (avoid double-counting)
            # Position in y corresponds to predicting token at (pos + 1 + i)
            # We've already counted up to `counted_up_to`, so skip those
            if pos == 0:
                count_start = 0
            else:
                # Skip positions we've already counted
                # counted_up_to is the absolute token position
                # y[i] predicts token at position (pos + 1 + i)
                count_start = max(0, counted_up_to - (pos + 1))

            if count_start >= len(y):
                pos += stride
                continue

            y_new = y[count_start:]
            logits_new = logits[:, count_start:, :]

            loss = F.cross_entropy(
                logits_new.view(-1, vocab_size), y_new.view(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += len(y_new)

            if save_predictions:
                probs = F.softmax(logits_new.squeeze(0).float(), dim=-1)
                top_k = torch.topk(probs, k=5, dim=-1)
                all_targets.append(y_new.cpu().numpy())
                all_predictions.append(probs.argmax(dim=-1).cpu().numpy())
                all_top_k_indices.append(top_k.indices.cpu().numpy())
                all_top_k_probs.append(top_k.values.cpu().numpy())

            # Update counted_up_to
            counted_up_to = pos + 1 + len(y)

            # Move window
            pos += stride

    predictions_dict = None
    if save_predictions and all_targets:
        predictions_dict = {
            "targets": np.concatenate(all_targets),
            "predictions": np.concatenate(all_predictions),
            "top_k_indices": np.concatenate(all_top_k_indices),
            "top_k_probs": np.concatenate(all_top_k_probs),
        }

    return total_loss, total_tokens, predictions_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detailed per-session evaluation of a single xlstm checkpoint."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory. If not provided, uses checkpoint config.",
    )
    parser.add_argument(
        "--test_dir_name",
        type=str,
        default="test",
        help="Name of test directory. If not provided, uses data_dir/test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device (default: cuda:0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--save_logits",
        type=str,
        default=None,
        help="Path to save predictions (.npz file) for downstream analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    cfg, state = load_checkpoint(args.ckpt, device)
    step = state.get("step", extract_step_from_path(args.ckpt))

    data_dir = args.data_dir if args.data_dir else cfg.data_dir
    test_dir = os.path.join(data_dir, args.test_dir_name)

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return

    print(f"Step: {step}")
    print(f"Model: d_model={cfg.d_model}, blocks={cfg.num_blocks}, heads={cfg.num_heads}")
    print(f"Seq len: {cfg.seq_len}")
    print(f"Data: {test_dir}")

    # Build and load model
    model = build_model(cfg, device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    # Load session data
    loader = SessionDataLoader(test_dir, context_prefix=cfg.context_prefix)
    print(f"Sessions: {loader.num_sessions}")
    print(f"Total tokens: {loader.total_tokens}")

    # Sliding window parameters
    seq_len = cfg.seq_len
    stride = seq_len // 2
    print(f"Stride: {stride}")

    # Evaluate
    total_loss = 0.0
    total_tokens = 0
    save_predictions = args.save_logits is not None

    all_session_preds = [] if save_predictions else None
    all_session_ids = [] if save_predictions else None

    with torch.no_grad():
        for session_idx in tqdm(range(loader.num_sessions), desc="Evaluating sessions"):
            tokens, context = loader.get_session(session_idx)

            loss, n_tokens, preds = evaluate_session(
                model=model,
                tokens=tokens,
                context=context,
                seq_len=seq_len,
                stride=stride,
                device=device,
                amp_dtype=cfg.amp_dtype,
                vocab_size=cfg.vocab_size,
                save_predictions=save_predictions,
            )

            total_loss += loss
            total_tokens += n_tokens

            if save_predictions and preds is not None:
                all_session_preds.append(preds)
                all_session_ids.append(
                    np.full(len(preds["targets"]), session_idx, dtype=np.uint32)
                )

    nll = total_loss / max(1, total_tokens)
    bpb = nll_to_bpb(nll)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Total tokens evaluated: {total_tokens}")
    print(f"Test NLL: {nll:.4f}")
    print(f"Test BPB: {bpb:.4f}")
    print("=" * 50)

    # Save predictions if requested
    if save_predictions and all_session_preds:
        print(f"\nSaving predictions to: {args.save_logits}")
        np.savez_compressed(
            args.save_logits,
            targets=np.concatenate([p["targets"] for p in all_session_preds]).astype(np.uint16),
            predictions=np.concatenate([p["predictions"] for p in all_session_preds]).astype(np.uint16),
            top_k_indices=np.concatenate([p["top_k_indices"] for p in all_session_preds]).astype(np.uint16),
            top_k_probs=np.concatenate([p["top_k_probs"] for p in all_session_preds]).astype(np.float16),
            session_ids=np.concatenate(all_session_ids),
        )
        total_preds = sum(len(p["targets"]) for p in all_session_preds)
        print(f"Saved {total_preds} predictions")


if __name__ == "__main__":
    main()
