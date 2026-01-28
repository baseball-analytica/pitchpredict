#!/usr/bin/env python3
"""
Evaluation script for xlstm checkpoints.

Evaluates one or more trained checkpoints on the test set and reports test loss metrics.

Usage:
    python tools/eval_xlstm.py \
        --ckpt_pattern "/raid/ckpts/pitch_xlstm/*.pt" \
        --output_json eval_results.json

    Or set CHECKPOINT_LIST below and run without --ckpt_pattern:
    python tools/eval_xlstm.py --output_json eval_results.json
"""

# =============================================================================
# CHECKPOINT LIST - Set this to evaluate specific checkpoints instead of glob
# If non-empty, this takes priority over --ckpt_pattern
# =============================================================================
CHECKPOINT_LIST: list[str] = [

    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0058000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0059000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0060000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0061000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0062000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0063000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0064000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0065000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0066000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0067000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0068000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0069000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0070000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0071000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0072000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0073000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0074000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0075000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0076000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0077000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0078000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0079000.pt",
    "/raid/ckpts/pitch_xlstm_sessions/ckpt_step_0080000.pt",
]
# =============================================================================

import argparse
import glob
import json
import math
import os
import random
import re
from dataclasses import asdict
from datetime import datetime

import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.deep.dataset import (
    PackedPitchDataset,
    chunk_to_context,
)

# Import model components from xlstm.py
from pitchpredict.backend.algs.xlstm.model import BaseballxLSTM
from tools.xlstm import Config, set_seed, setup_wandb


def nll_to_bpb(nll: float) -> float:
    """Convert natural-log NLL (per token) to bits-per-byte (base-2)."""
    return float(nll / math.log(2.0))


def build_model(cfg: Config, device: torch.device) -> BaseballxLSTM:
    """Instantiate the xLSTM model (biases will be loaded from checkpoint)."""
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
        act_ckpt=False,  # No need for activation checkpointing during eval
        tie_weights=cfg.tie_weights,
        logits_softcap=cfg.logits_softcap,
        eod_id=cfg.eod_id,
        num_pitchers=cfg.num_pitchers,
        num_batters=cfg.num_batters,
        num_fielders=cfg.num_fielders,
    )
    # Biases will be loaded from checkpoint.
    torch.cuda.set_device(device)
    model.to(device)
    return model


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
) -> tuple[float, float]:
    """Run a full pass over `loader` without gradient, return (nll, bpb)."""
    model.eval()
    total_nll: float = 0.0
    total_tokens = 0
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    with torch.no_grad():
        for chunk in tqdm(loader, desc="Evaluating", leave=False):
            x = chunk.x.to(device, non_blocking=True)
            y = chunk.y.to(device, non_blocking=True)
            x_ctx = chunk_to_context(chunk, device)

            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                logits = model(x, x_ctx)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum", ignore_index=0
                )

            total_nll += loss.item()
            total_tokens += (y != 0).sum().item()  # Only count non-PAD tokens

    nll = total_nll / max(1, total_tokens)
    bpb = nll_to_bpb(nll)
    return nll, bpb


def extract_step_from_path(path: str) -> int:
    """Extract step number from checkpoint filename (e.g., ckpt_step_0010000.pt -> 10000)."""
    basename = os.path.basename(path)
    match = re.search(r"step_(\d+)", basename)
    if match:
        return int(match.group(1))
    # Fallback: try to find any number in the filename
    numbers = re.findall(r"\d+", basename)
    if numbers:
        return int(numbers[-1])
    return 0


def strip_compiled_prefix(state_dict: dict) -> dict:
    """Strip '_orig_mod.' prefix from compiled model state dict keys."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod.") :]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def load_checkpoint(ckpt_path: str, device: torch.device) -> tuple[Config, dict]:
    """Load checkpoint and return config + state dict."""
    map_location = {"cuda:%d" % 0: "cuda:%d" % device.index}
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Strip compiled model prefixes if present
    if "model" in state:
        state["model"] = strip_compiled_prefix(state["model"])

    # Reconstruct config from checkpoint
    ckpt_config = state.get("config", {})
    cfg = Config(**ckpt_config)

    return cfg, state


def print_summary(results: list[dict], output_json: str) -> None:
    """Print formatted summary table to stdout."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 80)
    print("                        EVALUATION RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Checkpoint':<50} {'Step':>8} {'Test NLL':>12} {'Test BPB':>12}")
    print("-" * 80)

    # Results
    best_result = None
    best_nll = float("inf")

    for r in results:
        ckpt_name = os.path.basename(r["checkpoint"])
        if len(ckpt_name) > 48:
            ckpt_name = "..." + ckpt_name[-45:]

        print(
            f"{ckpt_name:<50} {r['step']:>8d} {r['test_nll']:>12.4f} {r['test_bpb']:>12.4f}"
        )

        if r["test_nll"] < best_nll:
            best_nll = r["test_nll"]
            best_result = r

    print("-" * 80)

    if best_result:
        print(
            f"\nBest: {os.path.basename(best_result['checkpoint'])} "
            f"(NLL: {best_result['test_nll']:.4f}, BPB: {best_result['test_bpb']:.4f})"
        )

    print(f"\nResults saved to: {output_json}")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate xlstm checkpoints on the test set."
    )
    parser.add_argument(
        "--ckpt_pattern",
        type=str,
        default=None,
        help='Glob pattern to find checkpoint files (e.g., "/raid/ckpts/*.pt"). '
        "Not required if CHECKPOINT_LIST is set at the top of the script.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory containing test/ subdirectory. If not provided, uses the data_dir from checkpoint config.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="new_eval_results.json",
        help="Path to save JSON results (default: eval_results.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Wandb run ID to resume.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation (default: 128)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Parse device
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    if args.run_id is not None:
        setup_wandb(cfg=Config(), run_id=args.run_id)
        wandb.define_metric("eval_step")
        wandb.define_metric("final_test_loss", step_metric="eval_step")

    # Find all checkpoints (CHECKPOINT_LIST takes priority over glob)
    if CHECKPOINT_LIST:
        ckpt_paths = [p for p in CHECKPOINT_LIST if os.path.exists(p)]
        missing = [p for p in CHECKPOINT_LIST if not os.path.exists(p)]
        if missing:
            print(
                f"Warning: {len(missing)} checkpoint(s) from CHECKPOINT_LIST not found:"
            )
            for p in missing:
                print(f"  - {p}")
        source = "CHECKPOINT_LIST"
    elif args.ckpt_pattern:
        ckpt_paths = glob.glob(args.ckpt_pattern)
        source = f"glob pattern: {args.ckpt_pattern}"
    else:
        print("Error: No checkpoints specified. Either:")
        print("  1. Set CHECKPOINT_LIST at the top of the script, or")
        print("  2. Provide --ckpt_pattern argument")
        return

    if not ckpt_paths:
        print(f"Error: No checkpoints found from {source}")
        return

    # Sort by step number
    ckpt_paths = sorted(ckpt_paths, key=extract_step_from_path)
    print(f"Found {len(ckpt_paths)} checkpoint(s) to evaluate (from {source})")

    results = []
    data_dir_used = None

    for i, ckpt_path in enumerate(ckpt_paths):
        print(
            f"\n[{i + 1}/{len(ckpt_paths)}] Evaluating: {os.path.basename(ckpt_path)}"
        )

        try:
            # Load checkpoint
            cfg, state = load_checkpoint(ckpt_path, device)
            step = state.get("step", extract_step_from_path(ckpt_path))

            # Determine data directory
            data_dir = args.data_dir if args.data_dir else cfg.data_dir
            test_dir = os.path.join(data_dir, "test")

            if not os.path.exists(test_dir):
                print(f"  Warning: Test directory not found: {test_dir}")
                print("  Skipping this checkpoint.")
                continue

            data_dir_used = data_dir

            print(f"  Step: {step}")
            print(
                f"  Model: d_model={cfg.d_model}, blocks={cfg.num_blocks}, heads={cfg.num_heads}"
            )
            print(f"  Batch size: {cfg.micro_batch_size}, Seq len: {cfg.seq_len}")

            model = build_model(cfg, device)
            model.load_state_dict(state["model"], strict=True)
            model = torch.compile(model, mode="default")

            # Load test dataset
            test_ds = PackedPitchDataset(
                data_dir=test_dir,
                seq_len=cfg.seq_len,
                tokens_file="pitch_seq.bin",
                context_prefix=cfg.context_prefix,
            )

            test_loader = DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
            )

            print(f"  Test samples: {len(test_ds)}")

            # Evaluate
            nll, bpb = evaluate(model, test_loader, device, cfg.amp_dtype)

            print(f"  Test NLL: {nll:.4f}")
            print(f"  Test BPB: {bpb:.4f}")

            # Store result
            results.append(
                {
                    "checkpoint": ckpt_path,
                    "step": step,
                    "test_nll": nll,
                    "test_bpb": bpb,
                    "config": asdict(cfg),
                }
            )

            # Log to wandb
            if args.run_id is not None:
                wandb.log({"eval_step": step, "final_test_loss": nll})

            # Free memory
            del model
            del test_loader
            del test_ds
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error evaluating checkpoint: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Print summary
    print_summary(results, args.output_json)

    # Save JSON results
    output_data = {
        "eval_time": datetime.now().isoformat(),
        "ckpt_source": "CHECKPOINT_LIST" if CHECKPOINT_LIST else args.ckpt_pattern,
        "data_dir": data_dir_used,
        "device": str(device),
        "seed": args.seed,
        "num_checkpoints": len(results),
        "results": results,
    }

    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Full results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
