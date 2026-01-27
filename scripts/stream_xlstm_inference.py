#!/usr/bin/env python3
"""Stream inference from an xLSTM checkpoint directly in the terminal.

uv run stream_xlstm_inference --ckpt /raid/ckpts/pitch_xlstm/ckpt_step_0010000.pt \
    --data-dir /raid/kline/pitchpredict/.pitchpredict_data/test
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib.util
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pitchpredict.backend.algs.deep.dataset import (
    PackedPitchContext,
    PackedPitchDataset,
    PackedPitchChunk,
    chunk_to_context,
)
from pitchpredict.backend.algs.deep.types import PitchToken


def nll_to_bpb(nll: float) -> float:
    """Convert natural-log NLL (per token) to bits-per-byte (base-2)."""
    return float(nll / math.log(2.0))


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype: str
) -> tuple[float, float]:
    """Run a full pass over `loader` without gradient, return (nll, bpb)."""
    model.eval()
    total_nll: float = 0.0
    total_tokens = 0
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    with torch.no_grad():
        for ix, chunk in enumerate(loader):
            x = chunk.x.to(device, non_blocking=True)
            y = chunk.y.to(device, non_blocking=True)
            x_ctx = chunk_to_context(chunk, device)
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                logits = model(x, x_ctx)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
                )
            total_nll += loss.item()
            total_tokens += y.numel()
    nll = total_nll / max(1, total_tokens)
    bpb = nll_to_bpb(nll)
    model.train()
    return nll, bpb


def _load_xlstm_module() -> object:
    """Dynamically import `scripts/xlstm.py` so we can reuse its classes."""
    xlstm_path = Path(__file__).resolve().parent / "xlstm.py"
    if not xlstm_path.exists():
        raise FileNotFoundError(f"xlstm.py not found at {xlstm_path}")

    spec = importlib.util.spec_from_file_location(
        "pitchpredict_scripts_xlstm", xlstm_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import xlstm module from {xlstm_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _build_model(
    xlstm_mod: object,
    cfg_obj: object,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate BaseballxLSTM with the config pulled from the checkpoint."""
    model = xlstm_mod.BaseballxLSTM(
        vocab_size=cfg_obj.vocab_size,
        d_model=cfg_obj.d_model,
        num_heads=cfg_obj.num_heads,
        num_blocks=cfg_obj.num_blocks,
        dqk_factor=cfg_obj.dqk_factor,
        dropout=cfg_obj.dropout,
        denom_floor=cfg_obj.denom_floor,
        gate_softcap=cfg_obj.gate_softcap,
        detach_interval=cfg_obj.tbptt,
        act_ckpt=bool(cfg_obj.act_ckpt),
        tie_weights=bool(cfg_obj.tie_weights),
        logits_softcap=cfg_obj.logits_softcap,
        eod_id=cfg_obj.eod_id,
        num_pitchers=cfg_obj.num_pitchers,
        num_batters=cfg_obj.num_batters,
    )
    xlstm_mod.init_gate_biases_v2(model)
    state_dict = maybe_strip_compile_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state dict mismatch: missing={missing} unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model


def _batchify_context(ctx: PackedPitchContext) -> PackedPitchContext:
    """Add a batch dimension (B=1) to every tensor in the context tuple."""
    return PackedPitchContext(
        **{field: getattr(ctx, field).unsqueeze(0) for field in ctx._fields}  # type: ignore[arg-type]
    )


def _format_token(token_id: int) -> str:
    try:
        return PitchToken(token_id).name
    except ValueError:
        return f"ID_{token_id}"


def _stream_rows(
    logits: torch.Tensor,
    chunk: PackedPitchChunk,
    *,
    topk: int,
    delay: float,
    limit: int,
) -> None:
    """Print streaming predictions row-by-row to stdout."""
    steps = min(limit, logits.size(1))
    probs = F.softmax(logits, dim=-1)
    token_inputs = chunk.x.tolist()
    targets = chunk.y.tolist()

    print(f"\nStreaming {steps} tokens (delay={delay:.3f}s, topk={topk})", flush=True)
    print("-" * 80, flush=True)
    header = f"{'idx':>4} | {'in':>6} | {'target':>24} | {'pred':>24} | top{topk}"
    print(header, flush=True)
    print("-" * 80, flush=True)

    for idx in range(steps):
        row_logits = logits[0, idx]
        row_probs = probs[0, idx]
        pred_id = int(torch.argmax(row_probs).item())
        pred_prob = float(row_probs[pred_id].item())
        k = min(topk, row_probs.size(-1))
        top_vals = torch.topk(row_probs, k=k)
        top_pairs = zip(top_vals.indices.tolist(), top_vals.values.tolist())
        top_summary = ", ".join(
            f"{_format_token(tok)}:{prob:.2%}" for tok, prob in top_pairs
        )
        print(
            f"{idx:4d} | {token_inputs[idx]:6d} | {_format_token(targets[idx]):>24} | "
            f"{_format_token(pred_id):>24} ({pred_prob:.2%}) | {top_summary}",
            flush=True,
        )
        if delay > 0:
            time.sleep(delay)
    print("-" * 80, flush=True)


def _prepare_batch(
    chunk: PackedPitchChunk, device: torch.device
) -> tuple[torch.LongTensor, PackedPitchContext]:
    x = chunk.x.unsqueeze(0).to(device=device, dtype=torch.long)
    ctx = chunk_to_context(chunk, device)
    ctx = _batchify_context(ctx)
    return x, ctx


def _select_chunk(
    dataset: PackedPitchDataset, *, chunk_index: int | None
) -> tuple[int, PackedPitchChunk]:
    if len(dataset) == 0:
        raise ValueError("dataset contains no chunks, check data_dir/seq_len settings")
    if chunk_index is None:
        chunk_index = random.randrange(len(dataset))
    if not (0 <= chunk_index < len(dataset)):
        raise IndexError(
            f"chunk_index {chunk_index} out of range (0 <= idx < {len(dataset)})"
        )
    chunk = dataset[chunk_index]
    return chunk_index, chunk


def maybe_strip_compile_prefix(
    state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if state_dict and all(key.startswith(prefix) for key in state_dict.keys()):
        print(f"[load] detected torch.compile checkpoint; stripping '{prefix}' prefix")
        stripped = OrderedDict()
        for key, value in state_dict.items():
            stripped[key[len(prefix) :]] = value
        return stripped
    return state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream predictions from an xLSTM checkpoint."
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help="Path to checkpoint produced by scripts/xlstm.py",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing pitch_seq.bin + context bins",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the checkpoint on the test set."
    )
    parser.add_argument(
        "--context-prefix",
        type=str,
        default=None,
        help="Override context prefix (defaults to cfg.context_prefix)",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Specific chunk index to stream. Random when omitted.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Dataset token offset before chunk selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device string (e.g. cuda:0, cpu). Auto-detect when omitted.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=128,
        help="Maximum number of time-steps to stream.",
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="How many tokens to show per step."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds to sleep between rows (0 to disable).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional RNG seed for chunk sampling."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    if "config" not in state or "model" not in state:
        raise KeyError("checkpoint missing 'config' or 'model' entries")

    xlstm_mod = _load_xlstm_module()
    cfg = xlstm_mod.Config(**state["config"])

    data_dir = Path(args.data_dir or cfg.data_dir).expanduser()
    context_prefix = args.context_prefix or cfg.context_prefix
    dataset = PackedPitchDataset(
        data_dir=str(data_dir),
        seq_len=cfg.seq_len,
        tokens_file="pitch_seq.bin",
        context_prefix=context_prefix,
    )
    dataset.set_offset(int(args.offset))

    model = _build_model(xlstm_mod, cfg, state["model"], device)
    if args.eval:
        sampler: DistributedSampler[torch.LongTensor] = DistributedSampler(
            dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.micro_batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        test_nll, test_bpb = evaluate(model, loader, device, "bf16")
        print(f"[eval] test_nll={test_nll:.4f}  test_bpb={test_bpb:.4f}", flush=True)
    else:
        chunk_idx, chunk = _select_chunk(dataset, chunk_index=args.chunk_index)
        print(f"[info] Loaded chunk {chunk_idx} from {data_dir}", flush=True)

        x_batch, ctx_batch = _prepare_batch(chunk, device)
        with torch.no_grad():
            logits = model(x_batch, ctx_batch)

        _stream_rows(
            logits,
            chunk,
            topk=max(1, args.topk),
            delay=max(0.0, args.delay),
            limit=max(1, args.max_steps),
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
