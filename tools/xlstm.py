import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb # type: ignore

from pitchpredict.backend.algs.xlstm.model import (
    BaseballxLSTM,
    RMSNorm,
    init_gate_biases,
)
from tools.deep.types import PitchToken
from tools.deep.dataset import PackedPitchDataset, chunk_to_context

"""
uv run torchrun --standalone --nproc_per_node=6 tools/xlstm.py \
    --resume_path /raid/ckpts/pitch_xlstm/ckpt_step_0010000.pt \
    --run_id r7gi1bee
"""


# Allow TF32 (throughput improvement)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility.
    Note: Full determinism is not guaranteed with CuDNN, but this gets you close.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_setup(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize a default process group for DDP.
    We use 'nccl' for multi-GPU single-node training on NVIDIA GPUs.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def ddp_cleanup() -> None:
    """Tear down the default process group when training finishes or is aborted."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return True if we are in rank 0 process; use this to guard logging and checkpointing."""
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def log_if_main(s: str) -> None:
    """Print only from main process to avoid clutter."""
    if is_main_process():
        print(s, flush=True)


def setup_wandb(cfg: "Config", run_id: str | None = None) -> None:
    if is_main_process():
        if run_id is not None:
            wandb.init(project="baseball-xlstm", id=run_id, resume="allow")
        else:
            wandb.init(
                project="baseball-xlstm",
                config=(asdict(cfg)),
                id=cfg.run_id,
                resume=("never" if cfg.run_id is None else "allow"),
            )
            cfg.run_id = wandb.run.id


@dataclass
class Config:
    """All hyperparameters and run-time options in one place.
    Adjust them from CLI flags in main().
    """

    # Data
    data_dir: str = "/raid/kline/pitchpredict/.pitchpredict_session_data"
    out_dir: str = "/raid/ckpts/pitch_xlstm_sessions"
    context_prefix: str = "pitch_context"
    run_id: str | None = None
    # Seq
    vocab_size: int = 258
    seq_len: int = 512

    # Model
    d_model: int = 384
    num_blocks: int = 12
    dqk_factor: float = 0.5
    num_heads: int = 8
    dropout: float = 0.0
    tie_weights: bool = False
    logits_softcap: float = 30.0

    # Baseball
    num_pitchers: int = 3000
    num_batters: int = 3700
    num_fielders: int = 3000  # shared embedding for all fielder positions

    # Gating
    denom_floor: float = 1.0
    gate_softcap: float = 15.0

    # Optim & Schedule
    lr: float = 5e-4
    betas: tuple[float, float] = (0.99, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.10
    warmup_steps: int = 2000
    lr_min_scale: float = 0.10
    grad_clip: float = 0.50

    # Batch & Train
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    max_steps: int = 100000

    # Memory features
    tbptt: int = 0  # 0 = off; else detach interval (e.g., 256)
    act_ckpt: int = 0  # 0/1 toggle

    # Misc
    eod_id: int = PitchToken.SESSION_END.value
    amp_dtype: str = "bf16"
    amp_scalar_init: float = 2.0
    seed: int = 64
    resume_path: str | None = None
    log_interval: int = 1
    eval_interval: int = 500
    ckpt_interval: int = 1000


def create_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with standard hyperparameters; we separate out weight decay from LayerNorm/embeddings."""
    decay, no_decay = [], []
    for m in model.modules():
        for name, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            # no decay for norms & embeddings & biases
            if isinstance(m, (nn.LayerNorm, RMSNorm, nn.Embedding)) or name.endswith(
                "bias"
            ):
                no_decay.append(p)
            else:
                decay.append(p)

    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        fused=True,
    )
    return optim


class WarmupCosineScheduler:
    """Simple warmup → cosine decay scheduler over *steps* (not tokens)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_scale: float = 0.10,
        base_lr: float | None = None,
    ):
        self.opt = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.max_steps = max(1, int(max_steps))
        self.min_lr_scale = float(min_lr_scale)
        self.base_lr = (
            base_lr if base_lr is not None else optimizer.param_groups[0]["lr"]
        )
        self.step_num = 0

    def step(self) -> None:
        """Update learning rate after each optimizer step."""
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            scale = self.step_num / self.warmup_steps
        else:
            t = (self.step_num - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            scale = self.min_lr_scale + (1.0 - self.min_lr_scale) * cos

        for pg in self.opt.param_groups:
            pg["lr"] = self.base_lr * scale


def nll_to_bpb(nll: float) -> float:
    """Convert natural-log NLL (per token) to bits-per-byte (base-2)."""
    return float(nll / math.log(2.0))


def build_model(cfg: Config, device: torch.device) -> BaseballxLSTM:
    """Instantiate the xLSTM model and initialize gate biases."""
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
        act_ckpt=bool(cfg.act_ckpt),
        tie_weights=cfg.tie_weights,
        logits_softcap=cfg.logits_softcap,
        eod_id=cfg.eod_id,
        num_pitchers=cfg.num_pitchers,
        num_batters=cfg.num_batters,
        num_fielders=cfg.num_fielders,
    )
    init_gate_biases(model)
    torch.cuda.set_device(device)
    model.to(device)
    return model


def load_ckpt_if_any(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    cfg: Config,
    device: torch.device,
    *,
    load_optimizer: bool = True,
    continue_scheduler: bool = True,
    override_opt_hparams: bool = False,  # <-- make new config win
) -> int:
    if cfg.resume_path is None:
        return 0
    map_location = {"cuda:%d" % 0: "cuda:%d" % device.index}
    state = torch.load(cfg.resume_path, map_location=map_location)
    cfg.run_id = state["config"].get("run_id", cfg.run_id)

    # 1) Weights
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        print(f"[ckpt] missing={missing} unexpected={unexpected}")

    # 2) Optimizer (optional)
    if load_optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        if override_opt_hparams:
            # Re-apply *new* LR / WD / betas from cfg (so new config wins)
            for pg in optimizer.param_groups:
                # Keep the per-group WD structure (decay vs no_decay)
                if pg.get("weight_decay", None) is not None:
                    # If this group is meant to decay, leave it as-is or set to cfg.weight_decay
                    # If it's a no_decay group, it will already be 0.0
                    if pg["weight_decay"] != 0.0:
                        pg["weight_decay"] = cfg.weight_decay
                pg["lr"] = cfg.lr
            optimizer.defaults.update(lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)

    # 3) Scheduler (optional)
    if continue_scheduler and "scheduler" in state:
        scheduler.step_num = int(state["scheduler"].get("step_num", 0))
    else:
        scheduler.step_num = 0  # start the schedule fresh with your new warmup/decay

    step = int(state.get("step", 0))
    print(f"[ckpt] resumed from {cfg.resume_path} (step={step})")
    return step


def save_ckpt(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    step: int,
    cfg: Config,
) -> None:
    if not is_main_process():
        return
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, f"ckpt_step_{step:07d}.pt")
    state = {
        "config": asdict(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {"step_num": scheduler.step_num},
        "step": step,
    }
    torch.save(state, path)
    log_if_main(f"[ckpt] saved {path}")


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
                    logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum", ignore_index=0
                )
            total_nll += loss.item()
            total_tokens += (y != 0).sum().item()  # don't count PAD tokens
    nll = total_nll / max(1, total_tokens)
    bpb = nll_to_bpb(nll)
    model.train()
    return nll, bpb


def train(rank: int, world_size: int, cfg: Config) -> None:
    ddp_setup(rank, world_size)
    
    set_seed(cfg.seed + rank)

    device = torch.device(f"cuda:{rank}")

    if is_main_process():
        log_if_main(
            "Preparing dataset ..."
        )

    val_dir = os.path.join(cfg.data_dir, "val")
    test_dir = os.path.join(cfg.data_dir, "test")

    train_ds = PackedPitchDataset(
        data_dir=str(cfg.data_dir),
        seq_len=cfg.seq_len,
        tokens_file="pitch_seq.bin",
        context_prefix=cfg.context_prefix,
    )
    val_ds = PackedPitchDataset(
        data_dir=str(val_dir),
        seq_len=cfg.seq_len,
        tokens_file="pitch_seq.bin",
        context_prefix=cfg.context_prefix,
    )
    test_ds = PackedPitchDataset(
        data_dir=str(test_dir),
        seq_len=cfg.seq_len,
        tokens_file="pitch_seq.bin",
        context_prefix=cfg.context_prefix,
    )

    train_sampler: DistributedSampler[torch.LongTensor] = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    val_sampler: DistributedSampler[torch.LongTensor] = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    test_sampler: DistributedSampler[torch.LongTensor] = DistributedSampler(
        test_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.micro_batch_size,
        sampler=train_sampler,
        num_workers=28,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.micro_batch_size,
        sampler=val_sampler,
        num_workers=14,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.micro_batch_size,
        sampler=test_sampler,
        num_workers=14,
        pin_memory=True,
        drop_last=True,
    )

    _model = build_model(cfg, device)
    _model = torch.compile(_model, mode="default")  # type: ignore
    if world_size > 1:
        model = DDP(
            _model,
            device_ids=[rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        model = _model  # type: ignore

    optimizer = create_optimizer(model, cfg)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        min_lr_scale=cfg.lr_min_scale,
        base_lr=cfg.lr,
    )

    use_bf16 = True
    autocast_dtype = torch.bfloat16

    global_step = 0
    raw_model = model.module if isinstance(model, DDP) else model
    if cfg.resume_path is not None and os.path.exists(cfg.resume_path):
        global_step = load_ckpt_if_any(raw_model, optimizer, scheduler, cfg, device)

    setup_wandb(cfg)

    model.train()

    tokens_per_microbatch_per_gpu = cfg.micro_batch_size * cfg.seq_len
    tokens_per_step_all_gpus = tokens_per_microbatch_per_gpu * world_size
    tokens_per_update = tokens_per_step_all_gpus * cfg.grad_accum_steps
    if is_main_process():
        log_if_main(
            f"[setup] world_size={world_size}  device={device}  bf16={use_bf16}  tbptt={cfg.tbptt}  act_ckpt={bool(cfg.act_ckpt)}"
        )
        log_if_main(
            f"[setup] tokens/update ≈ {tokens_per_update:,}  (per-GPU microbatch={cfg.micro_batch_size} × seq_len={cfg.seq_len}, grad_accum={cfg.grad_accum_steps})"
        )
        log_if_main(
            f"[model] parameters ≈ {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
        )
    t0 = time.time()
    running_loss = torch.zeros((), device=device)
    while global_step < cfg.max_steps:
        offset = torch.randint(0, cfg.seq_len, (1,), device=device)
        if dist.is_initialized():
            dist.broadcast(offset, src=0)
        train_ds.set_offset(int(offset.item()))
        epoch = global_step // max(1, len(train_loader)) + 1
        train_sampler.set_epoch(epoch)
        log_if_main(f"[train] epoch {epoch:02d} with offset {offset.item()} starting ...")

        for it, chunk in enumerate(train_loader):
            last_micro = ((it + 1) % cfg.grad_accum_steps) == 0
            sync_ctx = (
                nullcontext() if (world_size == 1 or last_micro) else model.no_sync()
            )

            x = chunk.x.to(device, non_blocking=True)
            y = chunk.y.to(device, non_blocking=True)
            x_ctx = chunk_to_context(chunk, device)
            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    logits = model(x, x_ctx)
                    loss = (
                        F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            y.view(-1),
                            reduction="mean",
                            ignore_index=0,
                        )
                        / cfg.grad_accum_steps
                    )

                loss.backward()

            if last_micro:
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                running_loss += loss.detach() * cfg.grad_accum_steps

                if is_main_process() and (global_step % cfg.log_interval == 0):
                    avg_loss = (running_loss / cfg.log_interval).item()
                    lr_now = optimizer.param_groups[0]["lr"]
                    wandb.log(
                        {
                            "loss": avg_loss,
                            "lr": lr_now,
                        },
                        step=global_step,
                    )
                    running_loss.zero_()

                if (global_step % cfg.eval_interval == 0) or (
                    global_step == cfg.max_steps
                ):
                    model.eval()
                    val_nll, val_bpb = evaluate(model, val_loader, device, "bf16")
                    if is_main_process():
                        elapsed = time.time() - t0
                        log_if_main(
                            f"[eval @ step {global_step}] val_nll={val_nll:.4f}  val_bpb={val_bpb:.4f}  elapsed={elapsed / 60:.1f}m"
                        )
                        wandb.log(
                            {
                                "val_loss": val_nll,
                            },
                            step=global_step,
                        )
                    model.train()

                if (global_step % cfg.ckpt_interval == 0) or (
                    global_step == cfg.max_steps
                ):
                    if is_main_process():
                        save_ckpt(raw_model, optimizer, scheduler, global_step, cfg)

            if global_step >= cfg.max_steps:
                break
    model.eval()
    test_nll, test_bpb = evaluate(model, test_loader, device, "bf16")
    if is_main_process():
        log_if_main(f"[final test] nll={test_nll:.4f}  bpb={test_bpb:.4f}")
        wandb.log(
            {
                "test_loss": test_nll,
            },
            step=global_step,
        )
    ddp_cleanup()


def parse_args() -> Config:
    """Parse command-line flags into a Config dataclass.

    We expose the most useful knobs; for anything else, edit the dataclass defaults.
    """
    p = argparse.ArgumentParser(
        description="Train xLSTM (≈215M) on enwik8 with 2k context length."
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=Config.data_dir,
        help="Directory to cache/read enwik8.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=Config.out_dir,
        help="Directory for checkpoints & logs.",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=Config.seq_len,
        help="Context length (e.g., 2048).",
    )
    p.add_argument(
        "--micro_batch_size",
        type=int,
        default=Config.micro_batch_size,
        help="Per-GPU microbatch (sequences).",
    )
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=Config.grad_accum_steps,
        help="Gradient accumulation steps.",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=Config.max_steps,
        help="Number of optimizer steps.",
    )
    p.add_argument(
        "--log_interval",
        type=int,
        default=Config.log_interval,
        help="Number of optimizer steps between logs.",
    )
    p.add_argument(
        "--eval_interval",
        type=int,
        default=Config.eval_interval,
        help="Number of optimizer steps between evaluations.",
    )
    p.add_argument(
        "--ckpt_interval",
        type=int,
        default=Config.ckpt_interval,
        help="Number of optimizer steps between checkpoints.",
    )
    p.add_argument("--lr", type=float, default=Config.lr, help="Peak learning rate.")
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=Config.warmup_steps,
        help="Warmup steps before cosine.",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=Config.dropout,
        help="Dropout in residual blocks.",
    )
    p.add_argument(
        "--resume_path", type=str, default=None, help="Path to a checkpoint to resume."
    )
    p.add_argument(
        "--run_id", type=str, default=None, help="Wandb run ID to resume."
    )
    p.add_argument(
        "--amp_dtype",
        type=str,
        choices=["bf16", "fp16"],
        default=Config.amp_dtype,
        help="Mixed precision dtype.",
    )
    p.add_argument("--seed", type=int, default=Config.seed, help="Base random seed.")
    p.add_argument(
        "--num_blocks",
        type=int,
        default=Config.num_blocks,
        help="Number of xLSTM blocks.",
    )
    p.add_argument(
        "--tbptt",
        type=int,
        default=Config.tbptt,
        help="Detach interval. 0 disables TBPTT.",
    )
    p.add_argument(
        "--act_ckpt",
        type=int,
        default=Config.act_ckpt,
        help="0/1. Activation checkpointing across depth.",
    )
    p.add_argument(
        "--num_heads",
        type=int,
        default=Config.num_heads,
        help="Number of mLSTM heads.",
    )
    p.add_argument(
        "--tie_weights",
        type=int,
        default=Config.tie_weights,
        help="0/1. Tie weights between embedding and output layer.",
    )
    p.add_argument(
        "--logits_softcap",
        type=float,
        default=Config.logits_softcap,
        help="Softcap for logits.",
    )
    p.add_argument(
        "--dqk_factor",
        type=float,
        default=Config.dqk_factor,
        help="Factor for dqk head width.",
    )
    p.add_argument(
        "--gate_softcap",
        type=float,
        default=Config.gate_softcap,
        help="Softcap for gate.",
    )

    args = p.parse_args()

    # Build config from args; keep model shape as the 215M recipe.
    cfg = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        ckpt_interval=args.ckpt_interval,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        dropout=args.dropout,
        resume_path=args.resume_path,
        run_id=args.run_id,
        amp_dtype=args.amp_dtype,
        seed=args.seed,
        num_blocks=args.num_blocks,
        dqk_factor=args.dqk_factor,
        tie_weights=args.tie_weights,
        logits_softcap=args.logits_softcap,
        gate_softcap=args.gate_softcap,
        tbptt=args.tbptt,
        act_ckpt=args.act_ckpt,
        num_heads=args.num_heads,
    )
    return cfg


def main() -> None:
    cfg = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if is_main_process():
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)
    try:
        train(local_rank, world_size, cfg)
    finally:
        ddp_cleanup()


if __name__ == "__main__":
    main()
