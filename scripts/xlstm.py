import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import NamedTuple, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint as ckpt
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb # type: ignore

from pitchpredict.backend.algs.deep.nn import PitchToken
from pitchpredict.backend.algs.deep.dataset import PackedPitchDataset, PackedPitchChunk, PackedPitchContext, chunk_to_context

"""
uv run torchrun --standalone --nproc_per_node=6 scripts/xlstm.py \
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


def setup_wandb(cfg: "Config") -> None:
    if is_main_process():
        wandb.init(project="baseball-xlstm", config=(asdict(cfg)), id=cfg.run_id, resume=("never" if cfg.run_id is None else "allow"))
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

class mLSTMCellCore(nn.Module):
    def __init__(
        self,
        H: int,
        dqk: int,
        dhv: int,
        forget_gate: str = "sigmoid",
        denom_floor: float = 1.0,
        a_gate: float = 15.0,
    ):
        super().__init__()
        self.H, self.dqk, self.dhv = H, dqk, dhv
        self.forget_gate = forget_gate
        self.denom_floor = float(denom_floor)
        self.a_gate = float(a_gate)
        self.head_ln = nn.LayerNorm(dhv)  # head-wise norm

    def _log_gate(self, pre: torch.Tensor, kind: str) -> torch.Tensor:  # pre: [B,H]
        if kind == "exp":
            return pre
        if kind == "sigmoid":
            return -F.softplus(-pre)
        raise ValueError

    def step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        i_pre: torch.Tensor,
        f_pre: torch.Tensor,
        state: "MState",
    ) -> tuple[torch.Tensor, "MState"]:
        # shapes: q,k:[B,H,dqk], v,o:[B,H,dhv], i_pre,f_pre:[B,H]
        C, n, m = (
            state.C,
            state.n,
            state.m,
        )  # [B,H,dqk,dhv], [B,H,dqk], [B,H,1]

        # soft-cap gate pre-activations
        i_pre = softcap(i_pre, self.a_gate)
        f_pre = softcap(f_pre, self.a_gate)

        log_i = self._log_gate(i_pre, "exp")  # [B,H]
        log_f = self._log_gate(f_pre, self.forget_gate)  # [B,H]

        m_t = torch.maximum(log_f + m.squeeze(-1), log_i).unsqueeze(-1)  # [B,H,1]
        i_s = torch.exp(log_i.unsqueeze(-1) - m_t)  # [B,H,1]
        f_s = torch.exp(log_f.unsqueeze(-1) + m - m_t)  # [B,H,1]

        # write
        outer = torch.einsum("bhd,bhe->bhde", k, v)  # [B,H,dqk,dhv]
        C = f_s.unsqueeze(-1) * C + i_s.unsqueeze(-1) * outer
        n = f_s * n + i_s * k  # [B,H,dqk]

        # read (scale q)
        q_scaled = q / math.sqrt(self.dqk)
        y = torch.einsum("bhde,bhd->bhe", C, q_scaled)  # [B,H,dhv]
        den_n = torch.einsum("bhd,bhd->bh", n, q_scaled)  # [B,H]
        den_m = torch.exp(-m_t.squeeze(-1))  # [B,H]
        denom = (
            torch.maximum(den_n.abs(), den_m).clamp(min=self.denom_floor).unsqueeze(-1)
        )

        y = self.head_ln(y) / denom
        h = o * y  # [B,H,dhv]
        return h.reshape(h.size(0), -1), MState(C=C, n=n, m=m_t)


class MState(NamedTuple):
    C: torch.Tensor
    n: torch.Tensor
    m: torch.Tensor


class mLSTMLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        qk_ratio: float = 0.5,
        forget_gate: str = "sigmoid",
        denom_floor: float = 1.0,
        gate_softcap: float = 15.0,
    ):
        super().__init__()
        self.H = num_heads
        self.dhv = d_model // num_heads
        self.dqk = max(1, int(round(self.dhv * qk_ratio)))
        # Fuse projections to reduce kernel launches
        self.proj_qkv = nn.Linear(
            d_model, self.H * (2 * self.dqk + self.dhv), bias=True
        )
        self.proj_ifo = nn.Linear(
            d_model, self.H * (self.dhv + 2), bias=True
        )  # [o (H*dhv), i (H), f (H)]
        self.cell = mLSTMCellCore(
            self.H, self.dqk, self.dhv, forget_gate, denom_floor, gate_softcap
        )  # core updates, no linears
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def init_state(self, B: int, device: torch.device, dtype: torch.dtype) -> MState:
        return MState(
            C=torch.zeros(B, self.H, self.dqk, self.dhv, device=device, dtype=dtype),
            n=torch.zeros(B, self.H, self.dqk, device=device, dtype=dtype),
            m=torch.zeros(B, self.H, 1, device=device, dtype=dtype),
        )

    @torch.no_grad()
    def _shape_notes(self) -> None:  # for reference
        # proj_qkv: [B,L, H*(2*dqk+dhv)] -> split -> [B,H,L,dqk/dhv]
        pass

    def forward(
        self,
        x: torch.Tensor,
        detach_interval: int = 0,
        proj_chunk: int = 4096,
        reset_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, MState]:
        B, T, _ = x.shape
        state = self.init_state(B, x.device, x.dtype)
        outputs: list[torch.Tensor] = []
        tokens_seen = 0

        for t0 in range(0, T, proj_chunk):
            t1 = min(T, t0 + proj_chunk)
            xb = x[:, t0:t1]  # [B,L,D], L<=proj_chunk
            chunk_len = t1 - t0

            chunk_reset = None
            if reset_mask is not None:
                chunk_reset = reset_mask[:, t0:t1]  # [B,L]

            qkv = (
                self.proj_qkv(xb)
                .view(B, chunk_len, self.H, 2 * self.dqk + self.dhv)
                .transpose(1, 2)
            )
            ifo = (
                self.proj_ifo(xb)
                .view(B, chunk_len, self.H, self.dhv + 2)
                .transpose(1, 2)
            )

            q = qkv[..., : self.dqk]  # [B,H,L,dqk]
            k = qkv[..., self.dqk : 2 * self.dqk]  # [B,H,L,dqk]
            v = qkv[..., 2 * self.dqk :]  # [B,H,L,dhv]

            o = torch.sigmoid(ifo[..., : self.dhv])  # [B,H,L,dhv]
            i_pre = ifo[..., self.dhv]  # [B,H,L]
            f_pre = ifo[..., self.dhv + 1]  # [B,H,L]

            offset = 0
            while offset < chunk_len:
                seg_len = chunk_len - offset
                if detach_interval:
                    steps_to_detach = detach_interval - (tokens_seen % detach_interval)
                    if steps_to_detach == detach_interval:
                        steps_to_detach = detach_interval
                    seg_len = min(seg_len, steps_to_detach)

                eod_cut = None
                if chunk_reset is not None:
                    any_mask = chunk_reset[:, offset:].any(dim=0)
                    if any_mask.any():
                        first_rel = int(
                            torch.nonzero(any_mask, as_tuple=False)[0].item()
                        )
                        eod_cut = offset + first_rel + 1
                        seg_len = min(seg_len, eod_cut - offset)

                sl = slice(offset, offset + seg_len)
                seg_out, state = self._chunk_parallel(
                    q[..., sl, :],
                    k[..., sl, :],
                    v[..., sl, :],
                    o[..., sl, :],
                    i_pre[..., sl],
                    f_pre[..., sl],
                    state,
                )
                outputs.append(seg_out)
                tokens_seen += seg_len

                if (eod_cut is not None) and (eod_cut == offset + seg_len):
                    end_mask = chunk_reset[:, eod_cut - 1]  # type: ignore
                    if end_mask.any():
                        keep = (~end_mask).to(state.C.dtype)
                        keep_C = keep.view(B, 1, 1, 1)
                        keep_n = keep.view(B, 1, 1)
                        keep_m = keep.view(B, 1, 1)
                        state = MState(
                            C=state.C * keep_C,
                            n=state.n * keep_n,
                            m=state.m * keep_m,
                        )

                offset += seg_len

                if detach_interval and (tokens_seen % detach_interval == 0):
                    state = MState(
                        C=state.C.detach(), n=state.n.detach(), m=state.m.detach()
                    )

        y = (
            torch.cat(outputs, dim=1)
            if outputs
            else x.new_zeros(B, 0, self.H * self.dhv)
        )
        y = self.out_proj(y)
        return y, state

    def _chunk_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        i_pre: torch.Tensor,
        f_pre: torch.Tensor,
        state: MState,
    ) -> tuple[torch.Tensor, MState]:
        """
        Vectorized chunk computation that replaces the inner time-step loop.
        Shapes: q,k:[B,H,L,dqk], v,o:[B,H,L,dhv], i_pre,f_pre:[B,H,L]
        """
        cell = self.cell
        B, _, L, _ = q.shape
        assert L > 0, "Chunk length must be positive"

        i_pre = softcap(i_pre, cell.a_gate)
        f_pre = softcap(f_pre, cell.a_gate)
        log_i = cell._log_gate(i_pre, "exp")  # [B,H,L]
        log_f_gate = cell._log_gate(f_pre, cell.forget_gate)  # [B,H,L]

        S = torch.cumsum(log_f_gate, dim=-1)
        log_i_minus_S = log_i - S
        u = torch.cummax(log_i_minus_S, dim=-1)[0]
        u = torch.maximum(u, state.m)
        m = S + u  # [B,H,L]

        m_prev = torch.cat([state.m, m[..., :-1]], dim=-1)
        log_i_s = log_i - m
        log_f_s = log_f_gate + m_prev - m

        i_s = torch.exp(log_i_s)
        log_f_prefix = torch.cumsum(log_f_s, dim=-1)  # log prod of f_s up to t
        decay = torch.exp(log_f_prefix)

        scale = 1.0 / math.sqrt(self.dqk)
        q_scaled = q * scale

        # Lower-triangular weights
        diff = log_f_prefix.unsqueeze(-1) - log_f_prefix.unsqueeze(-2)  # [B,H,L,L]

        # 1) Hard-mask the upper triangle *before* exp to avoid overflow
        mask = torch.ones(
            diff.size(-2), diff.size(-1), device=diff.device, dtype=torch.bool
        ).tril_()
        diff = diff.masked_fill(~mask, float("-inf"))

        # 2) Be extra safe: in the valid region, any tiny positive noise becomes 0
        diff = diff.clamp_max(0)

        A = torch.exp(diff)  # exp(-inf)=0; no overflow anywhere
        W = A * i_s.unsqueeze(-2)  # [B,H,L,L] * [B,H,1,L] -> [B,H,L,L]

        S_scores = torch.matmul(k, q.transpose(-2, -1)) * scale  # [B,H,L,L] (j,i)
        y_contrib = torch.matmul(W * S_scores.transpose(-2, -1), v)  # [B,H,L,dhv]

        base_ctx = torch.matmul(q_scaled, state.C)  # [B,H,L,dhv]
        y_raw = y_contrib + decay.unsqueeze(-1) * base_ctx

        N = torch.matmul(W, k)  # [B,H,L,dqk]
        N = N + decay.unsqueeze(-1) * state.n.unsqueeze(-2)

        den_n = (N * q_scaled).sum(dim=-1)  # [B,H,L]
        den_m = torch.exp(-m)
        denom = torch.maximum(den_n.abs(), den_m).clamp(min=cell.denom_floor)

        h_hat = y_raw / denom.unsqueeze(-1)
        y = cell.head_ln(h_hat)
        h = o * y  # [B,H,L,dhv]
        chunk_out = h.transpose(1, 2).reshape(B, L, self.H * self.dhv)

        f_prod = torch.exp(log_f_prefix[..., -1:])
        W_end = W[..., -1, :]  # [B,H,L]
        weighted_v = W_end.unsqueeze(-1) * v
        C_delta = torch.matmul(k.transpose(-2, -1), weighted_v)  # [B,H,dqk,dhv]
        C_last = f_prod.unsqueeze(-1) * state.C + C_delta
        n_last = N[..., -1, :]  # [B,H,dqk]
        m_last = m[..., -1:]  # [B,H,1]

        next_state = MState(C=C_last, n=n_last, m=m_last)
        return chunk_out, next_state


EPS = 1e-8


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def softcap(x: torch.Tensor, a: float) -> torch.Tensor:
    return torch.tanh(x / a) * a


def logit(p: float) -> float:
    """Inverse of the logistic sigmoid for initializing biases: log(p/(1-p))."""
    return float(math.log(p / (1.0 - p)))


class SwiGLU(nn.Module):
    # projection factor ≈ 2.66 as in Fig. 8; you can tune
    def __init__(self, dim: int, proj_factor: float = 2.66):
        super().__init__()
        hidden = int(dim * proj_factor)
        self.up = nn.Linear(dim, hidden, bias=True)
        self.gate = nn.Linear(dim, hidden, bias=True)
        self.down = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)) * self.gate(x))


class xLSTMBlock(nn.Module):
    """
    Post up-projection: z = x + mLSTM(RMSNorm(x)); y = z + SwiGLU(RMSNorm(z))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dqk_factor: float = 0.5,
        forget_gate: str = "sigmoid",
        denom_floor: float = 1.0,
        gate_softcap: float = 15.0,
        dropout: float = 0.0,
        detach_interval: int = 0,
    ):
        super().__init__()
        self.pre1 = RMSNorm(d_model)
        self.seqmix = mLSTMLayer(
            d_model, num_heads, dqk_factor, forget_gate, denom_floor, gate_softcap
        )
        self.drop = nn.Dropout(dropout)
        self.pre2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, proj_factor=2.66)
        self.detach_interval = detach_interval

    def forward(
        self, x: torch.Tensor, reset_mask: torch.BoolTensor | None = None
    ) -> torch.Tensor:
        z = x + self.drop(
            self.seqmix(
                self.pre1(x),
                detach_interval=self.detach_interval,
                reset_mask=reset_mask,
            )[0]
        )
        y = z + self.drop(self.ff(self.pre2(z)))
        return y

class PitchContextAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_pitchers: int,
        num_batters: int,
        num_fielders: int = 4000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_pitchers = num_pitchers
        self.num_batters = num_batters
        self.num_fielders = num_fielders

        # Entity embeddings
        self.pitched_emb = nn.Embedding(num_pitchers+1, 64, padding_idx=0)
        self.batter_emb = nn.Embedding(num_batters+1, 64, padding_idx=0)
        self.fielder_emb = nn.Embedding(num_fielders+1, 32, padding_idx=0)  # shared for all fielder positions

        # State embeddings
        self.emb_p_throws = nn.Embedding(3, 8) # 0, 1, 2
        self.emb_b_hits = nn.Embedding(4, 8) # 0, 1, 2, 3
        self.emb_balls = nn.Embedding(5, 8) # 0-4
        self.emb_strikes = nn.Embedding(4, 8) # 0-3
        self.emb_outs = nn.Embedding(4, 8) # 0-3
        self.emb_order = nn.Embedding(5, 8) # 0-4
        self.emb_bases = nn.Embedding(9, 16) # 0-7 (bitmask) + pad
        self.emb_inning = nn.Embedding(25, 16) # 1-24 (clip larger)

        # cat_emb_dim: pitcher(64) + batter(64) + 8*fielder(32) + p_throws(8) + b_hits(8) + balls(8) + strikes(8) + outs(8) + order(8) + bases(16) + inning(16)
        self.cat_emb_dim = 64 + 64 + 8*32 + 8 + 8 + 8 + 8 + 8 + 8 + 16 + 16  # = 464

        # Continuous: pitcher_age, batter_age, score_bat, score_fld, pitch_number, game_date,
        #             batter_days_since_prev_game, pitcher_days_since_prev_game, strike_zone_top, strike_zone_bottom
        self.num_continuous = 10

        self.cont_proj = nn.Sequential(
            nn.Linear(self.num_continuous, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + self.cat_emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, chunk: PackedPitchContext) -> torch.Tensor:

        pid = chunk.pitcher_id % self.num_pitchers
        bid = chunk.batter_id % self.num_batters

        pe = self.pitched_emb(pid+1)
        be = self.batter_emb(bid+1)

        # Fielder embeddings (positions 2-9)
        f2 = self.fielder_emb((chunk.fielder_2_id % self.num_fielders) + 1)
        f3 = self.fielder_emb((chunk.fielder_3_id % self.num_fielders) + 1)
        f4 = self.fielder_emb((chunk.fielder_4_id % self.num_fielders) + 1)
        f5 = self.fielder_emb((chunk.fielder_5_id % self.num_fielders) + 1)
        f6 = self.fielder_emb((chunk.fielder_6_id % self.num_fielders) + 1)
        f7 = self.fielder_emb((chunk.fielder_7_id % self.num_fielders) + 1)
        f8 = self.fielder_emb((chunk.fielder_8_id % self.num_fielders) + 1)
        f9 = self.fielder_emb((chunk.fielder_9_id % self.num_fielders) + 1)

        te = self.emb_p_throws(chunk.pitcher_throws.clamp(0, 2))
        he = self.emb_b_hits(chunk.batter_hits.clamp(0, 3))
        ball_e = self.emb_balls(chunk.count_balls.clamp(0, 4))
        str_e = self.emb_strikes(chunk.count_strikes.clamp(0, 3))
        out_e = self.emb_outs(chunk.outs.clamp(0, 3))
        base_e = self.emb_bases(chunk.bases_state.clamp(0, 8))
        order_e = self.emb_order(chunk.number_through_order.clamp(0, 4))
        inn_e = self.emb_inning(chunk.inning.clamp(0, 24))

        cont_input = torch.stack([
            chunk.pitcher_age,
            chunk.batter_age,
            chunk.score_bat,
            chunk.score_fld,
            chunk.pitch_number,
            chunk.game_date,
            chunk.batter_days_since_prev_game.float(),
            chunk.pitcher_days_since_prev_game.float(),
            chunk.strike_zone_top,
            chunk.strike_zone_bottom,
        ], dim=-1)

        cont_feats = self.cont_proj(cont_input)

        combined = torch.cat([
            pe, be,
            f2, f3, f4, f5, f6, f7, f8, f9,
            te, he, ball_e, str_e, out_e, order_e, base_e, inn_e,
            cont_feats,
        ], dim=-1)

        return self.fusion(combined)

class FusionLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_model*2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_seq: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_seq, x_ctx], dim=-1)

        out = self.proj(combined)
        out = self.norm(out)
        out = self.dropout(out)
        return out
        

class BaseballxLSTM(nn.Module):
    """
    All-mLSTM, post-up blocks. Operates mLSTM at d_model.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dqk_factor: float = 0.5,
        dropout: float = 0.0,
        denom_floor: float = 1.0,
        gate_softcap: float = 15.0,
        detach_interval: int = 0,
        act_ckpt: bool = False,
        tie_weights: bool = False,
        logits_softcap: float = 30.0,
        eod_id: int = 0xFB,
        num_pitchers: int = 1730,
        num_batters: int = 1923,
        num_fielders: int = 4000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.act_ckpt = bool(act_ckpt)
        self.logits_softcap = logits_softcap

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.context_adapter = PitchContextAdapter(d_model, num_pitchers, num_batters, num_fielders)
        self.fusion = FusionLayer(d_model)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                xLSTMBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dqk_factor=dqk_factor,
                    forget_gate="sigmoid",
                    denom_floor=denom_floor,
                    gate_softcap=gate_softcap,
                    dropout=dropout,
                    detach_interval=detach_interval,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_embed.weight  # paper did not tie; keep toggle
        self.eod_id = eod_id

    def forward(self, x_seq_ids: torch.LongTensor, x_ctx: PackedPitchContext) -> torch.Tensor:
        x_seq = self.token_embed(x_seq_ids)
        x_ctx = self.context_adapter(x_ctx)
        x = self.fusion(x_seq, x_ctx)
        for blk in self.blocks:
            if self.act_ckpt and self.training:
                # checkpoint expects a fn taking tensors; thread mask explicitly
                x = ckpt(
                    lambda _x: blk(_x),
                    x,
                    use_reentrant=False,
                )
            else:
                x = blk(x)
        x = self.norm_out(x)
        logits = self.lm_head(x)
        if self.logits_softcap is not None and self.logits_softcap > 0:
            logits = softcap(logits, self.logits_softcap)  # a = 30 in paper
        return logits


def init_gate_biases_v2(model: nn.Module) -> None:
    """
    - set input gate bias ≈ -10 (Sec. 3.2; Fig. 11)
    - set forget gate bias to logit(0.95) if sigmoid; 0 if exp
    - set output gate bias +1
    """

    def logit(p: float) -> float:
        return float(math.log(p / (1.0 - p)))

    for layer in model.modules():
        if isinstance(layer, mLSTMLayer) and layer.proj_ifo.bias is not None:
            with torch.no_grad():
                bias = layer.proj_ifo.bias.view(layer.H, layer.dhv + 2)
                bias[:, : layer.dhv].fill_(+1.0)  # output gate
                bias[:, layer.dhv].fill_(-10.0)  # input gate
                forget_bias = (
                    logit(0.95) if layer.cell.forget_gate == "sigmoid" else 0.0
                )
                bias[:, layer.dhv + 1].fill_(forget_bias)


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
    init_gate_biases_v2(model)
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
