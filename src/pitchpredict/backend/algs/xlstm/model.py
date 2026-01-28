# SPDX-License-Identifier: MIT
"""xLSTM model architecture for pitch prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from pitchpredict.backend.algs.xlstm.tokens import PitchToken


@dataclass
class ModelConfig:
    """Model architecture configuration (inference-only subset of training config)."""

    vocab_size: int = 258
    seq_len: int = 512
    d_model: int = 384
    num_blocks: int = 12
    num_heads: int = 8
    dqk_factor: float = 0.5
    dropout: float = 0.0
    denom_floor: float = 1.0
    gate_softcap: float = 15.0
    logits_softcap: float = 30.0
    tie_weights: bool = False
    eod_id: int = PitchToken.SESSION_END.value

    # Baseball-specific
    num_pitchers: int = 3000
    num_batters: int = 3700
    num_fielders: int = 3000


class MState(NamedTuple):
    """mLSTM recurrent state."""
    C: torch.Tensor  # [B, H, dqk, dhv] - memory matrix
    n: torch.Tensor  # [B, H, dqk] - normalizer
    m: torch.Tensor  # [B, H, 1] - max log input gate


class PackedPitchContext(NamedTuple):
    """Context features for each position in the sequence."""
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


def softcap(x: torch.Tensor, a: float) -> torch.Tensor:
    """Soft-cap activation: tanh(x/a) * a."""
    return torch.tanh(x / a) * a


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class mLSTMCellCore(nn.Module):
    """Core mLSTM cell with exponential gating."""

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
        self.head_ln = nn.LayerNorm(dhv)

    def _log_gate(self, pre: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "exp":
            return pre
        if kind == "sigmoid":
            return -F.softplus(-pre)
        raise ValueError(f"Unknown gate kind: {kind}")

    def step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        i_pre: torch.Tensor,
        f_pre: torch.Tensor,
        state: MState,
    ) -> tuple[torch.Tensor, MState]:
        """Single step update. Shapes: q,k:[B,H,dqk], v,o:[B,H,dhv], i_pre,f_pre:[B,H]"""
        C, n, m = state.C, state.n, state.m

        # Soft-cap gate pre-activations
        i_pre = softcap(i_pre, self.a_gate)
        f_pre = softcap(f_pre, self.a_gate)

        log_i = self._log_gate(i_pre, "exp")
        log_f = self._log_gate(f_pre, self.forget_gate)

        m_t = torch.maximum(log_f + m.squeeze(-1), log_i).unsqueeze(-1)
        i_s = torch.exp(log_i.unsqueeze(-1) - m_t)
        f_s = torch.exp(log_f.unsqueeze(-1) + m - m_t)

        # Write
        outer = torch.einsum("bhd,bhe->bhde", k, v)
        C = f_s.unsqueeze(-1) * C + i_s.unsqueeze(-1) * outer
        n = f_s * n + i_s * k

        # Read (scale q)
        q_scaled = q / math.sqrt(self.dqk)
        y = torch.einsum("bhde,bhd->bhe", C, q_scaled)
        den_n = torch.einsum("bhd,bhd->bh", n, q_scaled)
        den_m = torch.exp(-m_t.squeeze(-1))
        denom = torch.maximum(den_n.abs(), den_m).clamp(min=self.denom_floor).unsqueeze(-1)

        y = self.head_ln(y) / denom
        h = o * y
        return h.reshape(h.size(0), -1), MState(C=C, n=n, m=m_t)


class mLSTMLayer(nn.Module):
    """mLSTM layer with fused projections."""

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

        # Fused projections
        self.proj_qkv = nn.Linear(d_model, self.H * (2 * self.dqk + self.dhv), bias=True)
        self.proj_ifo = nn.Linear(d_model, self.H * (self.dhv + 2), bias=True)
        self.cell = mLSTMCellCore(self.H, self.dqk, self.dhv, forget_gate, denom_floor, gate_softcap)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def init_state(self, B: int, device: torch.device, dtype: torch.dtype) -> MState:
        return MState(
            C=torch.zeros(B, self.H, self.dqk, self.dhv, device=device, dtype=dtype),
            n=torch.zeros(B, self.H, self.dqk, device=device, dtype=dtype),
            m=torch.zeros(B, self.H, 1, device=device, dtype=dtype),
        )

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
            xb = x[:, t0:t1]
            chunk_len = t1 - t0

            chunk_reset = None
            if reset_mask is not None:
                chunk_reset = reset_mask[:, t0:t1]

            qkv = self.proj_qkv(xb).view(B, chunk_len, self.H, 2 * self.dqk + self.dhv).transpose(1, 2)
            ifo = self.proj_ifo(xb).view(B, chunk_len, self.H, self.dhv + 2).transpose(1, 2)

            q = qkv[..., :self.dqk]
            k = qkv[..., self.dqk:2 * self.dqk]
            v = qkv[..., 2 * self.dqk:]

            o = torch.sigmoid(ifo[..., :self.dhv])
            i_pre = ifo[..., self.dhv]
            f_pre = ifo[..., self.dhv + 1]

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
                        first_rel = int(torch.nonzero(any_mask, as_tuple=False)[0].item())
                        eod_cut = offset + first_rel + 1
                        seg_len = min(seg_len, eod_cut - offset)

                sl = slice(offset, offset + seg_len)
                seg_out, state = self._chunk_parallel(
                    q[..., sl, :], k[..., sl, :], v[..., sl, :],
                    o[..., sl, :], i_pre[..., sl], f_pre[..., sl], state
                )
                outputs.append(seg_out)
                tokens_seen += seg_len

                if (eod_cut is not None) and (eod_cut == offset + seg_len):
                    end_mask = chunk_reset[:, eod_cut - 1]  # type: ignore
                    if end_mask.any():
                        keep = (~end_mask).to(state.C.dtype)
                        state = MState(
                            C=state.C * keep.view(B, 1, 1, 1),
                            n=state.n * keep.view(B, 1, 1),
                            m=state.m * keep.view(B, 1, 1),
                        )

                offset += seg_len

                if detach_interval and (tokens_seen % detach_interval == 0):
                    state = MState(C=state.C.detach(), n=state.n.detach(), m=state.m.detach())

        y = torch.cat(outputs, dim=1) if outputs else x.new_zeros(B, 0, self.H * self.dhv)
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
        """Vectorized chunk computation."""
        cell = self.cell
        B, _, L, _ = q.shape

        i_pre = softcap(i_pre, cell.a_gate)
        f_pre = softcap(f_pre, cell.a_gate)
        log_i = cell._log_gate(i_pre, "exp")
        log_f_gate = cell._log_gate(f_pre, cell.forget_gate)

        S = torch.cumsum(log_f_gate, dim=-1)
        log_i_minus_S = log_i - S
        u = torch.cummax(log_i_minus_S, dim=-1)[0]
        u = torch.maximum(u, state.m)
        m = S + u

        m_prev = torch.cat([state.m, m[..., :-1]], dim=-1)
        log_i_s = log_i - m
        log_f_s = log_f_gate + m_prev - m

        i_s = torch.exp(log_i_s)
        log_f_prefix = torch.cumsum(log_f_s, dim=-1)
        decay = torch.exp(log_f_prefix)

        scale = 1.0 / math.sqrt(self.dqk)
        q_scaled = q * scale

        # Lower-triangular weights
        diff = log_f_prefix.unsqueeze(-1) - log_f_prefix.unsqueeze(-2)
        mask = torch.ones(diff.size(-2), diff.size(-1), device=diff.device, dtype=torch.bool).tril_()
        diff = diff.masked_fill(~mask, float("-inf"))
        diff = diff.clamp_max(0)

        A = torch.exp(diff)
        W = A * i_s.unsqueeze(-2)

        S_scores = torch.matmul(k, q.transpose(-2, -1)) * scale
        y_contrib = torch.matmul(W * S_scores.transpose(-2, -1), v)

        base_ctx = torch.matmul(q_scaled, state.C)
        y_raw = y_contrib + decay.unsqueeze(-1) * base_ctx

        N = torch.matmul(W, k) + decay.unsqueeze(-1) * state.n.unsqueeze(-2)

        den_n = (N * q_scaled).sum(dim=-1)
        den_m = torch.exp(-m)
        denom = torch.maximum(den_n.abs(), den_m).clamp(min=cell.denom_floor)

        h_hat = y_raw / denom.unsqueeze(-1)
        y = cell.head_ln(h_hat)
        h = o * y
        chunk_out = h.transpose(1, 2).reshape(B, L, self.H * self.dhv)

        # Final state
        f_prod = torch.exp(log_f_prefix[..., -1:])
        W_end = W[..., -1, :]
        weighted_v = W_end.unsqueeze(-1) * v
        C_delta = torch.matmul(k.transpose(-2, -1), weighted_v)
        C_last = f_prod.unsqueeze(-1) * state.C + C_delta
        n_last = N[..., -1, :]
        m_last = m[..., -1:]

        return chunk_out, MState(C=C_last, n=n_last, m=m_last)


class SwiGLU(nn.Module):
    """SwiGLU feedforward block."""

    def __init__(self, dim: int, proj_factor: float = 2.66):
        super().__init__()
        hidden = int(dim * proj_factor)
        self.up = nn.Linear(dim, hidden, bias=True)
        self.gate = nn.Linear(dim, hidden, bias=True)
        self.down = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)) * self.gate(x))


class xLSTMBlock(nn.Module):
    """xLSTM block: z = x + mLSTM(RMSNorm(x)); y = z + SwiGLU(RMSNorm(z))"""

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
        self.seqmix = mLSTMLayer(d_model, num_heads, dqk_factor, forget_gate, denom_floor, gate_softcap)
        self.drop = nn.Dropout(dropout)
        self.pre2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, proj_factor=2.66)
        self.detach_interval = detach_interval

    def forward(self, x: torch.Tensor, reset_mask: torch.BoolTensor | None = None) -> torch.Tensor:
        z = x + self.drop(self.seqmix(self.pre1(x), detach_interval=self.detach_interval, reset_mask=reset_mask)[0])
        y = z + self.drop(self.ff(self.pre2(z)))
        return y


class PitchContextAdapter(nn.Module):
    """Embeds pitch context features into d_model dimensions."""

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
        self.pitched_emb = nn.Embedding(num_pitchers + 1, 64, padding_idx=0)
        self.batter_emb = nn.Embedding(num_batters + 1, 64, padding_idx=0)
        self.fielder_emb = nn.Embedding(num_fielders + 1, 32, padding_idx=0)

        # State embeddings
        self.emb_p_throws = nn.Embedding(3, 8)
        self.emb_b_hits = nn.Embedding(4, 8)
        self.emb_balls = nn.Embedding(5, 8)
        self.emb_strikes = nn.Embedding(4, 8)
        self.emb_outs = nn.Embedding(4, 8)
        self.emb_order = nn.Embedding(5, 8)
        self.emb_bases = nn.Embedding(9, 16)
        self.emb_inning = nn.Embedding(25, 16)

        # cat_emb_dim: pitcher(64) + batter(64) + 8*fielder(32) + state embeddings
        self.cat_emb_dim = 64 + 64 + 8 * 32 + 8 + 8 + 8 + 8 + 8 + 8 + 16 + 16  # = 464

        # Continuous features
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

    def forward(self, ctx: PackedPitchContext) -> torch.Tensor:
        pid = ctx.pitcher_id % self.num_pitchers
        bid = ctx.batter_id % self.num_batters

        pe = self.pitched_emb(pid + 1)
        be = self.batter_emb(bid + 1)

        # Fielder embeddings
        f2 = self.fielder_emb((ctx.fielder_2_id % self.num_fielders) + 1)
        f3 = self.fielder_emb((ctx.fielder_3_id % self.num_fielders) + 1)
        f4 = self.fielder_emb((ctx.fielder_4_id % self.num_fielders) + 1)
        f5 = self.fielder_emb((ctx.fielder_5_id % self.num_fielders) + 1)
        f6 = self.fielder_emb((ctx.fielder_6_id % self.num_fielders) + 1)
        f7 = self.fielder_emb((ctx.fielder_7_id % self.num_fielders) + 1)
        f8 = self.fielder_emb((ctx.fielder_8_id % self.num_fielders) + 1)
        f9 = self.fielder_emb((ctx.fielder_9_id % self.num_fielders) + 1)

        te = self.emb_p_throws(ctx.pitcher_throws.clamp(0, 2))
        he = self.emb_b_hits(ctx.batter_hits.clamp(0, 3))
        ball_e = self.emb_balls(ctx.count_balls.clamp(0, 4))
        str_e = self.emb_strikes(ctx.count_strikes.clamp(0, 3))
        out_e = self.emb_outs(ctx.outs.clamp(0, 3))
        base_e = self.emb_bases(ctx.bases_state.clamp(0, 8))
        order_e = self.emb_order(ctx.number_through_order.clamp(0, 4))
        inn_e = self.emb_inning(ctx.inning.clamp(0, 24))

        cont_input = torch.stack([
            ctx.pitcher_age,
            ctx.batter_age,
            ctx.score_bat,
            ctx.score_fld,
            ctx.pitch_number,
            ctx.game_date,
            ctx.batter_days_since_prev_game.float(),
            ctx.pitcher_days_since_prev_game.float(),
            ctx.strike_zone_top,
            ctx.strike_zone_bottom,
        ], dim=-1)

        cont_feats = self.cont_proj(cont_input)

        combined = torch.cat([
            pe, be, f2, f3, f4, f5, f6, f7, f8, f9,
            te, he, ball_e, str_e, out_e, order_e, base_e, inn_e,
            cont_feats,
        ], dim=-1)

        return self.fusion(combined)


class FusionLayer(nn.Module):
    """Combines token embeddings with context embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_seq, x_ctx], dim=-1)
        out = self.proj(combined)
        out = self.norm(out)
        out = self.dropout(out)
        return out


class BaseballxLSTM(nn.Module):
    """xLSTM model for baseball pitch prediction."""

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

        self.blocks = nn.ModuleList([
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
            for _ in range(num_blocks)
        ])

        self.norm_out = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        self.eod_id = eod_id

    def forward(self, x_seq_ids: torch.LongTensor, x_ctx: PackedPitchContext) -> torch.Tensor:
        """Forward pass.

        Args:
            x_seq_ids: Token IDs [B, T]
            x_ctx: Context features (PackedPitchContext with [B, T] tensors)

        Returns:
            Logits [B, T, vocab_size]
        """
        x_seq = self.token_embed(x_seq_ids)
        x_ctx_emb = self.context_adapter(x_ctx)
        x = self.fusion(x_seq, x_ctx_emb)

        for blk in self.blocks:
            if self.act_ckpt and self.training:
                x = torch_checkpoint(lambda _x: blk(_x), x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm_out(x)
        logits = self.lm_head(x)

        if self.logits_softcap is not None and self.logits_softcap > 0:
            logits = softcap(logits, self.logits_softcap)

        return logits


def init_gate_biases(model: nn.Module) -> None:
    """Initialize gate biases for stable training.

    - Input gate bias: -10 (sparse writes initially)
    - Forget gate bias: logit(0.95) for sigmoid gate
    - Output gate bias: +1
    """
    def logit(p: float) -> float:
        return float(math.log(p / (1.0 - p)))

    for layer in model.modules():
        if isinstance(layer, mLSTMLayer) and layer.proj_ifo.bias is not None:
            with torch.no_grad():
                bias = layer.proj_ifo.bias.view(layer.H, layer.dhv + 2)
                bias[:, :layer.dhv].fill_(+1.0)  # output gate
                bias[:, layer.dhv].fill_(-10.0)  # input gate
                forget_bias = logit(0.95) if layer.cell.forget_gate == "sigmoid" else 0.0
                bias[:, layer.dhv + 1].fill_(forget_bias)


def build_model(cfg: ModelConfig, device: torch.device | str = "cpu") -> BaseballxLSTM:
    """Build and initialize an xLSTM model from config."""
    model = BaseballxLSTM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        dqk_factor=cfg.dqk_factor,
        denom_floor=cfg.denom_floor,
        gate_softcap=cfg.gate_softcap,
        dropout=cfg.dropout,
        detach_interval=0,  # No TBPTT for inference
        act_ckpt=False,  # No activation checkpointing for inference
        tie_weights=cfg.tie_weights,
        logits_softcap=cfg.logits_softcap,
        eod_id=cfg.eod_id,
        num_pitchers=cfg.num_pitchers,
        num_batters=cfg.num_batters,
        num_fielders=cfg.num_fielders,
    )
    init_gate_biases(model)
    model.to(device)
    return model
