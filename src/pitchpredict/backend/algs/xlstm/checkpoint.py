# SPDX-License-Identifier: MIT
"""Checkpoint loading with HuggingFace Hub integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from pitchpredict.backend.algs.xlstm.model import BaseballxLSTM, ModelConfig, build_model, init_gate_biases

# Environment variable names
ENV_XLSTM_REPO = "PITCHPREDICT_XLSTM_REPO"
ENV_XLSTM_REVISION = "PITCHPREDICT_XLSTM_REVISION"
ENV_MODEL_DIR = "PITCHPREDICT_MODEL_DIR"
ENV_XLSTM_PATH = "PITCHPREDICT_XLSTM_PATH"
ENV_DEVICE = "PITCHPREDICT_DEVICE"

# Defaults
DEFAULT_REPO = "baseball-analytica/pitchpredict-xlstm"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pitchpredict" / "xlstm"

_MISSING_XLSTM_DEPS = (
    "Unable to load xLSTM weights. Ensure network access for Hugging Face downloads, "
    "or set PITCHPREDICT_XLSTM_PATH to a local checkpoint directory containing "
    "model.safetensors and config.json."
)


def _download_hf_file(
    filename: str,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a single file from HuggingFace Hub with file locking."""
    repo_id = repo_id or os.environ.get(ENV_XLSTM_REPO, DEFAULT_REPO)
    revision = revision or os.environ.get(ENV_XLSTM_REVISION)
    cache_dir = Path(cache_dir or os.environ.get(ENV_MODEL_DIR, DEFAULT_CACHE_DIR))

    try:
        from huggingface_hub import hf_hub_download
        from filelock import FileLock
    except ImportError as e:
        raise ImportError(_MISSING_XLSTM_DEPS) from e

    cache_dir.mkdir(parents=True, exist_ok=True)

    lock_path = cache_dir / f"{filename}.lock"
    with FileLock(str(lock_path)):
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=str(cache_dir),
                local_dir=str(cache_dir),
            )
            return Path(local_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download {filename} from {repo_id}. "
                f"Error: {e}\n"
                f"You can manually download and set {ENV_XLSTM_PATH}."
            ) from e


def _resolve_local_path(checkpoint_path: str | Path | None = None) -> Path | None:
    """Check explicit path and env var. Returns None if neither is set."""
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        return path

    env_path = os.environ.get(ENV_XLSTM_PATH)
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ENV_XLSTM_PATH}={path}")
        return path

    return None


def _load_safetensors(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    **hf_kwargs: Any,
) -> tuple[dict[str, Any], ModelConfig]:
    """Load model weights from safetensors + config.json.

    Resolution priority:
    1. Explicit checkpoint_path (directory containing model.safetensors + config.json)
    2. PITCHPREDICT_XLSTM_PATH env var (same)
    3. Download from HuggingFace Hub
    """
    local = _resolve_local_path(checkpoint_path)

    if local is not None:
        # Local path: expect a directory with both files
        if local.is_dir():
            weights_path = local / "model.safetensors"
            config_path = local / "config.json"
        else:
            # Pointed directly at the safetensors file
            weights_path = local
            config_path = local.parent / "config.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in {local}")
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found alongside {weights_path}")
    else:
        # Download both files from HF
        weights_path = _download_hf_file("model.safetensors", **hf_kwargs)
        config_path = _download_hf_file("config.json", **hf_kwargs)

    # Load config
    with open(config_path) as f:
        ckpt_config = json.load(f)

    cfg = ModelConfig(
        vocab_size=ckpt_config.get("vocab_size", ModelConfig.vocab_size),
        seq_len=ckpt_config.get("seq_len", ModelConfig.seq_len),
        d_model=ckpt_config.get("d_model", ModelConfig.d_model),
        num_blocks=ckpt_config.get("num_blocks", ModelConfig.num_blocks),
        num_heads=ckpt_config.get("num_heads", ModelConfig.num_heads),
        dqk_factor=ckpt_config.get("dqk_factor", ModelConfig.dqk_factor),
        dropout=ckpt_config.get("dropout", ModelConfig.dropout),
        denom_floor=ckpt_config.get("denom_floor", ModelConfig.denom_floor),
        gate_softcap=ckpt_config.get("gate_softcap", ModelConfig.gate_softcap),
        logits_softcap=ckpt_config.get("logits_softcap", ModelConfig.logits_softcap),
        tie_weights=ckpt_config.get("tie_weights", ModelConfig.tie_weights),
        eod_id=ckpt_config.get("eod_id", ModelConfig.eod_id),
        num_pitchers=ckpt_config.get("num_pitchers", ModelConfig.num_pitchers),
        num_batters=ckpt_config.get("num_batters", ModelConfig.num_batters),
        num_fielders=ckpt_config.get("num_fielders", ModelConfig.num_fielders),
    )

    # Load weights
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError(_MISSING_XLSTM_DEPS) from e
    state_dict = load_file(str(weights_path), device=str(device))

    return state_dict, cfg


def load_model(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    **resolve_kwargs: Any,
) -> tuple[BaseballxLSTM, ModelConfig]:
    """Load xLSTM model from checkpoint.

    Loads model weights from safetensors format and config from JSON.

    Args:
        checkpoint_path: Path to directory containing model.safetensors + config.json,
            or path to model.safetensors directly. If None, downloads from HuggingFace.
        device: Device to load model on (default: cuda if available, else cpu)
        **resolve_kwargs: Additional arguments (repo_id, revision, cache_dir)

    Returns:
        Tuple of (model, config)
    """
    # Resolve device
    if device is None:
        env_device = os.environ.get(ENV_DEVICE)
        if env_device:
            device = torch.device(env_device)
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    state_dict, cfg = _load_safetensors(checkpoint_path, device=device, **resolve_kwargs)

    # Build model and load weights
    model = build_model(cfg, device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing}")

    model.eval()
    return model, cfg
