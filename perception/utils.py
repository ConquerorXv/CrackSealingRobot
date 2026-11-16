from __future__ import annotations

from pathlib import Path
import torch


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_weights(model: torch.nn.Module, path: str | Path, device: torch.device) -> None:
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Expected a LiteCMNet checkpoint dict with key 'model'")
    model.load_state_dict(ckpt["model"], strict=True)
