from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def save_run(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    epoch: int,
    metrics: dict[str, float],
) -> Path:
    """Save everything needed to inspect or resume a training run."""
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "resolved_config.yaml"
    OmegaConf.save(config=cfg, f=config_path, resolve=True)

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    save_json(run_dir / "metrics.json", metrics)
    return checkpoint_path


def load_run(run_dir: str | Path, device: torch.device):
    """Return the saved configuration and checkpoint from a run directory."""
    run_dir = Path(run_dir)
    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "checkpoint.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing saved configuration: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    cfg = OmegaConf.load(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return cfg, checkpoint
