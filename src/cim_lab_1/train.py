from __future__ import annotations

from pathlib import Path
import random

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .data import build_loaders
from .models import build_model, count_parameters
from .run_io import save_json, save_run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Hydra creates a fresh directory for this run and stores its own config
    # snapshot under run_dir/.hydra. We also save a resolved, easy-to-find copy.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loaders(
        cfg.data_dir,
        cfg.augmentation,
        cfg.training.batch_size,
        cfg.training.num_workers,
    )

    model = build_model(cfg.model).to(device)
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg.training)

    wandb_run = None
    if cfg.logging.use_wandb:
        import wandb

        display_name = None if cfg.run.name == "auto" else cfg.run.name
        wandb_run = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=display_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(run_dir),
            job_type="training",
        )
        save_json(
            run_dir / "wandb_run.json",
            {
                "id": wandb_run.id,
                "name": wandb_run.name,
                "project": wandb_run.project,
                "entity": wandb_run.entity,
            },
        )

    final_metrics = {}
    for epoch in range(cfg.training.epochs):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, device, optimizer=optimizer
        )
        test_loss, test_accuracy = run_epoch(model, test_loader, loss_fn, device)
        final_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        print(final_metrics)
        if wandb_run is not None:
            wandb_run.log(final_metrics)

    checkpoint_path = save_run(
        run_dir=run_dir,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        epoch=cfg.training.epochs,
        metrics=final_metrics,
    )
    print(f"Saved run to {run_dir}")

    if wandb_run is not None:
        wandb_run.summary.update(final_metrics)
        wandb_run.summary["local_run_dir"] = str(run_dir)

        if cfg.logging.log_model_artifact:
            import wandb

            artifact = wandb.Artifact(
                name=f"{cfg.model.name}-{wandb_run.id}",
                type="model",
                metadata={"run_name": cfg.run.name, **final_metrics},
            )
            artifact.add_file(str(checkpoint_path), name="checkpoint.pt")
            artifact.add_file(str(run_dir / "resolved_config.yaml"), name="resolved_config.yaml")
            artifact.add_file(str(run_dir / "metrics.json"), name="metrics.json")
            wandb_run.log_artifact(artifact)

        wandb_run.finish()


if __name__ == "__main__":
    main()
