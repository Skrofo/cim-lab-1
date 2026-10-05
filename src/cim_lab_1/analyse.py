from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2

from .metrics import accuracy, prediction_consistency, representation_similarity
from .models import build_model
from .run_io import load_run
from .transforms import rotate, scale_contrast, translate


def load_model_from_run(run_dir: str, device: torch.device):
    saved_cfg, checkpoint = load_run(to_absolute_path(run_dir), device)
    model = build_model(saved_cfg.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, saved_cfg


def analyse_transform(model, loader, device, transform_fn, strengths):
    results = {"strength": [], "accuracy": [], "consistency": [], "similarity": []}

    for strength in strengths:
        batch_accuracy = []
        batch_consistency = []
        batch_similarity = []

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            transformed_images = transform_fn(images, strength)

            with torch.inference_mode():
                reference_logits = model(images)
                transformed_logits = model(transformed_images)
                reference_features = model.forward_features(images)
                transformed_features = model.forward_features(transformed_images)

            batch_accuracy.append(accuracy(transformed_logits, labels).item())
            batch_consistency.append(prediction_consistency(reference_logits, transformed_logits).item())
            batch_similarity.append(
                representation_similarity(reference_features, transformed_features).item()
            )

        results["strength"].append(strength)
        results["accuracy"].append(sum(batch_accuracy) / len(batch_accuracy))
        results["consistency"].append(sum(batch_consistency) / len(batch_consistency))
        results["similarity"].append(sum(batch_similarity) / len(batch_similarity))

    return results


def plot_results(all_results, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    keys = ["accuracy", "consistency", "similarity"]
    titles = ["Transformed accuracy", "Prediction consistency", "Latent similarity"]
    for name, results in all_results.items():
        for axis, key, title in zip(axes, keys, titles):
            axis.plot(results["strength"], results[key], marker="o", label=name)
            axis.set_title(title)
            axis.set_xlabel("Transformation strength")
            axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../conf", config_name="analyse")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = Path(to_absolute_path(cfg.run_dir))
    model, saved_cfg = load_model_from_run(str(run_dir), device)
    print(f"Loaded {saved_cfg.model.name} from {run_dir}")

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.2860], std=[0.3530]),
        ]
    )
    dataset = datasets.FashionMNIST(cfg.data_dir, train=False, download=True, transform=transform)
    dataset = Subset(dataset, range(min(cfg.analysis.max_examples, len(dataset))))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    output_dir = Path(to_absolute_path(cfg.output_dir)) if cfg.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        "translation": (translate, list(cfg.analysis.translation_pixels)),
        "rotation": (rotate, list(cfg.analysis.rotation_degrees)),
        "contrast": (scale_contrast, list(cfg.analysis.contrast_factors)),
    }
    for name, (function, strengths) in experiments.items():
        results = analyse_transform(model, loader, device, function, strengths)
        plot_results({name: results}, output_dir / f"{name}.png")
        print(f"Saved {output_dir / f'{name}.png'}")


if __name__ == "__main__":
    main()
