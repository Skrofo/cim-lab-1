# Run management update

Replace the corresponding files in the CIM Lab 1 skeleton with the files in this package.

## Training

Automatic timestamped run:

```bash
uv run python -m cim_lab_1.train model=cnn
```

Named run:

```bash
uv run python -m cim_lab_1.train model=cnn run.name=cnn_baseline
```

Named augmented run:

```bash
uv run python -m cim_lab_1.train model=cnn augmentation=geometric run.name=cnn_augmented
```

Each execution receives a separate timestamped directory, even if the same name is reused:

```text
outputs/runs/cnn_baseline/2026-10-02_09-30-15/
├── .hydra/
├── checkpoint.pt
├── resolved_config.yaml
├── metrics.json
└── wandb_run.json       # only when W&B is enabled
```

## W&B

```bash
uv run python -m cim_lab_1.train \
  model=cnn \
  run.name=cnn_baseline \
  logging.use_wandb=true
```

W&B stores the metric history and run configuration. The local directory remains the canonical self-contained copy. If enabled, the checkpoint, resolved configuration and final metrics are also logged as a model artifact.

## Analysis

Pass the run directory rather than a checkpoint path:

```bash
uv run python -m cim_lab_1.analyse \
  run_dir=outputs/runs/cnn_baseline/2026-10-02_09-30-15
```

The analysis script reads `resolved_config.yaml`, rebuilds the correct architecture, and loads `checkpoint.pt` automatically. Analysis figures are stored under the selected run directory.
