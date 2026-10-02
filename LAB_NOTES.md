# Instructor notes

## Intended balance

The repository provides:

- working Hydra configuration and training/evaluation structure;
- data loading and checkpointing scaffolding;
- W&B integration;
- transformation functions and plotting boilerplate.

Students must complete:

- missing MLP and CNN layers;
- one deliberate data-pipeline repair;
- prediction-consistency and representation-similarity metrics;
- the transformation-analysis loop;
- experiment selection, execution and interpretation.

This should allow an early successful run without reducing the exercise to changing configuration values.

## Suggested core order

1. Repair `data.py` and complete `models.py`.
2. Train MLP and CNN with default settings.
3. Implement metrics and run tests.
4. Complete `analyse_transform`.
5. Produce transformation curves.
6. Train an augmented CNN and compare exported plots.

## Optional instructor solution hints

- In `data.py`, import `torch` and use `v2.ToDtype(torch.float32, scale=True)`.
- MLP encoder: `Linear(hidden_dim, latent_dim)` followed by `ReLU`.
- CNN second block: `Conv2d(c1, c2, 3, padding=1)`, `ReLU`, `MaxPool2d(2)`.
- Prediction consistency compares the two `argmax` outputs.
- Representation similarity uses `F.cosine_similarity(..., dim=1).mean()`.
- Analysis should calculate reference outputs once per batch and compare each transformed version to them.

## Important conceptual distinction

For pooled latent vectors and class predictions, the analysis measures invariance or stability. To assess equivariance of spatial feature maps, transform and align the feature maps before comparing them.
