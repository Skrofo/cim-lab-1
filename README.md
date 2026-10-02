# CIM Lab 1: Architecture, Augmentation and Invariance

In this lab you will compare a fully connected network and a convolutional neural network on FashionMNIST, then examine how their predictions and learned representations respond to image transformations.

## Learning objectives

By the end of the lab, you should be able to:

- configure and train MLP and CNN classifiers with Hydra;
- compare models while controlling experimental settings;
- measure prediction consistency and representation stability;
- distinguish invariance from equivariance;
- evaluate how data augmentation affects clean and transformed accuracy;
- record results with Weights & Biases.

## Setup
See the machine setup guide for WSL etc. from the previous lab [here](https://github.com/ivorsimpson/cim-lab-0/blob/main/README.md)

In terms of storing your code on GitHub, you have too choices, either forking this repository (and cloning that) or cloning this, and changing the "origin" see the [Github support guide from the last lab for details](https://github.com/ivorsimpson/cim-lab-0/blob/main/docs/github-support.md)

## Running the code

```bash
uv sync
```

All Python commands must run through the supplied project environment:

```bash
uv run python -m cim_lab_1.train
```

or activate it first:

```bash
source .venv/bin/activate
python -m cim_lab_1.train
```

Do not install packages globally.

## Quick start

1. Check the available configuration:

```bash
uv run python -m cim_lab_1.train --cfg job
```

2. Train the default MLP, name your run:

```bash
uv run python -m cim_lab_1.train model=mlp run.name=mlp_baseline
```

3. Train the CNN:

```bash
uv run python -m cim_lab_1.train model=cnn run.name=cnn_baseline
```

4. Analyse a saved checkpoint under controlled transformations:

```bash
uv run python -m cim_lab_1.analyse run_dir=outputs/runs/[run-name]/[timestamp]
```
HINT: Use Tab complete to find the names of the timestamp directories

Named run:

```bash
uv run python -m cim_lab_1.train model=cnn run.name=cnn_baseline
```

Named augmented run:

```bash
uv run python -m cim_lab_1.train model=cnn augmentation=geometric run.name=cnn_augmented
```

## Core tasks

### 1. Complete and compare the models

Open `src/cim_lab_1/models.py`.

- Complete the missing MLP and CNN components marked `TODO`.
- Determine the shape of the latent representation returned by each model.
- Record the number of trainable parameters.
- Keep the models reasonably comparable, but explain why exact parameter matching is not necessarily sufficient for a fair comparison.

### 2. Configure experiments

Use Hydra rather than editing constants in Python.

Try command-line overrides such as:

```bash
uv run python -m cim_lab_1.train model=cnn training.lr=0.01
uv run python -m cim_lab_1.train model=cnn training.optimizer=sgd
```

Run a small learning-rate comparison:

```bash
uv run python -m cim_lab_1.train --multirun model=cnn training.lr=0.0001,0.001,0.01
```

Record clean test accuracy and training loss.

### 3. Measure transformation robustness

Complete the missing metrics in `src/cim_lab_1/metrics.py` and the analysis loop in `src/cim_lab_1/analyse.py`.

To run the analysis pipeline call:


For translation, rotation and contrast changes, plot transformation strength against:

- transformed-image classification accuracy;
- prediction consistency with the untransformed image;
- cosine similarity between latent representations.

Consider:

- Is the transformation label-preserving?
- Does representation stability imply prediction stability?
- Are the latent vectors being tested for invariance or equivariance?

### 4. Test augmentation

Retrain at least one model with a suitable augmentation configuration.

Compare:

- clean accuracy;
- transformed accuracy;
- prediction consistency;
- latent similarity.

Discuss whether improved robustness creates a trade-off with clean accuracy or sensitivity to useful image information.

### 5. Record the experiment

Enable W&B using:

```bash
uv run python -m cim_lab_1.train run.name=[name] logging.use_wandb=true
```

Log the configuration, training loss, clean accuracy and checkpoint path. Never commit an API key.

## Suggested output

Produce one figure containing transformation strength on the horizontal axis and curves for:

- MLP;
- CNN;
- augmented CNN.

Your figure should report accuracy, prediction consistency and latent cosine similarity.

## Extensions

- Generate an animation showing an image being progressively transformed, with class probabilities plotted alongside it.
- Examine spatial CNN feature maps and test translation equivariance after aligning the transformed feature maps.
- Load a small pretrained vision model and compare its global embedding stability. Be explicit that embedding stability is not the same as rotation equivariance.
- Repeat selected experiments with multiple seeds and show the variability.

## Repository structure

```text
conf/                  Hydra configuration
src/cim_lab_1/         Training and analysis code
outputs/               Generated results, ignored by Git
tests/                 Small checks for metrics and model interfaces
```

## Minimum completion

- [ ] MLP and CNN both train successfully.
- [ ] At least two learning rates are compared.
- [ ] All three transformations are evaluated.
- [ ] Prediction consistency and latent similarity are plotted.
- [ ] One augmented model is compared with its unaugmented counterpart.
- [ ] One experiment is logged to W&B.
- [ ] Code and configuration are committed to Git.
