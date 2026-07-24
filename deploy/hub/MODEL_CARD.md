---
license: cc-by-nc-4.0
tags:
  - cellpose
  - cellpose-sam
  - image-segmentation
  - microscopy
  - biology
  - neuroscience
  - locus-coeruleus
pipeline_tag: image-segmentation
---

# BlueSpotter — LC-optimized Cellpose-SAM

Fine-tuned **Cellpose-SAM** weights for instance segmentation of **locus coeruleus
(LC) neurons** in neuromelanin / histology microscopy slices. Part of the
[BlueSpotter](https://github.com/TeamPrigge/BlueSpotter) project (priggelab.com):
a standardized, hardware-independent pipeline for quantitative LC analysis.

## What this is

- **Base model:** Cellpose-SAM (`cpsam`).
- **Task:** instance segmentation — each detected neuron gets a unique integer label.
- **Trained on:** paired image/mask crops from neuromelanin slice cohorts
  (see the BlueSpotter repo's data convention and manifest).

## Usage

```python
from huggingface_hub import hf_hub_download
from cellpose import models

# Download the fine-tuned weights from this repo
weights = hf_hub_download(repo_id="TeamPrigge/bluespotter-lc", filename="bluespotter_lc")

# Load exactly like the base Cellpose-SAM model, but with these weights
model = models.CellposeModel(gpu=True, pretrained_model=weights)

# Segment — masks is a label image; masks.max() == number of neurons found
masks, flows, styles = model.eval(image)
n_neurons = int(masks.max())
```

The inference call matches BlueSpotter's training-time QC (`src/bluespotter/viz.py`),
so masks here are directly comparable to the validation overlays produced during training.

## Intended use & limitations

- **Intended:** research-grade segmentation of LC neurons in similar imaging
  modalities to the training data.
- **Out of scope:** clinical/diagnostic use; cell types or modalities far from the
  training distribution. Always visually QC predictions before quantifying.

## Training

Fine-tuned via the BlueSpotter Colab workflow. Hyperparameters and per-epoch losses
for each run are tracked in MLflow; see `params.yaml` and `docs/SETUP.md` in the
[code repository](https://github.com/TeamPrigge/BlueSpotter).

## Citation

If you use these weights, please cite Cellpose-SAM and credit the BlueSpotter
project (priggelab.com).

> **License note:** `cc-by-nc-4.0` is a placeholder default for non-commercial
> research sharing — confirm or change it before publishing.
