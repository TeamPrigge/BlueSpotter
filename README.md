# BlueSpotter

Locus coeruleus–optimized neuronal segmentation platform.

BlueSpotter fine-tunes **Cellpose-SAM** for segmentation of locus coeruleus (LC)
neurons and provides a reproducible, cloud-only training workflow. It is the
implementation of the LC-Seg vision: a standardized, hardware-independent
pipeline for quantitative LC analysis.

## Where things live

| Asset | Location |
|-------|----------|
| Code (this repo, notebook, config) | GitHub `TeamPrigge/BlueSpotter` |
| Training images + masks | Google Drive `SoftwareTools/BlueSpotter/data` |
| Trained model weights | Google Drive `SoftwareTools/BlueSpotter/model` |
| Experiment tracking (MLflow) | Drive-synced `mlruns/` (overridable to a remote server) |

**Git holds code only — never image data or model weights.** Large binaries live
in Drive. See [docs/SETUP.md](docs/SETUP.md) for the full workflow.

## Quick start (Colab)

1. Open `notebooks/01_train_cellpose_sam.ipynb` in Google Colab (Pro, GPU runtime).
2. Run the setup cell — it mounts Drive, installs deps, and clones this repo.
3. Point `params.yaml` at your Drive folders (already defaulted to the paths above).
4. Run the training cell. Cellpose-SAM fine-tunes on your labeled data, MLflow
   logs the run, and the best model is written back to Drive.

## Repository layout

```
BlueSpotter/
├── notebooks/
│   └── 01_train_cellpose_sam.ipynb   # Colab training entry point
├── src/bluespotter/
│   ├── config.py                     # loads params.yaml, resolves Drive paths
│   ├── data.py                       # Drive→local caching, dataset loading
│   ├── mlflow_utils.py               # MLflow tracking setup
│   └── train.py                      # Cellpose-SAM transfer learning
├── params.yaml                       # all paths + hyperparameters
├── requirements.txt
├── .pre-commit-config.yaml           # nbstripout (keeps notebooks clean in git)
└── docs/SETUP.md
```

## Data convention

Cellpose expects paired image/mask files in one folder:

```
image001_img.tif      image001_masks.tif
image002_img.tif      image002_masks.tif
```

Filters (`_img`, `_masks`) are configurable in `params.yaml`.

## Status

Phase 0 — prototyping scaffold. Colab + Drive is the starting point; the MLflow
tracking URI is env-overridable so a hosted tracking server can be dropped in
later without code changes.
