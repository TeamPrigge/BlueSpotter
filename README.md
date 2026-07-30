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

**Git holds code and data *descriptions* — never image data or model weights.**
Large binaries live in Drive. DVC versions the manifests and a content index of
the Drive files they point at, so every training run can name the exact dataset
it used. See [docs/SETUP.md](docs/SETUP.md) for the workflow and
[docs/DVC.md](docs/DVC.md) for data versioning.

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
│   ├── manifest.py                   # manifest row → Drive path resolution
│   ├── manifest_sync.py              # DVC stage: snapshot manifests from Drive
│   ├── drive_index.py                # DVC stage: content-index the Drive files
│   ├── validate.py                   # DVC stage: schema + split-leakage checks
│   ├── mlflow_utils.py               # MLflow tracking + data provenance
│   └── train.py                      # Cellpose-SAM transfer learning
├── deploy/                          # model hosting: HF Hub upload + Gradio Space demo
│   ├── hub/                          #   push weights + model card to the Hub
│   └── space/                        #   runnable Gradio demo Space
├── tests/                            # synthetic-Drive tests for the data layer
├── dvc.yaml                          # the pipeline (dvc dag / dvc repro)
├── dvc.lock                          # pins the exact dataset version per run
├── reports/                          # committed metrics: validation + index summaries
├── params.yaml                       # all paths + hyperparameters
├── pyproject.toml                    # ruff + pytest config
├── requirements.txt
├── .pre-commit-config.yaml           # nbstripout (keeps notebooks clean in git)
└── docs/{SETUP,DVC}.md
```

## Data versioning

The images stay in Drive; DVC versions what points at them.

```bash
dvc repro validate index-train index-test   # data checks, no GPU needed
dvc push                                    # blobs -> Drive dvcstore
dvc metrics diff main                        # what changed on this branch
```

`validate` currently reports a **train/test group-leakage warning**: the same
animal (and one left/right hemisphere pair from a single section) appears in both
splits, which makes held-out scores optimistic. Details and the reasoning are in
[docs/DVC.md](docs/DVC.md).

## Hosting the trained model

Once a model is trained, `deploy/` publishes it so it can be used interactively:
push the weights + a model card to the Hugging Face Hub, then run a Gradio Space
demo that segments an uploaded LC slice. See [deploy/README.md](deploy/README.md).

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
