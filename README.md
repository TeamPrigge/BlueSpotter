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
| Experiment tracking (MLflow) | SQLite DB homed on Drive, artifacts in Drive `mlartifacts/` |
| Model versions | MLflow Model Registry (`models:/bluespotter-lc/<n>`) |

**Git holds code and data *descriptions* — never image data or model weights.**
Large binaries live in Drive. DVC versions the manifests and a content index of
the Drive files they point at, so every training run can name the exact dataset
it used. See [docs/SETUP.md](docs/SETUP.md) for the workflow and
[docs/DVC.md](docs/DVC.md) for data versioning.

## Quick start (Colab)

Everything runs in Colab. Nothing needs to run on a local machine.

1. **`notebooks/00_data_versioning.ipynb`** (CPU runtime). Snapshots the manifests
   from Drive, validates the splits, content-indexes the referenced images, pushes
   the blobs to Drive and commits `dvc.lock` + `reports/` to GitHub. Run this first,
   and again whenever the data changes.
2. **`notebooks/01_train_cellpose_sam.ipynb`** (GPU runtime). Confirms the data
   version is current, then fine-tunes Cellpose-SAM. MLflow logs the run with the
   dataset hash and git commit that produced it, registers the weights in the Model
   Registry, and writes the model back to Drive.

Both clone this repo using the `TOKEN_BlueSpotter` Colab Secret, so they can push
as well as pull. `params.yaml` already points at the Drive paths above.

## Repository layout

```
BlueSpotter/
├── notebooks/
│   ├── 00_data_versioning.ipynb      # DVC pipeline + publish to Drive/GitHub (CPU)
│   └── 01_train_cellpose_sam.ipynb   # Colab training entry point (GPU)
├── src/bluespotter/
│   ├── config.py                     # loads params.yaml, resolves Drive paths
│   ├── data.py                       # Drive→local caching, dataset loading
│   ├── naming.py                     # file name → mouse / channel / side / bregma AP
│   ├── discover.py                   # DVC stage: rebuild manifests by walking Drive
│   ├── drive_links.py                # optional: attach Drive file-IDs + shareable links
│   ├── manifest.py                   # manifest row → Drive path resolution
│   ├── manifest_sync.py              # DVC stage: snapshot manifests from Drive
│   ├── drive_index.py                # DVC stage: content-index the Drive files
│   ├── validate.py                   # DVC stage: schema + split-leakage checks
│   ├── mlflow_store.py               # SQLite backend store: Drive home, local working copy
│   ├── mlflow_utils.py               # MLflow tracking, provenance, model registration
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

The images stay in Drive; DVC versions what points at them. Run
`notebooks/00_data_versioning.ipynb` in Colab (CPU runtime is enough) — it runs the
stages, pushes the blobs to the Drive `dvcstore`, and commits `dvc.lock` plus the
reports to GitHub. Like everything else here, it needs no local terminal.

`validate` currently reports a **train/test group-leakage warning**: the same
animal (and one left/right hemisphere pair from a single section) appears in both
splits, which makes held-out scores optimistic. Details and the reasoning are in
[docs/DVC.md](docs/DVC.md).

## Experiment tracking

Run metadata lives in a SQLite database whose home is Drive but which SQLite only
ever opens on the Colab VM's local disk — the Drive FUSE mount does not provide the
file locking SQLite needs, and pointing it there corrupts the database.
`train.run()` restores it at the start, checkpoints it to Drive on a timer, and
publishes it at the end, so this is handled for you inside the training notebook.

The `bluespotter.mlflow_store` CLI (`status`, `restore`, `checkpoint`, `migrate`)
is wired into notebook cells; `00_data_versioning.ipynb` has the one-time
migration of the legacy `mlruns/` history.

A database backend is also what makes the **Model Registry** work at all; it is
unavailable on the legacy file store. Each run registers its weights as
`models:/bluespotter-lc/<n>`, tagged with the DVC dataset hash and git commit that
produced it. See [docs/MLFLOW.md](docs/MLFLOW.md) — including why migration must
happen before your first new run.

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
