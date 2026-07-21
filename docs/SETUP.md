# BlueSpotter setup & workflow

Cloud-only workflow: **code in GitHub, data/models in Google Drive, training on
Colab, experiment tracking in MLflow.** No local machine required.

## 1. One-time prep

### Google Drive
Your folders already exist:

```
SoftwareTools/BlueSpotter/
├── data/      <- upload paired training images + masks here
└── model/     <- trained weights land here automatically
```

**Data naming convention.** Cellpose pairs files by filename suffix:

```
sampleA_img.tif      sampleA_masks.tif
sampleB_img.tif      sampleB_masks.tif
```

- `*_img` = the raw image (single- or multi-channel).
- `*_masks` = the instance label image (each neuron a unique integer, background 0).

Masks are what you need most and are usually the bottleneck — label them in the
Cellpose GUI, QuPath, or napari. Start with as few as ~10–20 well-labeled images;
Cellpose-SAM fine-tunes from a strong prior.

### GitHub
This repo holds code only. If you want notebook diffs to stay clean:

```bash
pip install pre-commit
pre-commit install    # runs nbstripout on every commit
```

## 2. Train (Colab)

1. Open `notebooks/01_train_cellpose_sam.ipynb` in Colab.
2. Runtime → Change runtime type → **GPU** (A100/L4 on Pro is ideal).
3. Run the **Setup** cell: mounts Drive, installs deps, clones this repo.
4. Check `params.yaml` — especially `drive.root`. Default assumes My Drive:
   `/content/drive/MyDrive/SoftwareTools/BlueSpotter`. If BlueSpotter is in a
   **Shared drive** called TeamPrigge, change it to
   `/content/drive/Shareddrives/TeamPrigge/SoftwareTools/BlueSpotter`.
5. Run the **Train** cell.

What happens: data is copied Drive → local SSD, Cellpose-SAM fine-tunes, MLflow
logs params/metrics, and the best model is written to `BlueSpotter/model/`.

## 3. Inspect runs (MLflow)

Runs log to `BlueSpotter/mlruns/` on Drive by default. To browse them, in a Colab
cell:

```python
!mlflow ui --backend-store-uri "file:///content/drive/MyDrive/SoftwareTools/BlueSpotter/mlruns" --port 5000 &
```

then open the port via Colab's proxy. (A local `mlruns` in a dead Colab session
would vanish — that's why it lives on Drive.)

## 4. Migration path (later phases)

- **Persistent GPU**: move off Colab to a rented/cluster GPU when runs get long.
- **Hosted MLflow**: set `MLFLOW_TRACKING_URI` to a remote server (self-hosted VM
  or a service like DagsHub) — the code already respects that env var.
- **DVC**: once the curated dataset stabilizes, put it under DVC so exact dataset
  versions are pinned to each model. Drive can serve as the DVC remote (mind the
  API quota) or swap in object storage.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Drive data folder not found` | `drive.root` in params.yaml is wrong, or Drive isn't mounted. |
| `No files found in .../data` | Upload paired `*_img`/`*_masks` files to Drive `data/`. |
| Training very slow | Confirm GPU runtime; confirm data was cached locally, not read off Drive. |
| Colab disconnects mid-run | Expected on long runs — reduce epochs, or move to a persistent GPU. |
