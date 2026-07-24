"""BlueSpotter inference API (FastAPI, deployable to Google Cloud Run).

A single container that serves LC-neuron segmentation. Runs on CPU by default;
switch to an NVIDIA L4 by redeploying with GPU flags (see deploy.md) — no code
change. Same inference path as the training-time QC (src/bluespotter/viz.py) and
the Gradio Space, so results are consistent across all three.

Endpoints
---------
  GET  /health          liveness + which model is loaded
  POST /segment         multipart file upload -> JSON {n_neurons, overlay_png_b64, model}

Configuration (env vars, set in Cloud Run)
  ALLOWED_ORIGINS         comma-separated CORS origins
                          (default: https://priggelab.com,https://www.priggelab.com)
  BLUESPOTTER_MODEL_REPO  HF model repo with the fine-tuned weights
                          (default: TeamPrigge/bluespotter-lc)
  BLUESPOTTER_WEIGHTS     weights filename within that repo (default: bluespotter_lc)
"""
from __future__ import annotations

import base64
import io
import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from skimage.io import imread

from cellpose import models
from cellpose.utils import outlines_list

ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS", "https://priggelab.com,https://www.priggelab.com"
    ).split(",") if o.strip()
]
MODEL_REPO = os.environ.get("BLUESPOTTER_MODEL_REPO", "TeamPrigge/bluespotter-lc")
WEIGHTS_FILE = os.environ.get("BLUESPOTTER_WEIGHTS", "bluespotter_lc")

app = FastAPI(title="BlueSpotter inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

MODEL = None
MODEL_STATUS = "not loaded"


def _load_model():
    """Load fine-tuned weights from the Hub; fall back to base Cellpose-SAM."""
    global MODEL, MODEL_STATUS
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        gpu = False
    try:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(repo_id=MODEL_REPO, filename=WEIGHTS_FILE)
        MODEL = models.CellposeModel(gpu=gpu, pretrained_model=weights)
        MODEL_STATUS = f"fine-tuned:{MODEL_REPO}/{WEIGHTS_FILE} (gpu={gpu})"
    except Exception as e:  # noqa: BLE001
        MODEL = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")
        MODEL_STATUS = f"base cpsam fallback ({type(e).__name__}) (gpu={gpu})"


@app.on_event("startup")
def _startup():
    _load_model()


def _disp(img) -> np.ndarray:
    """1%-99% contrast-stretched 2D grayscale for display (matches viz.py)."""
    a = np.asarray(img).astype(float)
    if a.ndim == 3:
        a = np.moveaxis(a, int(np.argmin(a.shape)), 0)[0]
    mn, mx = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - mn) / (mx - mn), 0, 1) if mx > mn else a


def _overlay_png_b64(img, masks) -> str:
    """Red neuron outlines over the grayscale image, returned as base64 PNG."""
    disp = _disp(img)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(disp, cmap="gray")
    for o in outlines_list(masks):
        ax.plot(o[:, 0], o[:, 1], color="red", lw=0.8)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_STATUS}


def _cell_measurements(masks) -> list[dict]:
    """Per-neuron rows for the CSV: id, area (px), centroid (x, y)."""
    from skimage.measure import regionprops
    rows = []
    for p in regionprops(masks):
        cy, cx = p.centroid  # regionprops is (row, col) = (y, x)
        rows.append({
            "cell_id": int(p.label),
            "area_px": int(p.area),
            "centroid_x": round(float(cx), 1),
            "centroid_y": round(float(cy), 1),
        })
    return rows


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    data = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        img = imread(tmp.name)
    masks = np.asarray(MODEL.eval(img)[0]).astype(np.int32)
    cells = _cell_measurements(masks)
    return {
        "n_neurons": len(cells),
        "cells": cells,                      # -> results.csv on the client
        "overlay_png_b64": _overlay_png_b64(img, masks),
        "filename": file.filename,
        "model": MODEL_STATUS,
    }
