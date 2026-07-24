"""BlueSpotter — LC segmentation demo (Hugging Face Space).

Upload a microscopy slice; the fine-tuned Cellpose-SAM model segments locus
coeruleus neurons and returns an outline overlay plus a neuron count. The
inference + display logic mirrors BlueSpotter's training-time QC
(src/bluespotter/viz.py) so results are comparable.

Configuration (Space > Settings > Variables and secrets):
  BLUESPOTTER_MODEL_REPO  HF model repo with the fine-tuned weights
                          (default: TeamPrigge/bluespotter-lc)
  BLUESPOTTER_WEIGHTS     weights filename within that repo (default: bluespotter_lc)

If the fine-tuned weights can't be downloaded, the app falls back to the base
Cellpose-SAM model so the Space still runs, and says so in the UI.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless rendering on the Space
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr
from skimage.io import imread

from cellpose import models
from cellpose.utils import outlines_list

MODEL_REPO = os.environ.get("BLUESPOTTER_MODEL_REPO", "TeamPrigge/bluespotter-lc")
WEIGHTS_FILE = os.environ.get("BLUESPOTTER_WEIGHTS", "bluespotter_lc")


def _load_model():
    """Load fine-tuned weights from the Hub; fall back to base Cellpose-SAM."""
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        gpu = False
    try:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(repo_id=MODEL_REPO, filename=WEIGHTS_FILE)
        model = models.CellposeModel(gpu=gpu, pretrained_model=weights)
        return model, f"Fine-tuned BlueSpotter weights ({MODEL_REPO}/{WEIGHTS_FILE})"
    except Exception as e:  # noqa: BLE001
        model = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")
        return model, (f"⚠️ Could not load fine-tuned weights ({type(e).__name__}); "
                       f"using base Cellpose-SAM. Upload weights to {MODEL_REPO} "
                       f"and set BLUESPOTTER_WEIGHTS to enable the tuned model.")


MODEL, MODEL_STATUS = _load_model()


def _disp(img) -> np.ndarray:
    """1%-99% contrast-stretched 2D grayscale for display (matches viz.py)."""
    a = np.asarray(img).astype(float)
    if a.ndim == 3:
        a = np.moveaxis(a, int(np.argmin(a.shape)), 0)[0]  # channel-first, take ch0
    mn, mx = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - mn) / (mx - mn), 0, 1) if mx > mn else a


def _overlay(img, masks) -> np.ndarray:
    """Render red neuron outlines over the grayscale image as an RGB array."""
    disp = _disp(img)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(disp, cmap="gray")
    for o in outlines_list(masks):
        ax.plot(o[:, 0], o[:, 1], color="red", lw=0.8)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def segment(image_path):
    if not image_path:
        return None, "Upload an image to segment."
    img = imread(image_path)
    masks = np.asarray(MODEL.eval(img)[0])
    n = int(masks.max())
    overlay = _overlay(img, masks)
    return overlay, f"**{n}** neurons detected.\n\n_{MODEL_STATUS}_"


with gr.Blocks(title="BlueSpotter — LC segmentation") as demo:
    gr.Markdown(
        "# 🔵 BlueSpotter — LC neuron segmentation\n"
        "Fine-tuned **Cellpose-SAM** for locus coeruleus neurons. "
        "Upload a microscopy slice (TIFF/PNG/JPG) to get an outline overlay and a count.\n\n"
        "Part of the [BlueSpotter](https://github.com/TeamPrigge/BlueSpotter) project · priggelab.com"
    )
    with gr.Row():
        inp = gr.Image(type="filepath", label="Input slice", sources=["upload"])
        out = gr.Image(type="numpy", label="Segmentation (red outlines)")
    info = gr.Markdown()
    gr.Button("Segment", variant="primary").click(segment, inputs=inp, outputs=[out, info])

if __name__ == "__main__":
    demo.launch()
