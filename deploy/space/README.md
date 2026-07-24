---
title: BlueSpotter LC Demo
emoji: 🔵
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# BlueSpotter — LC segmentation demo

Interactive demo of BlueSpotter's fine-tuned **Cellpose-SAM** model for locus
coeruleus (LC) neuron segmentation. Upload a microscopy slice → get an outline
overlay and a neuron count.

- Code: https://github.com/TeamPrigge/BlueSpotter
- Model weights: https://huggingface.co/TeamPrigge/bluespotter-lc

## Configuration

Set these under **Settings → Variables and secrets** if your repo names differ
from the defaults:

| Variable | Default | Meaning |
|----------|---------|---------|
| `BLUESPOTTER_MODEL_REPO` | `TeamPrigge/bluespotter-lc` | HF model repo holding the weights |
| `BLUESPOTTER_WEIGHTS` | `bluespotter_lc` | weights filename inside that repo |

If the weights can't be loaded, the demo falls back to the base Cellpose-SAM model
and says so, so the Space still runs before weights are published.

> This README's YAML front matter is the Space configuration Hugging Face reads.
> `sdk_version` is pinned for reproducibility — bump it when you upgrade Gradio.
