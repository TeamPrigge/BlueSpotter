# BlueSpotter — model hosting & serving (roadmap stages 3–5)

Publish the fine-tuned Cellpose-SAM model so it can be *used*, not just trained.
Two serving paths, sharing the same inference logic:

1. **Hugging Face Hub + Space** (`hub/`, `space/`) — the trained weights + a model
   card in a model repo, and a Gradio demo Space. Quickest public demo.
2. **Google Cloud Run API** (`cloudrun/`) — a containerized FastAPI service the
   landing page (priggelab.com) calls directly. Runs on CPU by default; flip to an
   NVIDIA L4 with a deploy flag (no rebuild). This is the path wired into the site.

Weights live in Google Drive and are **never** committed to git. This directory
holds only the code + docs to push them to Hugging Face / Cloud Run.

```
deploy/
├── hub/
│   ├── MODEL_CARD.md      # HF model card (becomes the model repo's README)
│   └── upload_model.py    # push weights + card to the Hub (you run this, with a token)
├── space/
│   ├── app.py             # Gradio demo (loads weights from the Hub)
│   ├── requirements.txt
│   └── README.md          # Space config (YAML front matter) + docs
└── cloudrun/
    ├── main.py            # FastAPI: POST /segment, CORS locked to priggelab.com
    ├── Dockerfile         # one image for both CPU and L4 GPU
    ├── requirements.txt
    ├── wix_snippet.html   # drop-in widget for the WIX / landing page
    └── deploy.md          # gcloud commands: CPU deploy + one-flag L4 flip
```

The Hub upload (`hub/upload_model.py`) feeds **both** serving paths — the Space and
the Cloud Run API each download the weights from the same model repo. Do that first.

## Prerequisites (one-time, done by you)

Publishing to Hugging Face is an outbound, public-content action, so the token
steps are yours to run — the code here never contains a secret.

1. Create a Hugging Face account (or use the org, e.g. `TeamPrigge`).
2. Create a **write** token: https://huggingface.co/settings/tokens
3. Decide the two repo ids (defaults assume a `TeamPrigge` HF org):
   - model repo: `TeamPrigge/bluespotter-lc`
   - Space:      `TeamPrigge/bluespotter-demo`

> If your Hugging Face username/org differs from GitHub's `TeamPrigge`, substitute
> it everywhere below and update the defaults in `space/README.md` +
> `hub/MODEL_CARD.md`.

## Step 1 — push the model to the Hub

Run where the weights are reachable (e.g. Colab after training, or locally if you
downloaded them from Drive):

```bash
pip install huggingface_hub
export HF_TOKEN=hf_xxx   # your write token

python deploy/hub/upload_model.py \
  --weights "/content/drive/MyDrive/TeamPrigge/SoftwareTools/BlueSpotter/model/<your_model_file>" \
  --repo-id TeamPrigge/bluespotter-lc \
  --weights-name bluespotter_lc \
  --private          # drop --private when you're ready to make it public
```

This creates the repo, uploads the weights as `bluespotter_lc`, and uploads
`MODEL_CARD.md` as the repo README.

## Step 2 — publish the Space demo

The `space/` folder is a complete Gradio Space. Create the Space, then push these
three files to its git repo:

```bash
# One-time: create an empty Gradio Space named bluespotter-demo in the HF UI,
# then push the demo files into it.
git clone https://huggingface.co/spaces/TeamPrigge/bluespotter-demo
cp deploy/space/app.py deploy/space/requirements.txt deploy/space/README.md bluespotter-demo/
cd bluespotter-demo && git add . && git commit -m "BlueSpotter LC demo" && git push
```

If the model repo is **private**, add your `HF_TOKEN` as a Space secret so the app
can download the weights; a public model repo needs no secret.

## Verify

- Model page shows the weights file + rendered card: `https://huggingface.co/TeamPrigge/bluespotter-lc`
- Space builds and, on upload, returns a red-outline overlay + neuron count:
  `https://huggingface.co/spaces/TeamPrigge/bluespotter-demo`

## Cloud Run API (the path wired into the landing page)

For the site itself, deploy the FastAPI service in `cloudrun/` — CPU by default,
one-flag switch to an L4 GPU. Full instructions: [cloudrun/deploy.md](cloudrun/deploy.md).
It reuses the exact same `segment()` inference path as the Space, and downloads
weights from the same Hub model repo you created above.
