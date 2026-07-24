# Deploy the BlueSpotter inference API to Google Cloud Run

A single container that serves LC segmentation. Runs on **CPU** by default;
flip to an **NVIDIA L4** by redeploying with GPU flags — no code change, no
rebuild. The landing page (priggelab.com) calls it over HTTPS.

## 0. One-time setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

Pick a region that offers Cloud Run **GPU** so you can flip to L4 later without
migrating: `europe-west1`, `us-central1`, or `asia-southeast1` are safe choices.

## 1. Deploy on CPU (the "free-ish" default)

Run from this directory (it contains the `Dockerfile`):

```bash
gcloud run deploy bluespotter \
  --source . \
  --region europe-west1 \
  --cpu 4 --memory 8Gi \
  --timeout 300 \
  --min-instances 0 --max-instances 2 \
  --allow-unauthenticated \
  --set-env-vars ALLOWED_ORIGINS=https://priggelab.com,https://www.priggelab.com
```

`gcloud` builds the image with Cloud Build, pushes it to Artifact Registry, and
prints a **Service URL** like `https://bluespotter-xxxxx.europe-west1.run.app`.
Your endpoint is that URL + `/segment`.

**Cost / "free" reality:**
- `--min-instances 0` scales to zero, so you pay only while serving — and light
  traffic can sit inside Cloud Run's monthly free tier.
- The trade-off is **cold starts**: the image is large (torch + cellpose) and the
  model loads on first hit, so the first request after idle takes ~30–60s. If that
  bothers demo visitors, set `--min-instances 1` to keep one instance warm (this
  leaves the free tier and costs a few $/month).

## 2. Flip to L4 GPU (when CPU is too slow)

Same command, GPU flags added. **No code or image change.**

```bash
gcloud run deploy bluespotter \
  --source . \
  --region europe-west1 \
  --gpu 1 --gpu-type nvidia-l4 \
  --cpu 4 --memory 16Gi \
  --no-cpu-throttling \
  --timeout 300 \
  --min-instances 0 --max-instances 1 \
  --allow-unauthenticated \
  --set-env-vars ALLOWED_ORIGINS=https://priggelab.com,https://www.priggelab.com
```

Notes:
- **Quota:** GPU on Cloud Run needs a quota grant. Request "Cloud Run Admin API –
  Total Nvidia L4 GPU allocation" in the console *before* deploying; approval is
  not instant.
- **Not free:** L4 bills per-second while a request is in flight (it can still
  scale to zero), but it is real money — expect roughly a low-single-digit $/hour
  equivalent only during active use.
- The container's `torch` sees the GPU automatically (`torch.cuda.is_available()`),
  so `/health` will report `gpu=True`.

## 3. Point the model at your weights

Once you've pushed weights to the Hub (`deploy/hub/upload_model.py`), add:

```
--set-env-vars BLUESPOTTER_MODEL_REPO=TeamPrigge/bluespotter-lc,BLUESPOTTER_WEIGHTS=bluespotter_lc,ALLOWED_ORIGINS=https://priggelab.com,https://www.priggelab.com
```

If the model repo is **private**, mint a read token and pass it as a secret so the
container can download the weights:

```bash
gcloud run services update bluespotter --region europe-west1 \
  --set-env-vars HF_TOKEN=hf_read_token_xxx
```

Until weights are configured, the API falls back to base Cellpose-SAM and says so
in `/health` and each response's `model` field.

## 4. Wire it into the landing page

Take the Service URL and set it as `API_BASE` at the top of
[`landing/index.html`](../../landing/index.html), then host and embed that page
(see [landing/DEPLOY_AND_EMBED.md](../../landing/DEPLOY_AND_EMBED.md)).

**CORS gotcha:** the browser's request origin is *wherever the landing page is
served from* — i.e. its GitHub-Pages/host origin, which is what shows inside the
WIX iframe, NOT `priggelab.com` itself. So `ALLOWED_ORIGINS` must include that
hosting origin. Set it at deploy time, e.g.:

```
--set-env-vars ALLOWED_ORIGINS=https://teamprigge.github.io,https://www.priggelab.com,https://priggelab.com
```

## Verify

```bash
URL=$(gcloud run services describe bluespotter --region europe-west1 --format='value(status.url)')
curl "$URL/health"
curl -F "file=@some_slice.tif" "$URL/segment" | python -c "import sys,json;d=json.load(sys.stdin);print('neurons:',d['n_neurons'],'| model:',d['model'])"
```

## CPU-only slim image (optional)

If you decide you don't need the easy GPU flip and want a smaller/cheaper/faster
CPU image, install the CPU-only torch build. Add this line to the `Dockerfile`
**before** `pip install -r requirements.txt`:

```dockerfile
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This pins a CPU wheel so cellpose won't pull the large CUDA build. Trade-off: to
later use an L4 you must remove that line and rebuild (no longer a pure flag flip).
