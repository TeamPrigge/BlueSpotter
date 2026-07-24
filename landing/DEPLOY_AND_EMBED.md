# Go live: deploy the API + embed the landing page in WIX

The landing page and its demo are **built and verified working** (tested locally
against a mock backend — upload → segment → overlay + neuron count + `results.csv`).
Three steps remain, and they need YOUR Google Cloud + WIX accounts, so they're
listed here for you to run.

> **Why I couldn't finish these for you:** `gcloud` isn't installed on this Mac and
> I can't complete a Google sign-in on your behalf, so I cannot create the Cloud
> Run service (it also bills your account and creates public infrastructure). No
> live URL was fabricated. Everything below is ready to run.

## Step 1 — deploy the CPU API to Google Cloud Run (~10 min)

```bash
# Install the Google Cloud CLI (once)
brew install --cask google-cloud-sdk        # macOS

# Sign in + pick your project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# Deploy (CPU). Run from the repo root.
cd deploy/cloudrun
gcloud run deploy bluespotter \
  --source . --region europe-west1 \
  --cpu 4 --memory 8Gi --timeout 300 \
  --min-instances 0 --max-instances 2 --allow-unauthenticated \
  --set-env-vars ALLOWED_ORIGINS=https://teamprigge.github.io,https://www.priggelab.com,https://priggelab.com
```

It prints a **Service URL** like `https://bluespotter-xxxxx.europe-west1.run.app`.
Full details, the model-repo env vars, and the one-flag **L4 GPU** switch are in
[../deploy/cloudrun/deploy.md](../deploy/cloudrun/deploy.md).

## Step 2 — point the landing page at that URL

Edit one line near the bottom of [`index.html`](index.html):

```js
const API_BASE = "https://bluespotter-xxxxx.europe-west1.run.app";
```

Then host the page. Easiest is **GitHub Pages** from this repo:
copy `landing/index.html` to `docs/index.html` (this replaces the old compiled
bundle) and enable Pages → it serves at `https://teamprigge.github.io/BlueSpotter/`.
That origin is already in the `ALLOWED_ORIGINS` above.

## Step 3 — embed in WIX (priggelab.com)

In the WIX Editor:

1. **Add → Embed Code → Embed a Site** (this is an iframe element).
2. Paste your hosted landing-page URL (e.g. `https://teamprigge.github.io/BlueSpotter/`).
3. Size the element to fill the section; publish.

The demo runs *inside* that iframe, calling your Cloud Run URL directly. Because
the request comes from the iframe's origin (the GitHub-Pages URL), that origin —
not `priggelab.com` — is what must be in the API's `ALLOWED_ORIGINS` (it already
is, from Step 1).

> Prefer the demo widget only, not the whole page? Use **Embed a Widget → Custom
> code (HTML)** and paste just the `.demo` card markup + the `<script>` block from
> `index.html`. The iframe-the-whole-page route above is simpler and recommended.

## Local testing (no cloud needed)

```bash
python landing/mock_server.py     # serves the page + a fake /segment
open http://localhost:8765
```

The mock returns the exact JSON shape the real API returns, so the front-end behaves
identically — only the segmentation numbers are fake.

## What's verified vs. pending

| | Status |
|---|---|
| Landing page renders (design, fonts, all sections) | ✅ verified in-browser |
| Demo: upload → fetch → overlay + count + CSV download | ✅ verified against mock |
| Cloud Run API code + Dockerfile (returns count, overlay, per-cell CSV rows) | ✅ built, compiles |
| API deployed to your GCP project | ⏳ Step 1 (needs your account) |
| `API_BASE` set + page hosted | ⏳ Step 2 |
| Embedded in WIX | ⏳ Step 3 |
