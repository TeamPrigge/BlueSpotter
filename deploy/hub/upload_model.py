"""Upload a fine-tuned BlueSpotter Cellpose-SAM model to the Hugging Face Hub.

Weights live in Google Drive (never in git). This script pushes a chosen weights
file plus the model card to a Hugging Face **model** repo so the Space demo and an
inference API can load it with `hf_hub_download`.

Auth: provide a WRITE token via the HF_TOKEN env var (or run
`huggingface-cli login` first). This script never hard-codes a secret.

Examples
--------
    export HF_TOKEN=hf_xxx          # a write token from hf.co/settings/tokens
    python deploy/hub/upload_model.py \
        --weights "/content/drive/MyDrive/TeamPrigge/SoftwareTools/BlueSpotter/model/bluespotter_lc_20260724_120000" \
        --repo-id TeamPrigge/bluespotter-lc \
        --weights-name bluespotter_lc

Run with --private to keep the model repo private while iterating.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CARD = Path(__file__).with_name("MODEL_CARD.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="Path to the trained Cellpose model file (from the Drive model/ dir).")
    ap.add_argument("--repo-id", required=True,
                    help="Target HF model repo, e.g. TeamPrigge/bluespotter-lc")
    ap.add_argument("--weights-name", default=None,
                    help="Filename to store the weights as in the repo (default: keep the original name).")
    ap.add_argument("--private", action="store_true", help="Create the repo as private.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Create a WRITE token at "
                 "https://huggingface.co/settings/tokens and `export HF_TOKEN=...` first.")

    weights = Path(args.weights).expanduser()
    if not weights.is_file():
        sys.exit(f"Weights file not found: {weights}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")

    api = HfApi(token=token)
    print(f"Creating/ensuring repo: {args.repo_id} (private={args.private})")
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    dest = args.weights_name or weights.name
    print(f"Uploading weights: {weights.name}  ->  {args.repo_id}/{dest}")
    api.upload_file(path_or_fileobj=str(weights), path_in_repo=dest,
                    repo_id=args.repo_id, repo_type="model")

    if CARD.is_file():
        print("Uploading model card -> README.md")
        api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="model")
    else:
        print(f"WARNING: model card not found at {CARD}; skipping.")

    print(f"\nDone. https://huggingface.co/{args.repo_id}")
    print(f"The Space demo loads this via BLUESPOTTER_MODEL_REPO={args.repo_id} "
          f"and BLUESPOTTER_WEIGHTS={dest}")


if __name__ == "__main__":
    main()
