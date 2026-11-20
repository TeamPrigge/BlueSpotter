# Data versioning in BlueSpotter

## The problem this solves

The images live in Google Drive and always will — that is where they get uploaded,
browsed, and annotated. What Drive cannot tell you is *which* images a given
result came from. The manifests (`train.csv` / `test.csv`) are lists of links, so
if someone re-exports a `.czi`, re-runs Cellpose to regenerate a
`_cp_masks.png`, or replaces a file while keeping its name, nothing anywhere
records that the dataset changed. Six months later you cannot reconstruct what
"the model that scored 0.82" was trained on.

DVC closes that gap without moving a single image out of Drive. It versions the
*description* of the dataset — which files are in which split, and what those
files contained at the moment of the run. Descriptions are small, so git holds
them comfortably, and they are sufficient to make a run reproducible.

## How it fits together

```
   Google Drive                    this repo (git + DVC)              GitHub
   ────────────                    ─────────────────────              ──────
   data/training_data/train.csv ──▶ data/manifests/train.csv ─┐
   data/test_data/test.csv      ──▶ data/manifests/test.csv   │  hashes
                                                             ├─▶ dvc.lock ──▶ PR diff
   Data/NM_Slices/**            ──▶ data/index/*_index.csv  ──┘
     (images + masks, never copied)   size / mtime / md5 per file

   dvcstore/  ◀── dvc push ── DVC cache (manifest + index blobs)
```

Drive stays the authoring surface. The repo becomes the versioning surface. The
DVC remote is a plain directory inside your existing Drive folder, so the blobs
live alongside everything else.

## Why a directory remote and not `gdrive://`

DVC does have a native Google Drive backend, but it now requires each user to
create their own Google Cloud project and OAuth client, because DVC retired the
shared client ID it used to ship. That is real setup effort, real credentials to
manage, and a real thing to leak — in exchange for nothing we need.

Colab already mounts Drive as a filesystem, and Drive for Desktop does the same
on a laptop. So DVC can simply write files. The remote is configured as:

```
['remote "drivestore"']
    url = /content/drive/MyDrive/TeamPrigge/SoftwareTools/BlueSpotter/dvcstore
```

No tokens, no service accounts, nothing to rotate. The trade-off is that the path
differs per machine, which is what `.dvc/config.local` is for (see below).

## The pipeline

Five stages, defined in `dvc.yaml`:

| Stage | What it does |
|---|---|
| `discover` | Walks the Drive mount and regenerates `train.csv`, `test.csv` and `ap_position.csv` from what is actually there. Gated on `dvc.discover`. |
| `sync-manifests` | Copies those three CSVs from Drive into `data/manifests/`, where DVC hashes them. |
| `validate` | Schema, duplicate rows, exact split overlap, **group leakage**, and AP-coordinate sanity checks. |
| `index-train` / `index-test` / `index-ap` | Resolves every manifest row on Drive and records size, mtime and a content hash per file. |
| `train` | Fine-tunes Cellpose-SAM. Depends on the *indexes*, so a changed image invalidates the model even if no manifest row moved. |

Inspect it with `dvc dag`.

## Why the manifests are generated, not authored

`train.csv` started life as a hand-typed list. That works until it doesn't: the
labelled cohorts on Drive grew to several hundred image/mask pairs across a dozen
differently-shaped folder layouts, while the manifest still named 68 of them.
Every slice a colleague segments is another row nobody types.

So `discover` derives the manifests instead. It walks `data.nmslices_root`, finds
every `*_seg.npy` and every `*_cp_masks.png` with its matching image, and reads
mouse, channel, hemisphere and — where the name records it — the bregma
coordinate out of the file name (`src/bluespotter/naming.py`). Rows carry a
`rel_path`, so `manifest.py` no longer has to know each cohort's folder shape;
adding a cohort is now a matter of putting files on Drive, not editing code.

Two things it deliberately does *not* do:

- **Guess.** A file name it cannot parse is reported in `reports/discovery.json`
  and left out of the manifests. A missing row is visible; a row with invented
  metadata is not.
- **Split by image.** Splits are grouped by animal, via a deterministic hash of
  the mouse ID. Two hemispheres of one section share biology, staining batch and
  imaging session; scoring on one after training on the other measures
  memorisation of that animal. The hash means the split is identical on every
  machine and every rerun without a seed file to lose.

`discover` is off by default (`dvc.discover: false`) so a run on a machine
without the Drive mount cannot overwrite the manifests. Turn it on in Colab.

## `ap_position.csv` — the anterior-posterior subset

Roughly 500 labelled slices are named like `TH_DHC-2076-5.35_R_seg.npy`, where
`5.35` is the distance behind bregma in millimetres. That is a free, per-file
anatomical label, and it is the seed for a second model: predicting rostrocaudal
position from LC shape.

Those rows get their own manifest rather than an extra column on `train.csv`,
because the two datasets need different treatment:

- the AP model needs a split that is grouped by animal **and** stratified across
  the AP range, which is not the segmentation split;
- an AP row is only useful if the coordinate is trustworthy, so `validate`
  rejects any value outside the LC's −6.5…−4.5 mm window as a misparsed name;
- the AP dataset will version, train and publish on its own schedule.

The same slices still appear in `train.csv`, where the AP label is simply unused
— the AP file is a *view*, not a fork, and `validate` checks that every AP row
is still present in one of the segmentation splits.

Section ordinals like `ap2` in `DHC_0464_02.vsi - ap2_TH_left_seg.npy` are kept
in a separate `ap_index` column and never treated as millimetres. They order
sections within one animal but are not comparable across animals; mixing the two
scales would quietly corrupt any AP regression.

Run `python -m bluespotter.drive_links data/manifests/ap_position.csv` in Colab
to fill in Drive file-IDs and clickable `drive_url` links, so the coordinates can
be spot-checked against the actual images. It is optional and never fails the
pipeline — a manifest without links still trains.

The Drive-facing stages are marked `always_changed: true`. Drive can change at
any moment without anything in git changing, so those stages must re-check every
run. Downstream stages still only re-run when the resulting index actually
differs — so a no-op re-check is cheap.

## How you actually run this

**In Colab, via `notebooks/00_data_versioning.ipynb`.** BlueSpotter is a cloud-only
workflow — code in GitHub, data in Drive, execution in Colab — and data versioning
is no exception. That notebook needs only a CPU runtime, and it does the whole
loop: run the stages, show the reports, `dvc push` the blobs to Drive, then commit
`dvc.lock` and `reports/` and push them to GitHub using the `TOKEN_BlueSpotter`
Colab Secret. No local clone, no local terminal.

`notebooks/01_train_cellpose_sam.ipynb` then only *checks* that the index is
current before training, so a stale dataset version cannot quietly end up baked
into a model.

The underlying commands, for reference — the notebook runs these for you:

```bash
dvc repro validate index-train index-test   # data checks only, no GPU needed
dvc repro                                   # everything, including training
dvc push                                    # send blobs to the Drive dvcstore
dvc pull                                    # restore manifests/indexes on a fresh clone

dvc status                                  # what is stale?
dvc metrics show                            # current numbers
dvc metrics diff main                       # what changed vs main
dvc plots show                              # loss curves
```

The metafiles that get committed are `dvc.lock`, `dvc.yaml` and `reports/*.json`.
`core.autostage = true` means DVC stages them for you; the notebook does the
commit and push.

## What lands in git, and what does not

Committed: `dvc.yaml`, `dvc.lock`, `.dvc/config`, `.dvcignore`, `pyproject.toml`,
and everything under `reports/`.

Not committed: the images (they are in Drive), and the DVC-cached artifacts
`data/manifests/*.csv` and `data/index/*.csv` — those are gitignored, hashed into
`dvc.lock`, and restored with `dvc pull`.

The practical consequence: a pull request that changes the dataset shows up as a
`dvc.lock` hash change plus a readable diff in `reports/validation.json`. Nobody
has to trust a verbal "I updated the training set".

## hash_mode: what you are trading

Set in `params.yaml` under `dvc.hash_mode`:

- **`full`** — md5 over every byte. Exact. Slowest over a Drive mount.
- **`partial`** (current default) — md5 over the first and last 4 MiB plus the file
  length. Catches re-exports and regenerated masks, because image formats put
  headers at the front and TIFF keeps IFD offsets near the end. Blind to an edit
  confined to the middle of a file larger than 8 MiB.
- **`size_mtime`** — no content read at all. Fast, and catches almost nothing.

Hashes are memoised in a sidecar cache keyed by `(path, size, mtime)` in the local
scratch dir, so re-runs only read files that actually changed. The cache applies
git's "racily clean" rule: an entry whose file mtime is not comfortably older than
the moment the hash was recorded is treated as untrusted and re-read, because
Drive-backed mounts often round mtime to whole seconds and a fast same-size edit
would otherwise slip through. There is one residual hole, worth stating plainly:
nothing keyed on size and mtime can catch a writer that deliberately restores the
original mtime after editing. If you ever need certainty, delete the cache and
re-run with `hash_mode: full`.

Both trade-offs are pinned by tests in `tests/test_drive_index.py`, so they stay
deliberate rather than becoming folklore.

## Per-machine setup

`.dvc/config` holds the Colab path, since that is where training happens. On any
other machine, override it locally — `.dvc/config.local` is gitignored:

```bash
# macOS with Drive for Desktop
dvc remote modify --local drivestore url \
  "$HOME/Library/CloudStorage/GoogleDrive-<you>/My Drive/TeamPrigge/SoftwareTools/BlueSpotter/dvcstore"
```

`params.yaml` similarly holds Colab paths for `drive.root` and
`data.nmslices_root`. Point them at your own mount when working locally, but do
not commit that change.

## CI

`.github/workflows/ci.yml` runs on every pull request. GitHub runners have no
access to the lab's Drive, so CI checks what it *can* check: that `dvc.yaml`
parses and the DAG resolves, that `dvc.lock` is not stale relative to the stage
definitions, that the committed validation report has zero errors, that no
indexed file was missing from Drive at the time the index was produced, and that
the resolver/indexer/validator tests pass against a synthetic Drive tree.
Anything needing real images or a GPU runs in Colab.

## A note on the current train/test split

`validate` reports **group leakage** as a warning: the same mouse — and in at
least one case the left and right hemisphere of the same physical section —
appears in both `train.csv` and `test.csv`.

This matters because two sections from one animal share that animal's biology,
its staining batch, and its imaging session. A model that has seen one of them
has partly memorised the other, so the held-out score measures generalisation to
a *new section* rather than to a *new animal*. For a tool intended to compare
results across laboratories, generalising to new animals is the claim that
counts, and the honest unit of splitting is therefore the mouse.

It is a warning rather than an error because fixing it means re-splitting the
data, which is your call, not the pipeline's. Once you have a grouped split, set
`dvc.fail_on_group_leak: true` in `params.yaml` and CI will keep it that way.
Exact counts and the list of affected animals appear in
`reports/validation.json` and `reports/validation_detail.json` after you run
`dvc repro validate` on a machine with Drive mounted.

## MLflow ties into this

`log_data_provenance()` in `src/bluespotter/mlflow_utils.py` runs inside every
training run and tags it with the git commit, whether the working tree was dirty,
the manifest md5s, and the `dataset_hash` from each index summary. It also logs
`grouped_split=false` when leakage is present, so a leaky run can never be
mistaken for a clean held-out result when you are scanning the MLflow UI months
later.

MLflow answers "what did I try and what happened?". DVC answers "what exactly
went in, and can I get it back?". Keeping both costs almost nothing, and each
covers the other's blind spot.
