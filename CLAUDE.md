# BlueSpotter — orientation for a Claude session

Read this first. It is the handoff from the session that built the DVC + MLflow
layers (July 2026). Deep detail lives in [docs/DVC.md](docs/DVC.md) and
[docs/MLFLOW.md](docs/MLFLOW.md); this file is the map, the conventions, and the
things that will bite you.

---

## 1. What this is

BlueSpotter fine-tunes **Cellpose-SAM** to segment locus coeruleus (LC) neurons,
and is the implementation of the LC-Seg vision: a standardised, hardware-independent
platform for quantitative LC analysis, so measurements from different labs,
microscopes and animal models become comparable. Longer-term goals include
anatomical localisation within the LC, neuromelanin quantification, and
standardised per-neuron feature extraction. None of those are built yet.

Owner: Matthias Prigge (`prigge.matthias@gmail.com`), TeamPrigge.
Repo: `github.com/TeamPrigge/BlueSpotter`.

## 2. Hard constraints — do not violate these

**Everything runs in Colab. Nothing runs on the user's computer.** This is a stated
project rule ("cloud-only workflow" in README and docs/SETUP.md) and the user has
pushed back when it was broken. Concretely:

- Do **not** hand the user shell commands to run locally. Put the work in a
  notebook cell instead.
- Code lives in GitHub, data and models live in Google Drive, execution happens in
  Colab. Keep that separation.
- Image data never enters git. Model weights never enter git.
- The user has **Colab Pro** and ~1500 compute units. GPU runtime for training,
  CPU runtime for data work.

**Preferences:** be concise and direct; skip preamble; don't pad with caveats.
The user is a neuroscientist, technically fluent, and dislikes being told things
he already knows or being asked to re-explain project context — check this file and
the docs before asking.

## 3. Where things live

| Asset | Location |
|---|---|
| Code | GitHub `TeamPrigge/BlueSpotter` |
| Raw images + masks | Drive `MyDrive/TeamPrigge/Data/NM_Slices/` |
| Project folder on Drive | Drive `MyDrive/TeamPrigge/SoftwareTools/BlueSpotter/` |
| Manifests (authoring copy) | Drive `.../BlueSpotter/data/{training_data/train.csv, test_data/test.csv}` |
| Model weights | Drive `.../BlueSpotter/model/` |
| MLflow DB (durable home) | Drive `.../BlueSpotter/mlflow/mlflow.db` |
| MLflow artifacts | Drive `.../BlueSpotter/mlartifacts/` |
| DVC blob store | Drive `.../BlueSpotter/dvcstore/` |
| Legacy MLflow file store | Drive `.../BlueSpotter/mlruns/` (to be migrated, then deleted) |

All Drive paths in `params.yaml` are written as Colab mount paths
(`/content/drive/MyDrive/...`).

## 4. Repo map

```
src/bluespotter/
  config.py         loads params.yaml; resolves Drive + repo paths; tracking_uri()
  manifest.py       resolve_row(): manifest row -> Drive path. SINGLE SOURCE OF TRUTH
                    for the NM_Slices layout. Also loads image/mask pairs for training.
  manifest_sync.py  DVC stage: copy manifests Drive -> data/manifests/
  drive_index.py    DVC stage: content-index every Drive file a manifest references
  validate.py       DVC stage: schema, duplicates, split overlap, GROUP LEAKAGE
  mlflow_store.py   SQLite backend store: Drive home, local working copy, checkpointing
  mlflow_utils.py   tracking setup, data provenance tags, model registration
  train.py          Cellpose-SAM fine-tuning entry point
  data.py           legacy Drive->local folder caching (only used when use_manifest: false)
  viz.py            LR schedule plot + predicted-vs-GT overlays
  discover.py       DVC stage: crawl the Drive mount, rebuild all three manifests
  naming.py         filename -> mouse/channel/side/AP. Generic strain prefix, so
                    DBT-0033 != DHC-0033. Refuses out-of-range AP rather than guessing.
  prepare.py        write image + precomputed flows to local disk, resumably, so
                    training can stream instead of holding the corpus in RAM
  metrics.py        IoU matching, precision/recall/F1/AP. No ROC: instance
                    segmentation has no bounded true-negative class.
  evaluate.py       DVC stage: score the held-out split, one image at a time
  quality_gate.py   compare metrics to params thresholds; refuse stale metrics
  make_assertions.py  cut the committed assertion crops from real held-out slices
  render.py         reports/QUALITY.md: image | ground truth | prediction panels
  augment.py        horizontal mirror only (no DV flip - it would destroy the
                    orientation signal the planned AP model needs)
  scoring_set.py    blind export of the held-out split for human counting
  ap_scoring.py     AP bins recovered from Csilla's values; weighted-kappa agreement
notebooks/
  00_data_versioning.ipynb    CPU. DVC pipeline + push to Drive/GitHub. RUN FIRST.
  01_train_cellpose_sam.ipynb GPU. Verifies data version, then trains.
dvc.yaml            stages: sync-manifests, validate, index-train, index-test, train
params.yaml          ALL paths + hyperparameters. Never hard-code these in Python.
reports/             committed metrics: validation.json, *_index_summary.json
tests/               52 tests, synthetic Drive tree, no network/Drive needed
.github/workflows/ci.yml
```

## 5. The data-versioning design, and why

The manifests are **lists of links** — filename + Drive file-ID per image/mask pair.
Columns: `split, source, microscope, mouse, slice, channel, side, image_name,
image_id, mask_name, mask_id, mask_type`. Two mask conventions: `cp_masks_png`
(Zeiss czi→tif, ED cohort) and `seg_npy` (Olympus vsi, image+mask in one file).
Currently 60 train / 15 test pairs.

Because they are links, re-exporting a `.czi` or regenerating a `_cp_masks.png`
changes the dataset while the manifest and git stay identical. So:

1. `sync-manifests` snapshots the Drive CSVs into `data/manifests/` where DVC hashes them.
2. `index-train` / `index-test` resolve every row on Drive and record
   size/mtime/content-hash per file into `data/index/*_index.csv`, plus a
   `dataset_hash` summary in `reports/`.
3. `dvc.lock` (committed) therefore pins the exact bytes. A changed image changes the
   index hash, which invalidates the `train` stage and shows as a PR diff.

**The DVC remote is a plain directory on the mounted Drive path, not `dvc://gdrive`.**
DVC retired its shared OAuth client, so the gdrive backend needs a per-user Google
Cloud project. Colab already mounts Drive as a filesystem, so a directory remote
needs no tokens at all. Machine-specific paths belong in `.dvc/config.local`
(gitignored).

## 6. The MLflow design, and why

**The Model Registry requires a database-backed store** — it does not work on the
legacy `mlruns/` file store, which MLflow now lists as maintenance-mode and slated
for removal. The LC-Seg objectives require registering every released model, so a
database was mandatory.

**But SQLite cannot live on the Drive mount.** It needs POSIX advisory locking and
honest `fsync`/atomic-rename; a FUSE Drive mount provides neither, and the result is
`database disk image is malformed`, usually mid-training. So:

- Drive holds the **durable home** of `mlflow.db`.
- SQLite only ever opens a **working copy** on `/content` (local disk).
- `restore()` seeds local from Drive; `checkpoint()` publishes back using SQLite's
  **online backup API** (not `cp`), which is safe while MLflow is writing.
- Checkpoints stage to a temp file, verify with `integrity_check`, rotate the old copy
  to `.bak`, then swap — Drive always holds at least one intact DB.
- `PeriodicCheckpointer` publishes every `mlflow.checkpoint_every_minutes` (default 5)
  on a daemon thread, bounding what a Colab crash costs.

Artifacts stay on Drive; they are write-once blobs and FUSE handles them fine.

`log_data_provenance()` tags every run with git commit, dirty flag, manifest md5s,
`dataset_hash_train/test` and `grouped_split` — this is the join between DVC and
MLflow. DVC answers "what went in, can I get it back?"; MLflow answers "what did I
try, what happened?".

## 7. Gotchas — all of these were found empirically, do not re-learn them the hard way

1. **`mlflow migrate-filestore` REPLACES the target database, it does not merge.**
   Verified: a target with 1 run / 1 registered model / 1 version came out with 3
   migrated runs and **0 registry entries**. There is no `--yes` flag. The wrapper in
   `mlflow_store.migrate_filestore()` stages into a fresh temp file, refuses if the
   current DB has anything to lose, and passes no stdin so a prompt cannot hang a
   Colab cell. **Do not "simplify" it.**
2. **MLflow 3 registration:** `runs:/<id>/<path>` resolves only to a *logged model*,
   so registering a bare artifact directory fails. Use
   `mlflow.create_external_model()` then `models:/<model_id>` — that API exists for
   models whose artifacts live outside MLflow, which is this case.
3. **Never run `PRAGMA integrity_check` on the live local DB** before a checkpoint.
   While MLflow holds write locks it fails with `SQLITE_BUSY`, indistinguishable from
   corruption, and would refuse every checkpoint during a run. Verify the *copy*.
4. **SQLite WAL sidecars:** WAL mode is in the DB header and survives a copy. A copy
   without its `-wal` is missing committed transactions, and a stale `-wal` beside a
   *replaced* DB gets replayed onto it and corrupts it. Published copies are folded
   into one self-contained file; sidecars are removed wherever a DB is replaced.
5. **`!python -m bluespotter.x` in a notebook is a SUBPROCESS** and does not inherit
   `sys.path` edits made in the kernel. Hence `pip install -e .` (pyproject has a
   `[project]` table for exactly this) plus `PYTHONPATH`. This caused a
   `ModuleNotFoundError` the user hit.
6. **Drive mtime granularity:** the hash cache in `drive_index.py` applies git's
   "racily clean" rule — an entry whose mtime is not comfortably older than the
   recorded time is re-read, because Drive mounts round mtime to whole seconds and a
   fast same-size edit would otherwise return a stale hash.
7. **`hash_mode: partial`** (current default) hashes first+last 4 MiB + size. It is
   blind to an edit confined to the middle of a file >8 MiB. `full` is exact but slow
   over Drive. Both trade-offs are pinned by tests.
8. **MLflow UI in Colab needs `MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true`** —
   MLflow 3 validates the `Host` header and the Colab proxy rewrites it, giving
   "Invalid Host header / DNS rebinding" and a blank iframe. Keep the flag. Also
   hard-kill stale servers (`pkill -9 -f mlflow`) or the old one keeps port 5000.
9. **`cellpose.train.train_seg` has no `nchan` argument** (use `min_train_masks`) and
   **no per-epoch callback** — hence time-based DB checkpointing, not epoch-based.
   It prints loss only at epochs 0, 5, then every 10th.
10. **`dvc metrics show` flattens nested JSON.** Metrics files must be scalar-only;
    the per-mouse breakdown lives in `reports/validation_detail.json`, an output
    rather than a metric.
12. **Newer Cellpose `_seg.npy` does not embed the image.** Older ones had an
    `img` key; >=3 records only `filename`, an absolute path on the annotator's
    machine. The old loader caught the KeyError and printed `[skip]`, so ~680 of
    1,386 training rows silently vanished while every report still said 1,386.
    `find_image_for_seg()` resolves it; `load_manifest` now *raises* if more than
    5% of rows fail rather than training on a quietly halved dataset.
13. **`OSError 107` from the Drive mount is usually OOM, not flakiness.**
    `np.load(...).item()` unpickles the whole dict including multi-GB `flows`;
    on a 12 GB VM the kernel kills the FUSE daemon, and every subsequent path
    fails identically to a dropped mount. Hence `del blob; gc.collect()`.
14. **`skimage.io.imsave` silently transposes a leading axis of 4.** A
    `(4, H, W)` flows array round-trips as `(H, W, 4)`, and cellpose's
    `io.imread(labels_file)[1:]` then slices the wrong axis and trains on
    nonsense without raising. `prepare.py` uses `tifffile.imwrite`; a test pins
    the round-trip shape.
15. **cellpose `train_seg` passes `channel_axis` to `_get_batch`, which has
    never accepted it.** Only fires when `normed` is False, i.e. on every
    file-based run and never on an in-memory one — so it survives upstream
    (4.2.1.1 and main). `train._patch_cellpose_get_batch()` works around it and
    becomes a no-op once the signature gains the parameter.
16. **Do not commit anything under `reports/figures/`.** Regenerated on every
    model change; 34 MB per render accumulates in history forever. Gitignored,
    with a test that fails if they are ever tracked again.
17. **Check for an in-progress rebase before committing.** A paused
    `git pull --rebase` leaves a detached HEAD with the tree rewound; committing
    into that silently reverted a fix this session.

11. **`data/*` in `.gitignore`** would swallow DVC metafiles — there are explicit
    negations for `*.dvc` and `.gitignore`. Small committed metrics go to `reports/`.

## 8. Group leakage — FIXED (was the top open item)

The old text here described 8 of 9 test mice also appearing in train. That was
measured on the 60/15 dataset and is **no longer true**. `discover.py` now
assigns splits by a deterministic hash of the mouse ID, so an animal lands
wholly in train or wholly in test.

Measured after the rebuild: **0 mice and 0 slices in both splits.**

`dvc.fail_on_group_leak` is still `false` in params.yaml. It can now be flipped
to `true` — the split has been grouped, so the check should hold rather than
warn. Do that when convenient.

---

## 9. Current state (September 2026)

Branch **`feat/dvc-data-versioning`**, all pushed:

```
72909a1  AP level scoring, outline subset, reference panel
1a1b6a7  Export the held-out split for blind human counting
28d78df  Report image counts from split sizes, not in-memory lists
55b8fd6  Work around a cellpose bug that breaks file-based training
30e6e54  Stop plot_predictions loading the whole manifest
06b6c8f  Train the full dataset by streaming from disk
```

172 tests pass (1 skipped — needs cellpose, which CI does not install).
`main` still has none of this.

### Dataset

| | train | test | AP |
|---|---|---|---|
| rows | 1,386 | 248 | 1,443 |
| dataset_hash | `e1475bd01e2d` | `d3925dd93612` | `a7e8072c2591` |

117 animals, 0 files missing, 0 group leakage. Grew from 68 pairs / 11 mice by
crawling Drive with `discover.py`.

### Training works, on the full dataset

The blocker was memory: `train.py` held every image in RAM, which for 1,386
slices of ~10,000 px is well over 100 GB against 83 GB on the largest Colab
runtime. Fixed by using what cellpose already offers — `train_files` /
`train_labels_files` with `load_files=False`, so `_get_batch` reads only the
current batch. Peak RAM is batch size, not corpus size.

**Full images are preserved. There is no tiling.** An earlier plan to pre-cut
tiles was dropped: cellpose takes a random rotated 256 px crop per image per
batch anyway (`random_rotate_and_resize`, bsize=256), so pre-cutting would fix
the crop pattern and lose augmentation diversity while buying nothing.

`bluespotter.prepare` writes each image and its precomputed flows to local disk
once, resumably, and reports unloadable rows grouped by cause.

Verified end to end on an L4: loss 0.2845 -> 0.0778 reading from files, weights
saved to Drive, registered as `bluespotter-lc` v2, provenance stamped
(`commit=..., train_hash=e1475bd01e2d`). The MLflow SQLite store and Model
Registry have now been exercised in Colab, not just in a sandbox.

**Never yet run:** a full-dataset training run. Only smoke runs (60 images).

### Assertion set + CI

`assertions/` holds 40 real crops from 20 held-out animals, 1,066 labelled
cells, 15.5 MB — a deliberate, test-enforced exception to "image data never
enters git" (synthetic fixtures would pass a model that fails on real slides).
CI scores them on every push and publishes a metrics table to the Actions
summary. `reports/figures/` and `QUALITY.md` are gitignored: they are
regenerated on every model change and 34 MB per render has no place in history.

### Human scoring (in progress, not yet run)

`scoring_set.py` exports the 248 held-out sections for blind counting by three
students plus Csilla. Per-section scale driven by measured soma diameter (a
fixed output box is what made assertion somata 3 px wide at x0.15). Shuffled
opaque IDs; reference counts stay in `key.csv`, which scorers must not see. 50
sections, picked round-robin across animals, are also flagged for outlining.

`ap_scoring.py` recovers AP bins from the bregma values Csilla actually used
rather than picking a number, and scores agreement with quadratic-weighted
kappa so an adjacent-bin miss is not treated like rostral-called-caudal.

**Important:** Csilla judged AP from the image, exactly as the students will.
So there is **no ground truth for AP** — only human judgements. Any AP model can
at best learn human consensus, and inter-rater agreement is simultaneously the
ceiling and the label noise. Do not describe her values as labels.

### Known data problems

- ~7% of manifest rows fail to load. Two causes: one corrupt `.npy`
  (`UnpicklingError`), and images recorded as `C:/Users/iunone/Desktop/...`
  that never reached Drive. `prepare` reports them grouped by cause.
- `ED_timepoints_counts` on Drive marks some sections `bad` in a `curated`
  column. These should probably be excluded from the scoring set.

---

## 10. Do this next, in order

1. **Strain count.** `DHC-` vs `DBT-`/`TYR-` across the manifests. This decides
   whether an AP model is viable and is blocking everything AP-related:

   ```python
   import csv
   from collections import Counter
   rows = (list(csv.DictReader(open('data/manifests/train.csv')))
         + list(csv.DictReader(open('data/manifests/test.csv'))))
   print(Counter(r['mouse'].split('-')[0] for r in rows))
   ```

2. **`python -m bluespotter.prepare --split train --limit 50`** — measures GiB
   per slice. Multiply by ~28. If that exceeds runtime disk, the cache needs
   downscaling or streaming from Drive instead.

3. **`dvc repro sync-manifests`** — the last run reported
   `Version pinned : False`, i.e. it read the live Drive CSV rather than a
   revision in `dvc.lock`.

4. **Full training run.** `LIMIT = None`, `EPOCHS = None` in notebook 01 cell 12.
   At ~70 s/epoch for 55 images, 1,386 x 100 epochs is order of a day on an L4 —
   check compute-unit burn and reconsider 100 epochs first.

5. **`dvc repro evaluate` then `render`**, then set `quality_gate.enabled: true`
   with thresholds a little below what was actually measured.

6. **Then**: `notebooks/02_predict.ipynb` + `predict.py`. The user has said
   inference is what the lab will use day to day, far more than training.

### Open decisions that are the user's, not Claude's

- **AP model viability.** Matthias requires AP training data to be transgenic
  WT-like, excluding virally injected animals. But nearly the whole dataset is
  `DHC-` dbh-cre mice injected with AAV. If step 1 confirms this, the options
  are: widen "undegenerated" to include `hTyrNull`/saline controls (they express
  no tyrosinase, so LC should be anatomically normal); use early post-injection
  timepoints; or image the `+/+` TYR brains that exist but are not in NM_Slices.
- **The condition table.** The manifest has no genotype or condition column, and
  it is **not inferable from cohort folder names** — `NM_hightiter_*` only looks
  like an overexpression cohort. Condition lives in five per-cohort Drive sheets
  with different vocabularies (`Group` C/T, Control/Treatment, Saline/J60).
  Crucially, `CN_sections` shows condition can differ **per hemisphere** of one
  animal ("LC left: hTyrHA, LC right: hTyrNull"), so the key must be
  `(mouse, side)`, not mouse. `ap_scoring.read_wt_mice()` takes this explicitly
  and refuses to guess.
- Csilla's manual counts already exist on Drive (`ED_all_cell_counts` has
  `TH_Count_manual` alongside Cellpose counts; `MC chemogenetics` and
  `high_titer_behav_cells` have `th_counts` by `id, ap, side`). She may not need
  to recount.

---

## 11. Credentials — where the token goes

A Claude session's sandbox can **commit** to the mounted folder but **cannot push**:
no credentials, and `github.com` does not even resolve from it. Two places a token
belongs:

- **For Colab** (already set up): a Colab Secret named **`TOKEN_BlueSpotter`**. Both
  notebooks read it via `userdata.get('TOKEN_BlueSpotter')` and build an
  `https://x-access-token:<token>@github.com/...` remote, so the notebooks can push.
  Nothing secret is stored in the repo. Keep it this way.
- **For a Claude session to push:** plain-text `.env` in the repo root:

  ```
  GITHUB_TOKEN=github_pat_...
  ```

  `.env` is already in `.gitignore` (alongside `*.token` and `credentials.json`), so
  it cannot be committed. Plain text is fine here — but make it a **fine-grained PAT
  scoped to `TeamPrigge/BlueSpotter` only, Contents: read/write, with an expiry**, so
  a leak costs one repo rather than the whole account. Do not paste tokens into chat:
  a token pasted into a previous session's transcript became unrecoverable *and* is
  sitting in a local log in plaintext.

If a token is in `.env`, a session can push with:

```bash
git push "https://x-access-token:${GITHUB_TOKEN}@github.com/TeamPrigge/BlueSpotter.git" HEAD
```

Do not write that URL into `.git/config` — it would persist the token on disk.

## 12. Conventions

- **All tunables in `params.yaml`.** Never hard-code paths or hyperparameters.
- `manifest.resolve_row()` is the only place that knows the NM_Slices layout.
- New DVC stages: add to `dvc.yaml` with explicit `deps`/`params`, and remember
  `PYTHONPATH=src` in the `cmd` so stages work without an editable install.
- Drive-facing stages need `always_changed: true` — Drive changes without git changing.
- Tests must not require Drive, network or a GPU. Use the synthetic tree in
  `tests/conftest.py`.
- `ruff` config is pinned in `pyproject.toml` (`target-version = "py310"`, so the code
  stays runnable on older Colab images — do not let a linter rewrite
  `datetime.timezone.utc` to `datetime.UTC`).
- CI cannot see Drive. It checks the DAG, `dvc.lock` freshness, committed reports and
  the tests. Anything needing images or a GPU runs in Colab.
