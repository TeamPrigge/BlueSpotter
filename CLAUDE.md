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
11. **`data/*` in `.gitignore`** would swallow DVC metafiles — there are explicit
    negations for `*.dvc` and `.gitignore`. Small committed metrics go to `reports/`.

## 8. Known scientific issue — train/test group leakage

`validate` reports this as a **warning**, deliberately. The same animals appear in
both `train.csv` and `test.csv` (e.g. DHC-0929, 0932, 0934, 0935, 0936, 0893, 0464,
0703), and at least one physical section is split by hemisphere — `DHC-0935 slice4`
left in train, right in test.

Why it matters: two sections from one animal share its biology, staining batch and
imaging session, so a model that has seen one has partly memorised the other.
Held-out scores therefore measure generalisation to a *new section*, not a *new
animal*. For a platform whose purpose is cross-laboratory comparability, the honest
unit of splitting is the **mouse**.

It is a warning not an error because re-splitting is the user's scientific decision.
Once a grouped (leave-mice-out) split exists, set `dvc.fail_on_group_leak: true` in
`params.yaml` and CI will hold the line. Exact counts appear in
`reports/validation.json` after running the pipeline in Colab with Drive mounted.

**This has not been fixed. It is arguably the highest-value open item.**

## 9. Current state (end of the handoff session)

Branch **`feat/dvc-data-versioning`**, 5 commits ahead of `main`:

```
e541b19  Make the package installable so notebook shell cells work   <-- NOT PUSHED
3e896a0  Make the Colab config check survive an older clone
dc55514  Add Colab data-versioning notebook so nothing runs locally
141920b  Move MLflow to a SQLite backend store homed on Drive; add Model Registry
3db5ff1  Add DVC data versioning over Google Drive + MLflow provenance
```

- `origin/feat/dvc-data-versioning` is at `d91ed5d` (= `3e896a0`). **`e541b19` is
  unpushed** and it is the commit that fixes the `ModuleNotFoundError`.
- `main` is still at `57a9b69` — it contains **none** of this work. Both notebooks
  therefore set `BRANCH = 'feat/dvc-data-versioning'`; switch to `'main'` after merge.
- 52 tests pass, `ruff` clean. Tests use a synthetic Drive tree — no Drive or network
  needed, so they run anywhere.

**Verified working** (against a synthetic Drive tree and real MLflow 3.14, in a
sandbox — *not* yet against the real Drive):
DVC DAG + `repro` + `push`/`pull` round-trip; silent-Drive-edit detection invalidating
the train stage; missing-file detection; leakage detection; MLflow restore/checkpoint/
integrity/rotation/WAL/concurrent-write-backup; model registration with provenance
tags; recovery of runs+registry after deleting the local disk; legacy `mlruns/`
migration recovering 3 runs; CI scripts in both bootstrap and locked states.

**Never executed against the real Drive or a GPU.** Nobody has yet run
`00_data_versioning.ipynb` end to end, so there is no real `dvc.lock`, no real
`reports/*.json`, and the legacy `mlruns/` history is not yet migrated.

## 10. Do this next, in order

1. **Push `e541b19`** (see §11 for the token).
2. **Run `notebooks/00_data_versioning.ipynb`** in Colab, CPU runtime, with
   `BRANCH = 'feat/dvc-data-versioning'`. Expect the first `dvc.lock`, real
   `reports/*.json`, and the actual leakage numbers. First run is slow (hashes all
   Drive files); later runs only re-hash changes.
3. **Run its section 6 once** to migrate the legacy `mlruns/` history — *before* any
   new training run, for the reason in §7.1.
4. **Run `notebooks/01_train_cellpose_sam.ipynb`** on GPU. Confirm the run appears in
   MLflow with `dataset_hash_train` and a registered `models:/bluespotter-lc/1`.
5. **Open a PR** and merge to `main`, then flip `BRANCH` to `'main'` in both notebooks.
6. Then consider, roughly in value order:
   - **Grouped leave-mice-out split** (§8) — biggest scientific win.
   - **Real segmentation metrics**: held-out IoU / Dice / AP at IoU thresholds, per
     mouse. Training loss alone cannot tell you if the model is good, and this is what
     makes cross-lab comparison meaningful. Currently only losses are logged.
   - **`notebooks/02_predict.ipynb` + `src/bluespotter/predict.py`** — inference on new
     images producing masks + a features CSV. The user has said this is what the lab
     will actually use day to day, far more often than training.
   - Neuromelanin quantification and per-neuron feature extraction (LC-Seg objectives).
   - Prefect orchestration is in the vision doc but was judged **premature** — it wants
     a persistent worker and Colab is ephemeral. Revisit only when training moves to a
     persistent machine and runs unattended.
   - Wrapping Cellpose as an MLflow `pyfunc` would enable `mlflow models serve`;
     worth it when serving moves off the hand-rolled `deploy/cloudrun` app.

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
