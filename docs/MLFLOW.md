# Experiment tracking in BlueSpotter

## What changed and why

The run database now lives on Drive as a SQLite file, but SQLite only ever opens
a copy on local disk. Two reasons.

**The Model Registry requires a database-backed store.** MLflow's docs are
explicit: "Model Registry functionality requires a database-backed store." The
LC-Seg objectives say *every released model should be registered within the MLflow
Model Registry*. On the old `file://.../mlruns` store that was not difficult, it
was impossible. Relatedly, MLflow now lists the file system backend as **Legacy**,
"in maintenance mode and will not receive further updates", and slated for
removal; SQLite is the default. The `MLFLOW_ALLOW_FILE_STORE=true` flag the old
setup needed was MLflow warning about exactly this.

**SQLite cannot safely live on the Drive mount.** SQLite's correctness rests on
POSIX advisory locking (`fcntl`) to coordinate readers and writers, and on atomic
rename plus honest `fsync` to commit. A FUSE-backed Drive mount emulates the
filesystem API loosely and syncs in the background, so a lock can appear taken
when it is not and a commit can land half-written. The result is
`database disk image is malformed`, and it tends to appear mid-training. Putting
the `.db` on the mount and pointing MLflow at it looks like it works right up
until it destroys your history.

So the two jobs that were conflated are now separated:

| | Location | Why |
|---|---|---|
| Durable home of the DB | `{drive_root}/mlflow/mlflow.db` | survives Colab resets, backed up by Google, visible to the lab |
| Working copy | `{local_cache}/mlflow.db` on `/content` | where SQLite's locking and `fsync` actually behave |
| Artifacts | `{drive_root}/mlartifacts` | write-once blobs; FUSE handles these fine |

## The lifecycle

```
session start   restore()     Drive .db ──▶ local .db     (verified, schema upgraded)
during training checkpoint()  local .db ──▶ Drive .db     (every N minutes)
run end         checkpoint()  local .db ──▶ Drive .db     (always)
```

Copies use SQLite's **online backup API**, not `cp`. The backup API is designed to
read a database another connection is writing to, so a checkpoint taken mid-run is
a consistent database rather than a torn file. `cp` gives you neither guarantee.

## Crash safety

A Colab session can vanish without warning — idle timeout, GPU reclaim, a closed
tab. Two mechanisms bound the damage.

**Periodic checkpoints.** `mlflow.checkpoint_every_minutes` (default 5) publishes
the DB to Drive on a background thread while training runs, so a crash costs at
most that many minutes of logged history rather than the whole run. It is
time-based because cellpose's `train_seg` exposes no per-epoch callback. A failing
checkpoint is logged and swallowed — losing a checkpoint must never kill a run.

**Never leave Drive without an intact database.** A checkpoint writes to a
temporary file on Drive, verifies it with `PRAGMA integrity_check`, copies the
current database to `.bak`, and only then moves the new file into place. At every
instant at least one valid database exists on Drive, and `restore()` falls back to
`.bak` if the primary is unreadable.

One subtlety worth recording, because the obvious implementation is wrong: the
integrity check runs on the **destination**, not on the live source. Checking the
source would seem safer, but `PRAGMA integrity_check` against a database that
MLflow currently holds write locks on fails with `SQLITE_BUSY`, which is
indistinguishable from real corruption. Guarding on that would refuse every
checkpoint during a run — precisely when checkpoints matter. Verifying the copy is
both honest and stricter: it validates the exact bytes being published, and a
genuinely malformed source cannot produce a copy that passes.

## Commands

```bash
python -m bluespotter.mlflow_store status       # where things are, and are they intact
python -m bluespotter.mlflow_store restore      # Drive -> local (before training)
python -m bluespotter.mlflow_store checkpoint   # local -> Drive (after training)
python -m bluespotter.mlflow_store upgrade      # apply MLflow schema migrations
python -m bluespotter.mlflow_store migrate      # import legacy mlruns/ history
```

`train.run()` calls restore, periodic checkpoint and final checkpoint for you.

## Migrating your existing runs

**Do this before your first new training run.** Notebook section 4:

```bash
python -m bluespotter.mlflow_store migrate
```

### The trap this wrapper exists to avoid

`mlflow migrate-filestore` does **not** merge. If the target database already
exists it asks `Overwrite? [y/N]`, and answering yes replaces the entire
database — every existing run, every registered model, every model version.
Verified on a throwaway copy:

| | runs | registered models | model versions |
|---|---|---|---|
| before | 1 | 1 | 1 |
| after `y` | 3 (the migrated ones) | **0** | **0** |

There is no `--yes` flag and no merge mode. Two consequences the wrapper handles:

- **It never migrates into a live database.** The import goes into a brand-new
  temporary file, so the prompt cannot appear at all, and the result is installed
  only if the current database has nothing to lose. If it does, the command
  refuses and tells you what it would have destroyed. `--force` overrides, after
  backing the current database up to Drive.
- **The subprocess gets no stdin.** If a future MLflow version prompts anyway, it
  receives EOF and exits non-zero instead of hanging a Colab cell forever.

Needs mlflow ≥ 3.10, hence the bump in `requirements.txt`. The old `mlruns/`
folder is deliberately left in place — delete it only after confirming your runs
appear in the UI.

## A note on WAL sidecars

SQLite in WAL mode keeps recent commits in a `-wal` file beside the database, and
WAL mode is recorded in the database header, so it survives a copy. Two problems
follow, both handled in `mlflow_store`:

- A database copied to Drive **without** its `-wal` is missing committed
  transactions. So every published copy is folded into a single self-contained
  file (`wal_checkpoint(TRUNCATE)`, then `journal_mode=DELETE`) before it lands.
- A `-wal` can outlive the database it belonged to. If a *different* database is
  later placed at that path, SQLite finds the stale WAL and replays it onto the
  new file, corrupting it. So sidecars are removed wherever a database is
  replaced or discarded, not merely tidied up afterwards.

## The UI

Notebook cell 8 serves the local SQLite DB with the artifact root on Drive. Two
changes from before:

- Bound to `127.0.0.1` instead of `0.0.0.0`. Colab's `proxyPort` connects locally,
  so listening on every interface bought nothing.
- `MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true` is **kept**, because MLflow 3.x
  validates the `Host` header and the Colab proxy rewrites it; without it the UI
  returns 400. Binding to localhost is what limits the exposure.

Training and the UI now both open the same SQLite file on local disk. That is
fine — coordinating concurrent access is what SQLite's locking is for, and it
works properly there, which was the whole reason for moving off the mount.

## Model Registry

Each run registers its weights under `mlflow.registered_model_name`
(`bluespotter-lc`), and the version is tagged with `dataset_hash_train`,
`git_commit_short` and `grouped_split` from the DVC provenance layer. So a
registry entry answers "which data and which code produced this?" on its own.

This is what lets `deploy/` refer to `models:/bluespotter-lc/3` instead of a
filename on Drive that someone can overwrite.

Cellpose weights are not an MLflow "flavor", so what gets versioned is the logged
artifact rather than a loadable `pyfunc` model. That is enough for lineage,
promotion and rollback. Wrapping Cellpose in a `pyfunc` would additionally allow
`mlflow models serve`; worth doing when serving moves off the hand-rolled Cloud
Run app in `deploy/cloudrun`.

## How this relates to DVC

They answer different questions and neither substitutes for the other:

- **DVC** — what exactly went in, and can I get it back? (`dvc.lock` pins the
  dataset content hash.)
- **MLflow** — what did I try, what happened, and which weights came out?

`log_data_provenance()` is the join between them: every run carries the git
commit, whether the tree was dirty, the manifest md5s and the dataset hashes. See
[DVC.md](DVC.md).

## If you outgrow this

The SQLite-on-Drive arrangement is single-writer by nature: it assumes one Colab
session at a time. That is fine for one person training. Once two people train
concurrently, or you want the UI up without a notebook running, move to a hosted
Postgres and set `MLFLOW_TRACKING_URI` — `config.tracking_uri()` already gives the
env var precedence, so nothing in the code changes. `params.yaml` keeps
`mlflow.tracking_uri` for the same purpose.
