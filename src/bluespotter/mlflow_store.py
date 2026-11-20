"""MLflow backend store: SQLite database homed on Google Drive, operated locally.

WHY NOT JUST PUT THE .db ON DRIVE AND POINT MLFLOW AT IT
-------------------------------------------------------
Because SQLite would corrupt it. SQLite's durability rests on two things the
Colab Drive mount does not provide: POSIX advisory locking (`fcntl`), which it
uses to coordinate readers and writers, and atomic rename plus honest `fsync`,
which it uses to commit. A FUSE-backed Drive mount emulates the filesystem API
loosely and syncs in the background, so a lock can appear to be taken when it is
not and a commit can land half-written. The failure is not hypothetical and it is
not graceful: you get `database disk image is malformed`, usually while training.

So this module splits the two jobs that were conflated:

  Drive  is the durable **home**. The .db lives there, survives Colab resets, is
         backed up by Google, and is visible to the lab like everything else.
  Local  disk is the **working copy**. SQLite only ever opens a file on
         /content, where locking and fsync behave properly.

The database is restored from Drive at session start and checkpointed back
during and after training. Checkpoints use SQLite's own online backup API rather
than `cp`, so a copy taken while MLflow is mid-write is still a consistent
database rather than a torn file.

CRASH SAFETY
------------
Writing directly over the Drive copy would mean a session that dies mid-copy
destroys the only history. So a checkpoint writes to a temporary file on Drive,
verifies it with `PRAGMA integrity_check`, rotates the current copy to `.bak`,
and only then puts the new file in place. At every instant at least one intact
database exists on Drive. `restore()` will fall back to `.bak` if the primary is
unreadable.

Run standalone:

    python -m bluespotter.mlflow_store restore      # Drive -> local, before training
    python -m bluespotter.mlflow_store checkpoint   # local -> Drive, after training
    python -m bluespotter.mlflow_store migrate      # import legacy mlruns/ file store
    python -m bluespotter.mlflow_store status
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import Config, load_config


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def drive_db(cfg: Config) -> Path:
    """Durable home of the database, on Drive."""
    m = cfg.mlflow
    return cfg.drive_root / m.get("db_subdir", "mlflow") / m.get("db_name", "mlflow.db")


def local_db(cfg: Config) -> Path:
    """Working copy on local disk, where SQLite is safe to operate."""
    return cfg.local_cache / cfg.mlflow.get("db_name", "mlflow.db")


def artifact_root(cfg: Config) -> Path:
    """Artifacts stay on Drive: they are write-once blobs, which FUSE handles fine."""
    return cfg.drive_root / cfg.mlflow.get("artifact_subdir", "mlartifacts")


def sqlite_uri(path: Path) -> str:
    # Four slashes: sqlite:/// + an absolute POSIX path.
    return f"sqlite:///{Path(path).resolve()}"


# --------------------------------------------------------------------------- #
# integrity
# --------------------------------------------------------------------------- #
def integrity_ok(path: Path) -> bool:
    """True if `path` is a readable SQLite database that passes its own check."""
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return False
    try:
        con = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, timeout=10)
        try:
            result = con.execute("PRAGMA integrity_check").fetchone()
            return bool(result) and result[0] == "ok"
        finally:
            con.close()
    except sqlite3.DatabaseError:
        return False


_SIDECARS = ("-wal", "-shm", "-journal")


def _remove_with_sidecars(path: Path) -> None:
    """Delete a database and its WAL/SHM companions.

    Removing the sidecars is not tidiness, it is correctness. WAL mode is recorded
    in the database header and therefore survives a copy, so a `-wal` file can
    outlive the database it belonged to. If a *different* database is later put at
    that same path, SQLite will find the stale WAL and try to replay it — which
    corrupts the new file. Anywhere a database is replaced or discarded, its
    sidecars have to go with it.
    """
    path.unlink(missing_ok=True)
    for suffix in _SIDECARS:
        Path(str(path) + suffix).unlink(missing_ok=True)


def _make_self_contained(path: Path) -> None:
    """Fold any WAL content into the main file so the copy stands alone.

    A database destined for Drive, or about to be renamed, must be a single file:
    the `-wal` will not necessarily travel with it, and a database missing its WAL
    is missing committed transactions.
    """
    con = sqlite3.connect(str(path), timeout=60)
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.execute("PRAGMA journal_mode=DELETE")
        con.commit()
    except sqlite3.DatabaseError:
        pass   # not WAL, or nothing to fold — either way the file is already whole
    finally:
        con.close()
    for suffix in _SIDECARS:
        Path(str(path) + suffix).unlink(missing_ok=True)


def _online_backup(src: Path, dst: Path) -> None:
    """Copy a SQLite database consistently, even while it is being written to.

    `pages` + `sleep` make SQLite retry rather than fail when another connection
    holds a write lock, which is the normal state of affairs during training.
    Copying in page batches also means a long backup does not block writers for
    its whole duration.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    _remove_with_sidecars(dst)
    source = sqlite3.connect(str(src), timeout=60)
    try:
        target = sqlite3.connect(str(dst), timeout=60)
        try:
            source.backup(target, pages=256, sleep=0.05)
        finally:
            target.close()
    finally:
        source.close()
    _make_self_contained(dst)


# --------------------------------------------------------------------------- #
# restore: Drive -> local
# --------------------------------------------------------------------------- #
def restore(cfg: Config, force: bool = False) -> dict[str, Any]:
    """Make a usable local database, seeded from Drive if history exists there."""
    d, l = drive_db(cfg), local_db(cfg)
    l.parent.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"drive_db": str(d), "local_db": str(l)}

    if l.exists() and not force:
        if integrity_ok(l):
            info["action"] = "kept-existing-local"
            print(f"  [restore] local DB already present and intact: {l}")
            return info
        print(f"  [restore] local DB failed integrity check, re-seeding: {l}")
        _remove_with_sidecars(l)

    source = None
    if integrity_ok(d):
        source = d
    else:
        bak = d.with_suffix(d.suffix + ".bak")
        if integrity_ok(bak):
            print(f"  [restore] primary Drive DB unusable; falling back to {bak.name}")
            source = bak

    if source is None:
        info["action"] = "fresh"
        print(f"  [restore] no usable DB on Drive — starting a fresh one at {l}")
        # Touch an empty database so `mlflow db upgrade` has something to work on.
        sqlite3.connect(str(l)).close()
        return info

    # The Drive copy is at rest, but use the backup API anyway: it validates the
    # source as it reads and never produces a torn destination.
    _online_backup(source, l)
    if not integrity_ok(l):
        raise RuntimeError(f"restored DB failed integrity check: {l}")
    info["action"] = "restored"
    info["bytes"] = l.stat().st_size
    print(f"  [restore] {source} -> {l}  ({l.stat().st_size / 1024:.0f} KiB)")
    return info


# --------------------------------------------------------------------------- #
# checkpoint: local -> Drive
# --------------------------------------------------------------------------- #
def checkpoint(cfg: Config, quiet: bool = False) -> dict[str, Any]:
    """Publish the local database to Drive without ever leaving Drive empty."""
    d, l = drive_db(cfg), local_db(cfg)
    if not l.exists():
        if not quiet:
            print(f"  [checkpoint] nothing to publish, no local DB at {l}")
        return {"action": "noop-no-local"}

    # Deliberately NOT integrity-checking the live local DB here. While training
    # runs, MLflow holds write locks, and `PRAGMA integrity_check` against a busy
    # database fails with SQLITE_BUSY — indistinguishable from real corruption.
    # Guarding on that would refuse every checkpoint during a run, which is
    # exactly when checkpoints matter. Instead: attempt the backup, then verify
    # the *result*, which is at rest and can be checked honestly. A genuinely
    # malformed source cannot produce a passing copy, so the guarantee that Drive
    # is never overwritten with garbage is preserved — and it is now checked on
    # the bytes actually being published rather than on a proxy for them.
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.with_name(d.name + f".tmp{os.getpid()}")
    try:
        _online_backup(l, tmp)
    except sqlite3.DatabaseError as e:
        _remove_with_sidecars(tmp)
        raise RuntimeError(
            f"local DB failed to back up (integrity), refusing to publish to Drive: {l}: {e}"
        ) from e
    if not integrity_ok(tmp):
        _remove_with_sidecars(tmp)
        raise RuntimeError("checkpoint produced a file that failed its integrity "
                           "check; Drive left untouched")

    # Rotate, then swap. Order matters: the .bak is written before the primary is
    # touched, so a crash at any point leaves at least one intact database.
    bak = d.with_suffix(d.suffix + ".bak")
    if d.exists():
        try:
            shutil.copy2(d, bak)
        except OSError as e:  # non-fatal: we still have tmp + the live local copy
            print(f"  [checkpoint] warning: could not refresh {bak.name}: {e}")
    # Clear the destination's sidecars first: the incoming file is self-contained,
    # and a leftover -wal from the old database would be replayed onto it.
    for suffix in _SIDECARS:
        Path(str(d) + suffix).unlink(missing_ok=True)
    try:
        os.replace(tmp, d)          # atomic where supported
    except OSError:
        shutil.move(str(tmp), str(d))   # Drive mounts sometimes reject rename

    size = d.stat().st_size
    if not quiet:
        print(f"  [checkpoint] {l} -> {d}  ({size / 1024:.0f} KiB)")
    return {"action": "published", "bytes": size, "drive_db": str(d)}


def checkpoint_quietly(cfg: Config) -> None:
    """Best-effort checkpoint that never interrupts training."""
    try:
        checkpoint(cfg, quiet=True)
    except Exception as e:  # losing a checkpoint beats losing a run
        print(f"  [checkpoint] skipped: {type(e).__name__}: {e}")


class PeriodicCheckpointer:
    """Publish the DB to Drive every N minutes for the duration of a `with` block.

    A Colab session can vanish at any moment — idle timeout, GPU reclaim, a
    browser tab closing. Without this, an eight-hour run that dies at hour seven
    loses every metric it logged. The online backup API is explicitly designed to
    copy a database that another connection is writing to, so this is safe to run
    alongside training rather than something that needs to pause it.

    Uses a daemon thread so it can never hold up interpreter shutdown.
    """

    def __init__(self, cfg: Config, minutes: float | None = None):
        self.cfg = cfg
        if minutes is None:
            minutes = float(cfg.mlflow.get("checkpoint_every_minutes", 0) or 0)
        self.interval = minutes * 60.0
        self._stop: Any = None
        self._thread: Any = None
        self.n_checkpoints = 0

    def _loop(self) -> None:
        import threading
        assert isinstance(self._stop, threading.Event)
        while not self._stop.wait(self.interval):
            try:
                checkpoint(self.cfg, quiet=True)
                self.n_checkpoints += 1
            except Exception as e:
                print(f"  [checkpoint] periodic attempt failed: {type(e).__name__}: {e}")

    def __enter__(self) -> PeriodicCheckpointer:
        if self.interval <= 0:
            return self
        import threading
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="mlflow-db-checkpoint")
        self._thread.start()
        print(f"  DB checkpointing    : every {self.interval / 60:g} min to Drive")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=120)


# --------------------------------------------------------------------------- #
# schema + migration
# --------------------------------------------------------------------------- #
def db_upgrade(cfg: Config) -> bool:
    """Bring the local DB's schema up to date. MLflow refuses to start otherwise."""
    uri = sqlite_uri(local_db(cfg))
    r = subprocess.run([sys.executable, "-m", "mlflow", "db", "upgrade", uri],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print(f"  [db upgrade] failed:\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
        return False
    print("  [db upgrade] schema current")
    return True


def _db_contents(path: Path) -> dict[str, int]:
    """Row counts for the tables that matter, or {} if the DB is unusable/absent."""
    if not integrity_ok(path):
        return {}
    out: dict[str, int] = {}
    con = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=10)
    try:
        for table in ("runs", "registered_models", "model_versions", "experiments"):
            try:
                out[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                out[table] = 0
    finally:
        con.close()
    return out


def _has_content(counts: dict[str, int]) -> bool:
    """True if the DB holds anything a user would be upset to lose."""
    return any(counts.get(k, 0) for k in ("runs", "registered_models", "model_versions"))


def migrate_filestore(cfg: Config, source: Path | None = None,
                      force: bool = False) -> int:
    """Import an existing `mlruns/` file store into the SQLite database.

    READ THIS BEFORE CHANGING ANYTHING HERE
    ---------------------------------------
    `mlflow migrate-filestore` does NOT merge. If the target database already
    exists it prompts "Overwrite? [y/N]", and answering yes **replaces the whole
    database** — every existing run, every registered model, every model version
    is gone. Verified empirically: a target holding 1 run / 1 registered model /
    1 model version came out the other side with 3 migrated runs and 0 registry
    entries. There is no `--yes` flag and no merge mode.

    So this function never migrates into a live database. It migrates into a
    brand-new temporary file (which means the prompt cannot appear at all), then
    only installs the result if the current database has nothing to lose. If it
    does have content, we stop and say so rather than quietly destroying it.
    `force=True` overrides, after backing the current database up.

    stdin is closed for the subprocess as a second line of defence: if a future
    MLflow version prompts anyway, it gets EOF and exits non-zero instead of
    hanging a Colab cell forever.
    """
    src = Path(source) if source else cfg.mlruns_dir
    if not src.exists():
        print(f"  [migrate] no file store at {src} — nothing to import")
        return 0

    restore(cfg)
    existing = _db_contents(local_db(cfg))
    if _has_content(existing) and not force:
        print(f"  [migrate] REFUSING: the database already contains "
              f"{existing.get('runs', 0)} run(s), "
              f"{existing.get('registered_models', 0)} registered model(s), "
              f"{existing.get('model_versions', 0)} model version(s).")
        print("  [migrate] `mlflow migrate-filestore` replaces the target database "
              "rather than merging into it,")
        print("            so importing now would delete all of the above.")
        print("            Migrate before your first new run, or accept the loss "
              "with: ... migrate --force")
        return 1

    # A path that does not exist -> MLflow cannot prompt.
    tmp = local_db(cfg).with_name(f"migrated_{os.getpid()}.db")
    _remove_with_sidecars(tmp)

    print(f"  [migrate] {src}  ->  {tmp} (staging)")
    r = subprocess.run(
        [sys.executable, "-m", "mlflow", "migrate-filestore",
         "--source", str(src), "--target", sqlite_uri(tmp)],
        capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL,
    )
    print(r.stdout[-3000:] or "", end="")
    if r.returncode != 0 or not integrity_ok(tmp):
        print(r.stderr[-2000:])
        _remove_with_sidecars(tmp)
        print("  [migrate] FAILED — nothing was changed; mlruns/ and the existing "
              "database are untouched.")
        print("  [migrate] Note: needs mlflow >= 3.10.")
        return 1

    migrated = _db_contents(tmp)
    print(f"  [migrate] staged OK: {migrated.get('runs', 0)} run(s), "
          f"{migrated.get('experiments', 0)} experiment(s)")

    # Preserve whatever is currently on Drive before swapping the local DB.
    if _has_content(existing):
        print("  [migrate] --force: backing the current database up to Drive first")
        try:
            checkpoint(cfg, quiet=True)
        except RuntimeError as e:
            print(f"  [migrate] could not back up first: {e}")
            _remove_with_sidecars(tmp)
            return 1

    target = local_db(cfg)
    _make_self_contained(tmp)
    _remove_with_sidecars(target)
    os.replace(tmp, target)
    db_upgrade(cfg)
    checkpoint(cfg)
    print("  [migrate] done. mlruns/ is left in place as a fallback — delete it "
          "only after\n            confirming the runs appear in the UI.")
    return 0


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def status(cfg: Config) -> dict[str, Any]:
    d, l = drive_db(cfg), local_db(cfg)
    bak = d.with_suffix(d.suffix + ".bak")

    def describe(p: Path) -> str:
        if not p.exists():
            return "absent"
        age = time.time() - p.stat().st_mtime
        ok = "ok" if integrity_ok(p) else "CORRUPT"
        return f"{p.stat().st_size / 1024:8.0f} KiB  {ok:8}  modified {age / 60:.0f} min ago"

    print(f"  backend         : {cfg.mlflow.get('backend', 'file')}")
    print(f"  tracking URI    : {cfg.tracking_uri()}")
    print(f"  artifact root   : {artifact_root(cfg)}")
    print(f"  Drive DB        : {d}\n                    {describe(d)}")
    print(f"  Drive DB backup : {describe(bak)}")
    print(f"  local DB        : {l}\n                    {describe(l)}")
    legacy = cfg.mlruns_dir
    if legacy.exists():
        n = sum(1 for _ in legacy.glob("*/meta.yaml"))
        print(f"  legacy filestore: {legacy}  ({n} experiment dir(s)) "
              f"-- run `python -m bluespotter.mlflow_store migrate` to import")
    return {"drive_db": str(d), "local_db": str(l)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["restore", "checkpoint", "migrate", "status", "upgrade"])
    ap.add_argument("--params", default=None)
    ap.add_argument("--force", action="store_true",
                    help="restore: overwrite the local DB. "
                         "migrate: proceed even though it will discard existing runs "
                         "and the registry (backed up to Drive first)")
    ap.add_argument("--source", default=None, help="migrate: path to an mlruns/ dir")
    args = ap.parse_args(argv)

    cfg: Config = load_config(args.params)
    if args.action == "restore":
        restore(cfg, force=args.force)
    elif args.action == "checkpoint":
        checkpoint(cfg)
    elif args.action == "upgrade":
        return 0 if db_upgrade(cfg) else 1
    elif args.action == "migrate":
        return migrate_filestore(cfg, args.source, force=args.force)
    else:
        status(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
