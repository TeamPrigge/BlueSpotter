"""Tests for the Drive-homed SQLite backend store.

The whole point of this module is that the database survives things going wrong,
so most of these tests are about failure: corrupt files, interrupted copies, and
writes happening during a backup. A "it copied the file" test would prove nothing.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
import yaml

from bluespotter.config import Config, load_config
from bluespotter.mlflow_store import (
    PeriodicCheckpointer,
    artifact_root,
    checkpoint,
    drive_db,
    integrity_ok,
    local_db,
    restore,
    sqlite_uri,
    status,
)

BASE_PARAMS = {
    "drive": {
        "root": None,          # filled per-test
        "data_subdir": "data",
        "model_subdir": "model",
        "mlruns_subdir": "mlruns",
    },
    "data": {"local_cache": None, "image_filter": "_img", "mask_filter": "_masks"},
    "model": {"pretrained": "cpsam", "name": "bluespotter_lc"},
    "train": {"n_epochs": 1},
    "mlflow": {
        "experiment_name": "t",
        "backend": "sqlite",
        "db_subdir": "mlflow",
        "db_name": "mlflow.db",
        "artifact_subdir": "mlartifacts",
        "checkpoint_every_minutes": 0,
        "registered_model_name": "bluespotter-lc",
        "tracking_uri": "",
    },
}


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch) -> Config:
    """A Config whose 'Drive' and local cache are both temp dirs."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    import copy
    raw = copy.deepcopy(BASE_PARAMS)
    raw["drive"]["root"] = str(tmp_path / "drive")
    raw["data"]["local_cache"] = str(tmp_path / "local")
    p = tmp_path / "params.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(p)


def _make_db(path: Path, rows: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, note TEXT)")
    con.executemany("INSERT INTO runs (note) VALUES (?)", [(f"r{i}",) for i in range(rows)])
    con.commit()
    con.close()


def _count(path: Path) -> int:
    con = sqlite3.connect(str(path))
    try:
        return con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def test_tracking_uri_points_at_local_disk_not_drive(cfg: Config):
    """The critical invariant: SQLite must never be told to open a file on Drive."""
    uri = cfg.tracking_uri()
    assert uri.startswith("sqlite:///")
    assert str(cfg.local_cache) in uri
    assert str(drive_db(cfg)) not in uri


def test_artifacts_stay_on_drive(cfg: Config):
    assert str(cfg.drive_root) in str(artifact_root(cfg))


def test_env_var_overrides_backend(cfg: Config, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    assert cfg.tracking_uri() == "http://localhost:5000"


def test_sqlite_uri_is_absolute_with_three_slashes(tmp_path):
    assert sqlite_uri(tmp_path / "x.db") == f"sqlite:///{(tmp_path / 'x.db').resolve()}"


# --------------------------------------------------------------------------- #
# integrity detection
# --------------------------------------------------------------------------- #
def test_integrity_ok_accepts_a_real_db(tmp_path):
    db = tmp_path / "a.db"
    _make_db(db)
    assert integrity_ok(db) is True


@pytest.mark.parametrize("content", [b"", b"not a database at all", b"SQLite format 3\x00trunc"])
def test_integrity_ok_rejects_junk(tmp_path, content):
    db = tmp_path / "bad.db"
    db.write_bytes(content)
    assert integrity_ok(db) is False


def test_integrity_ok_rejects_missing(tmp_path):
    assert integrity_ok(tmp_path / "nope.db") is False


# --------------------------------------------------------------------------- #
# restore
# --------------------------------------------------------------------------- #
def test_restore_with_nothing_on_drive_creates_a_fresh_db(cfg: Config):
    info = restore(cfg)
    assert info["action"] == "fresh"
    assert local_db(cfg).exists()


def test_restore_seeds_local_from_drive(cfg: Config):
    _make_db(drive_db(cfg), rows=7)
    info = restore(cfg)
    assert info["action"] == "restored"
    assert _count(local_db(cfg)) == 7


def test_restore_keeps_an_intact_local_db(cfg: Config):
    _make_db(drive_db(cfg), rows=1)
    _make_db(local_db(cfg), rows=99)
    info = restore(cfg)
    assert info["action"] == "kept-existing-local"
    assert _count(local_db(cfg)) == 99      # local work is not silently discarded


def test_restore_force_overwrites_local(cfg: Config):
    _make_db(drive_db(cfg), rows=3)
    _make_db(local_db(cfg), rows=99)
    restore(cfg, force=True)
    assert _count(local_db(cfg)) == 3


def test_restore_replaces_a_corrupt_local_db(cfg: Config):
    _make_db(drive_db(cfg), rows=4)
    local_db(cfg).parent.mkdir(parents=True, exist_ok=True)
    local_db(cfg).write_bytes(b"corrupted")
    info = restore(cfg)
    assert info["action"] == "restored"
    assert _count(local_db(cfg)) == 4


def test_restore_falls_back_to_backup_when_primary_is_corrupt(cfg: Config):
    """The scenario the .bak exists for: Drive's primary copy got mangled."""
    d = drive_db(cfg)
    _make_db(d.with_suffix(d.suffix + ".bak"), rows=6)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_bytes(b"half-written garbage")
    info = restore(cfg)
    assert info["action"] == "restored"
    assert _count(local_db(cfg)) == 6


def test_restore_with_both_drive_copies_corrupt_starts_fresh_without_raising(cfg: Config):
    d = drive_db(cfg)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_bytes(b"junk")
    d.with_suffix(d.suffix + ".bak").write_bytes(b"junk")
    info = restore(cfg)
    assert info["action"] == "fresh"
    assert integrity_ok(local_db(cfg)) or local_db(cfg).exists()


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
def test_checkpoint_publishes_to_drive(cfg: Config):
    _make_db(local_db(cfg), rows=8)
    info = checkpoint(cfg)
    assert info["action"] == "published"
    assert _count(drive_db(cfg)) == 8


def test_checkpoint_with_no_local_db_is_a_noop(cfg: Config):
    assert checkpoint(cfg)["action"] == "noop-no-local"


def test_checkpoint_refuses_to_overwrite_drive_with_a_corrupt_local(cfg: Config):
    """Never let a damaged working copy destroy good history on Drive."""
    _make_db(drive_db(cfg), rows=10)
    local_db(cfg).parent.mkdir(parents=True, exist_ok=True)
    local_db(cfg).write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="integrity"):
        checkpoint(cfg)
    assert _count(drive_db(cfg)) == 10     # untouched


def test_checkpoint_rotates_previous_copy_to_bak(cfg: Config):
    _make_db(local_db(cfg), rows=2)
    checkpoint(cfg)
    # second round with different content
    local_db(cfg).unlink()
    _make_db(local_db(cfg), rows=5)
    checkpoint(cfg)
    d = drive_db(cfg)
    assert _count(d) == 5
    assert _count(d.with_suffix(d.suffix + ".bak")) == 2


def test_checkpoint_leaves_no_temp_files_behind(cfg: Config):
    _make_db(local_db(cfg), rows=3)
    checkpoint(cfg)
    leftovers = list(drive_db(cfg).parent.glob("*.tmp*"))
    assert leftovers == []


def _wal_db(path: Path, rows: int) -> None:
    """Create a database in WAL mode, so -wal/-shm sidecars exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, note TEXT)")
    con.executemany("INSERT INTO runs (note) VALUES (?)", [(f"r{i}",) for i in range(rows)])
    con.commit()
    con.close()


def test_published_db_is_self_contained_no_wal_sidecars(cfg: Config):
    """A file on Drive must not depend on a -wal that may not travel with it."""
    _wal_db(local_db(cfg), rows=6)
    checkpoint(cfg)
    d = drive_db(cfg)
    assert d.exists()
    assert not Path(str(d) + "-wal").exists()
    assert not Path(str(d) + "-shm").exists()
    assert _count(d) == 6      # WAL contents were folded in, not lost


def test_wal_contents_are_not_lost_when_publishing(cfg: Config):
    """Committed rows living in the WAL must survive the trip to Drive."""
    db = local_db(cfg)
    _wal_db(db, rows=2)
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA journal_mode=WAL")
    con.executemany("INSERT INTO runs (note) VALUES (?)", [("extra",), ("extra2",)])
    con.commit()          # sits in the -wal, not yet in the main file
    try:
        checkpoint(cfg)
        assert _count(drive_db(cfg)) == 4
    finally:
        con.close()


def test_stale_wal_is_cleared_before_a_db_is_replaced(cfg: Config):
    """A leftover -wal from an old DB would be replayed onto the new one."""
    d = drive_db(cfg)
    _wal_db(d, rows=3)
    # Simulate an orphaned sidecar next to the destination.
    Path(str(d) + "-wal").write_bytes(b"stale wal content that must not be replayed")
    _make_db(local_db(cfg), rows=9)
    checkpoint(cfg)
    assert not Path(str(d) + "-wal").exists()
    assert integrity_ok(d)
    assert _count(d) == 9


def test_restore_clears_sidecars_of_a_corrupt_local_db(cfg: Config):
    _make_db(drive_db(cfg), rows=5)
    l = local_db(cfg)
    l.parent.mkdir(parents=True, exist_ok=True)
    l.write_bytes(b"corrupt")
    Path(str(l) + "-wal").write_bytes(b"orphan")
    restore(cfg)
    assert not Path(str(l) + "-wal").exists()
    assert _count(l) == 5


def test_round_trip_preserves_content(cfg: Config):
    _make_db(local_db(cfg), rows=12)
    checkpoint(cfg)
    local_db(cfg).unlink()
    restore(cfg)
    assert _count(local_db(cfg)) == 12


def test_checkpoint_is_consistent_while_the_db_is_being_written(cfg: Config):
    """Backup during concurrent writes must yield a valid DB, not a torn file.

    This is the case a plain `cp` gets wrong and the reason the online backup API
    is used instead.
    """
    db = local_db(cfg)
    _make_db(db, rows=1)
    stop = threading.Event()

    def writer():
        con = sqlite3.connect(str(db), timeout=30)
        i = 0
        while not stop.is_set():
            i += 1
            try:
                con.execute("INSERT INTO runs (note) VALUES (?)", (f"live{i}",))
                con.commit()
            except sqlite3.OperationalError:
                time.sleep(0.005)
        con.close()

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        time.sleep(0.05)
        for _ in range(4):
            checkpoint(cfg, quiet=True)
            assert integrity_ok(drive_db(cfg)), "backup taken mid-write was corrupt"
    finally:
        stop.set()
        t.join(timeout=10)
    assert _count(drive_db(cfg)) >= 1


# --------------------------------------------------------------------------- #
# periodic checkpointer
# --------------------------------------------------------------------------- #
def test_periodic_checkpointer_publishes_while_running(cfg: Config):
    _make_db(local_db(cfg), rows=1)
    with PeriodicCheckpointer(cfg, minutes=0.02 / 60):   # ~20 ms
        time.sleep(0.25)
    assert drive_db(cfg).exists()
    assert integrity_ok(drive_db(cfg))


def test_periodic_checkpointer_disabled_by_zero(cfg: Config):
    _make_db(local_db(cfg), rows=1)
    with PeriodicCheckpointer(cfg, minutes=0) as pc:
        time.sleep(0.05)
    assert pc.n_checkpoints == 0
    assert not drive_db(cfg).exists()


def test_periodic_checkpointer_survives_a_failing_checkpoint(cfg: Config, monkeypatch):
    """A broken checkpoint must never propagate into the training loop."""
    _make_db(local_db(cfg), rows=1)
    import bluespotter.mlflow_store as ms
    monkeypatch.setattr(ms, "checkpoint",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("drive gone")))
    with PeriodicCheckpointer(cfg, minutes=0.02 / 60):
        time.sleep(0.15)     # must not raise


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def test_status_runs_on_an_empty_setup(cfg: Config, capsys):
    status(cfg)
    out = capsys.readouterr().out
    assert "tracking URI" in out
    assert "absent" in out
