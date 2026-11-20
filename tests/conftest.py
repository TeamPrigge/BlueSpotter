"""Shared fixtures: a synthetic Drive tree that mirrors the real NM_Slices layout.

These tests never touch Google Drive. They build a miniature version of it on
local disk with the same directory conventions, so the resolver, indexer and
validator can be exercised deterministically in CI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HEADER = ("split,source,microscope,mouse,slice,channel,side,"
          "image_name,image_id,mask_name,mask_id,mask_type")

ED = "NM_hightiter_histology_brains_ED"
LOW = "NM_lowtiter_histology_brains"


def _png_row(split: str, mouse: str, sl: str, side: str, stem: str) -> str:
    return (f"{split},NM_hightiter_histology_ED,Zeiss,{mouse},{sl},TH,{side},"
            f"{stem}.tif,img_{stem},{stem}_cp_masks.png,msk_{stem},cp_masks_png")


def _npy_row(split: str, mouse: str, side: str, stem: str) -> str:
    return (f"{split},NM_lowtiter_histology,Olympus,{mouse},,TH,{side},"
            f"{stem}.npy,id_{stem},{stem}.npy,id_{stem},"
            f"seg_npy (image+mask in one file)")


@pytest.fixture
def drive(tmp_path: Path) -> Path:
    """An empty NM_Slices-shaped tree. Returns the nmslices root."""
    nm = tmp_path / "Data" / "NM_Slices"
    (nm / ED / "processed" / "cropped" / "TH").mkdir(parents=True)
    (nm / ED / "masks" / "TH").mkdir(parents=True)
    (nm / LOW / "masks" / "npy_masks").mkdir(parents=True)
    return nm


@pytest.fixture
def make_png_pair(drive: Path):
    """Create image+mask files on the fake Drive for a cp_masks_png stem."""
    def _make(stem: str, img_bytes: bytes = b"IMG", mask_bytes: bytes = b"MSK") -> None:
        (drive / ED / "processed" / "cropped" / "TH" / f"{stem}.tif").write_bytes(img_bytes)
        (drive / ED / "masks" / "TH" / f"{stem}_cp_masks.png").write_bytes(mask_bytes)
    return _make


@pytest.fixture
def make_npy(drive: Path):
    def _make(stem: str, data: bytes = b"NPY") -> None:
        (drive / LOW / "masks" / "npy_masks" / f"{stem}.npy").write_bytes(data)
    return _make


@pytest.fixture
def write_manifest(tmp_path: Path):
    def _write(name: str, rows: list[str]) -> Path:
        p = tmp_path / name
        p.write_text("\n".join([HEADER, *rows]) + "\n")
        return p
    return _write


@pytest.fixture
def big_file_factory():
    """Write a file of a given size whose middle bytes can be mutated."""
    def _make(path: Path, size: int, fill: bytes = b"\x00") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"HEAD" + fill * (size - 8) + b"TAIL")
        return path

    def _mutate_middle(path: Path) -> None:
        size = path.stat().st_size
        with open(path, "r+b") as fh:
            fh.seek(size // 2)
            fh.write(b"\xff" * 16)

    _make.mutate_middle = _mutate_middle  # type: ignore[attr-defined]
    return _make


os.environ.setdefault("PYTHONHASHSEED", "0")
