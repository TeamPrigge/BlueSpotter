"""Parse BlueSpotter file names into structured metadata.

Every labelled LC file on Drive carries its own provenance in its name. Nobody
planned that as a schema — it accreted across three cohorts, two microscopes and
several people — but it is nonetheless the only place where mouse ID, channel,
hemisphere and (crucially) anterior-posterior position are recorded per file.
This module is the single place that knows how to read it, so the manifests can
be regenerated from Drive at any time instead of being hand-maintained.

Conventions seen in the wild, in the order we try them:

  1. AP-in-name (the useful one, ~500 files)
        TH_DHC-2076-5.35_R_seg.npy
        HA_DHC_1351-5.30_R_seg.npy      (underscore instead of hyphen)
        HA_DHC-0632-5.20_R .seg.npy     (stray space, dot before seg)
     -> channel=TH|HA, mouse=DHC-2076, ap_mm=-5.35, side=R
     The number after the mouse ID is the distance behind bregma in mm, always
     written positive; we store it negative because that is how the atlas reads.

  2. Ordinal-AP (NM_lowtiter_histology, Olympus vsi exports)
        DHC_0464_02.vsi - ap2_TH_left_seg.npy
     -> mouse=DHC-0464, ap_index=2, channel=TH, side=L
     `ap2` is a section ordinal, not millimetres: it orders slices within an
     animal but is not comparable across animals. Kept separate from ap_mm on
     purpose — mixing the two would silently corrupt any AP regression.

  3. Slice-index (NM_hightiter_histology_ED, Zeiss czi exports)
        DHC_0929_slice4_TH_R.tif  +  DHC_0929_slice4_TH_R_cp_masks.png
     -> mouse=DHC-0929, slice=slice4, channel=TH, side=R.  No AP information.

  4. Slidescanner tiles (no per-file AP, no hemisphere)
        M1_PM_13_merged.ome_CH2_reduced_seg.npy
     -> mouse=M1, slice=PM_13, channel=CH2.

Anything that matches none of these is returned as `kind="unknown"` rather than
guessed at, so discovery can report it instead of inventing metadata.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

# Bregma is ~ -5.0 to -5.9 mm for the mouse LC. Anything outside this window is
# almost certainly not an AP coordinate (a magnification, a date, a version), so
# we refuse to read it as one.
AP_MM_MIN, AP_MM_MAX = 4.5, 6.5

_SIDE_MAP = {
    "l": "L", "left": "L", "leftg": "L",
    "r": "R", "right": "R", "rightg": "R",
}

# Animal-ID prefixes are per-cohort (DHC, DBT, ...) and new ones appear whenever
# a new breeding line does. Capturing the prefix rather than hardcoding one means
# a new line costs nothing — and, critically, keeps DBT-0033 distinct from
# DHC-0033. Collapsing those would merge two different animals into one group and
# quietly break the leave-mice-out split.
_ID = r"(?P<strain>[A-Z]{2,4})[_-](?P<mouse>\d{3,4})"

# 1. TH_DHC-2076-5.35_R  /  HA_DHC_1351-5.30_R  /  TH_DBT-0033-5.70_R
_AP_MM = re.compile(
    r"^(?P<channel>[A-Za-z0-9]+)"
    r"[_-]" + _ID +
    r"[-_](?P<ap>\d\.\d{1,2})"
    r"[_ ]+(?P<side>[LRlr]|left|right)\b",
    re.IGNORECASE,
)

# 2. DHC_0464_02.vsi - ap2_TH_left
_AP_ORDINAL = re.compile(
    r"^" + _ID + r"[_-](?P<run>\d+)\.vsi\s*-\s*ap(?P<ap>\d+)"
    r"_(?P<channel>[A-Za-z0-9]+)_(?P<side>left|right|[LRlr])",
    re.IGNORECASE,
)

# 3. DHC_0929_slice4_TH_R  (animal first)
_SLICE_IDX = re.compile(
    r"^" + _ID + r"[_-](?P<slice>slice\d+)"
    r"_(?P<channel>[A-Za-z0-9]+)_(?P<side>[LRlr]|left|right)\b",
    re.IGNORECASE,
)

# 3b. TH-DHC-1142_slice2-R  (channel first, hyphens as separators)
_SLICE_IDX_CH_FIRST = re.compile(
    r"^(?P<channel>[A-Za-z0-9]+)[_-]" + _ID +
    r"[_-](?P<slice>slice\d+)"
    r"[_-](?P<side>[LRlr]|left|right)\b",
    re.IGNORECASE,
)

# 4. M1_PM_13_merged.ome_CH2_reduced
_SLIDESCANNER = re.compile(
    r"^(?P<mouse>M\d+)_(?P<slice>[A-Z]{2}_\d+)_merged\.ome_(?P<channel>CH\d+)",
    re.IGNORECASE,
)

# Suffixes that mark a file as a label rather than an image.
_MASK_SUFFIXES = ("_cp_masks", "_seg", " .seg", "_seg_2")


@dataclass(frozen=True)
class Parsed:
    """Metadata recovered from one file name."""

    kind: str          # ap_mm | ap_ordinal | slice_index | slidescanner | unknown
    mouse: str = ""    # normalised to DHC-0929 / M1
    channel: str = ""
    side: str = ""     # L | R | "" when the file is not lateralised
    slice: str = ""
    ap_mm: float | None = None      # millimetres from bregma, negative
    ap_index: int | None = None     # section ordinal within an animal

    @property
    def has_ap(self) -> bool:
        """True only for real, cross-animal-comparable AP coordinates."""
        return self.ap_mm is not None

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["ap_mm"] = "" if self.ap_mm is None else f"{self.ap_mm:.2f}"
        d["ap_index"] = "" if self.ap_index is None else str(self.ap_index)
        return d


def strip_role_suffix(name: str) -> tuple[str, str]:
    """Split `name` into (stem, role) where role is 'image' or 'mask'.

    Cellpose writes masks as `<stem>_cp_masks.png` next to `<stem>.tif`, and
    `<stem>_seg.npy` for the combined image+mask container. Normalising the stem
    is what lets us pair the two halves back up.
    """
    base = name
    for ext in (".png", ".npy", ".tif", ".tiff", ".jpg", ".jpeg"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    for suf in sorted(_MASK_SUFFIXES, key=len, reverse=True):
        if base.lower().endswith(suf.lower()):
            return base[: -len(suf)].rstrip(), "mask"
    return base, "image"


def _norm_side(raw: str) -> str:
    return _SIDE_MAP.get(raw.strip().lower(), "")


def _norm_channel(raw: str) -> str:
    # Channel names are open-ended (TH, HA, DBH, NET, CH1...); new markers get
    # added as the platform takes on new cohorts, so we normalise case rather
    # than validate against a fixed list that would reject them.
    return raw.strip().upper()


def _mouse(m: re.Match[str]) -> str:
    return f"{m.group('strain').upper()}-{m.group('mouse')}"


def parse(name: str) -> Parsed:
    """Parse a file name into `Parsed`. Never raises; unknown -> kind='unknown'."""
    stem, _role = strip_role_suffix(name)

    m = _AP_MM.match(stem)
    if m:
        ap = float(m.group("ap"))
        # Guard against reading a magnification or a version as a coordinate.
        if AP_MM_MIN <= ap <= AP_MM_MAX:
            return Parsed(
                kind="ap_mm",
                mouse=_mouse(m),
                channel=_norm_channel(m.group("channel")),
                side=_norm_side(m.group("side")),
                ap_mm=-ap,
            )

    m = _AP_ORDINAL.match(stem)
    if m:
        return Parsed(
            kind="ap_ordinal",
            mouse=_mouse(m),
            channel=_norm_channel(m.group("channel")),
            side=_norm_side(m.group("side")),
            ap_index=int(m.group("ap")),
        )

    for pattern in (_SLICE_IDX, _SLICE_IDX_CH_FIRST):
        m = pattern.match(stem)
        if m:
            return Parsed(
                kind="slice_index",
                mouse=_mouse(m),
                channel=_norm_channel(m.group("channel")),
                side=_norm_side(m.group("side")),
                slice=m.group("slice").lower(),
            )

    m = _SLIDESCANNER.match(stem)
    if m:
        return Parsed(
            kind="slidescanner",
            mouse=m.group("mouse").upper(),
            channel=_norm_channel(m.group("channel")),
            slice=m.group("slice").upper(),
        )

    return Parsed(kind="unknown")
