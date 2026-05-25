"""Enumerate ``cluster_NNN.ply`` under seg3d_split folders."""

from __future__ import annotations

import re
from pathlib import Path

_CLUSTER_PLY_PATTERN = re.compile(r"^cluster_(\d+)\.ply$", re.IGNORECASE)


def list_cluster_split_plys(split_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted `(cluster_index, path)` matching seg3d_split export naming."""
    if not split_dir.is_dir():
        raise NotADirectoryError(str(split_dir))
    out: list[tuple[int, Path]] = []
    for child in sorted(split_dir.iterdir()):
        if not child.is_file() or child.suffix.lower() != ".ply":
            continue
        m = _CLUSTER_PLY_PATTERN.match(child.name)
        if not m:
            continue
        out.append((int(m.group(1)), child))
    out.sort(key=lambda x: x[0])
    return out
