"""Tests for ``src.trace_render.cluster_paths`` (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ANY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ANY))

from src.trace_render.cluster_paths import list_cluster_split_plys  # noqa: E402


def test_list_cluster_split_plys_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        list_cluster_split_plys(tmp_path / "missing")


def test_list_cluster_split_plys_orders_and_skips_unclustered(tmp_path: Path) -> None:
    d = tmp_path / "split"
    d.mkdir()
    (d / "cluster_002.ply").write_text("x")
    (d / "cluster_000.ply").write_text("x")
    (d / "unclustered.ply").write_text("x")
    (d / "readme.txt").write_text("x")

    pairs = list_cluster_split_plys(d)
    assert pairs == [
        (0, d / "cluster_000.ply"),
        (2, d / "cluster_002.ply"),
    ]


def test_list_cluster_split_case_insensitive(tmp_path: Path) -> None:
    d = tmp_path / "split"
    d.mkdir()
    (d / "CLUSTER_001.ply").write_text("x")
    out = list_cluster_split_plys(d)
    assert len(out) == 1 and out[0][0] == 1
