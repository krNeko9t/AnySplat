#!/usr/bin/env python3
"""Export instascene ``instance_renders`` PNG tree from ``seg3d_split`` cluster PLYs.

Usage::
    python scripts/export_seg3d_vlm_views.py --job configs/vlm_export/example.export.json

See :mod:`src.trace_render` — trace pipeline stays in ``trace_instance_to_gaussians.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.trace_render.export_tree import export_seg3d_vlm_png_tree
from src.trace_render.job import load_job


def main() -> None:
    p = argparse.ArgumentParser(description="Seg3d split → VLM PNG export (standalone)")
    p.add_argument(
        "--job",
        "-j",
        type=str,
        required=True,
        help="JSON job file (see configs/vlm_export/example.export.json)",
    )
    args = p.parse_args()
    export_seg3d_vlm_png_tree(load_job(args.job))


if __name__ == "__main__":
    main()
