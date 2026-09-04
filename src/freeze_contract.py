"""Trainable-parameter fingerprint and the freeze lock contract.

Freezing has exactly three criteria: no autograd graph, no optimizer entry, and
weights that do not change. This module guards the first two at the instant
``BaseWrapper.setup()`` ends: does this run's *actual* trainable set match the
lock the recipe declares?

A fingerprint is fixed-width text -- one line per parameter, sorted by name --
headed by a sha256 over that body. The check compares the body *line by line*;
the hash is a human-facing summary, recomputed from the body rather than trusted,
so the text a reviewer signs off on in git diff is exactly the text enforced. ``base_lr`` is recorded in the header but
takes no part in the hash and is never compared (learning rate satisfies none of
the three criteria). ``dtype`` is both a body column and part of the hash, yet
carries no dedicated assertion: if it changes, the hash collides on its own and
a human is forced to look once.

There is a single capture point: the end of ``setup()``, before the strategy
wraps the module. ``configure_optimizers`` runs *after* the wrap, by which time
the DDP reducer has already registered whatever required grad at wrap time.

The generator (``scripts/freeze_lock.py``) and the checker (``BaseWrapper.setup``)
call the *same* ``capture()`` here. Two serialisers would mean the guard was
checking its own shadow.
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_DIR = REPO_ROOT / "config" / "experiment" / "locks"

_FLAG = {True: "T", False: "-"}
_DTYPE_ALIAS = {
    "float32": "fp32",
    "bfloat16": "bf16",
    "float16": "fp16",
    "float64": "fp64",
}
_DIFF_HEAD_LINES = 24


def lock_path(experiment: str) -> Path:
    """One experiment, exactly one lock. ``experiment`` is the ``+experiment=<X>`` name."""
    return LOCK_DIR / f"{experiment}.lock"


def _dtype_name(dtype) -> str:
    raw = str(dtype).removeprefix("torch.")
    return _DTYPE_ALIAS.get(raw, raw)


def _body_lines(module) -> list[str]:
    """Every parameter, one line each, sorted by name.

    Full coverage rather than the trainable subset: within the subset, "frozen"
    and "no longer exists" are indistinguishable, so an arch refactor that drops
    a module looks exactly like freezing it.
    """
    rows = sorted(module.named_parameters(), key=lambda row: row[0])
    return [
        f"{_FLAG[param.requires_grad]} {_dtype_name(param.dtype):<4} {param.numel():>12}  {name}"
        for name, param in rows
    ]


def structure_hash(body: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(body).encode()).hexdigest()


def capture(module, experiment: str) -> str:
    """Serialise an already-``setup()``-ed LightningModule into lock text."""
    cfg = module.optimizer_cfg
    keywords = list(cfg.freeze_keywords or [])
    body = _body_lines(module)
    header = [
        f"# freeze-lock: {experiment}",
        f"arch: {type(module).__name__}",
        f"freeze_keywords: [{', '.join(keywords)}]",
        f"base_lr: {cfg.lr}",
        f"structure: sha256:{structure_hash(body)}",
        "",
    ]
    return "\n".join(header + body) + "\n"


def lock_body(text: str) -> list[str]:
    """The per-parameter body of a lock: everything after the header's blank line.

    Note what this deliberately does *not* return: the lock's own ``structure:``
    line. Comparing against a hash the lock declares about itself would let a
    hand-edited body pass unnoticed, so the text a human signs off on in git diff
    would not be the text being enforced. The body is the contract; the hash is a
    summary printed for humans and recomputed from the body every time.
    """
    return text.partition("\n\n")[2].splitlines()


def experiment_name() -> str:
    """Identity key: hydra's experiment choice (the yaml filename under config/experiment).

    Not ``wandb.name``: across the 22 recipes, 4 differ from their filename and two
    names are used twice. A duplicated identity key means two recipes share one
    lock, which disables the guard outright.
    """
    from hydra.core.hydra_config import HydraConfig

    return HydraConfig.get().runtime.choices.get("experiment")


def verify(module, experiment: str, output_path) -> None:
    """Recompute the fingerprint and compare it to the lock. A mismatch is fatal.

    Every rank computes and compares its own: it is a pure CPU walk, and freezing
    that disagrees between ranks is the single hardest failure to spot in a loss
    curve.
    """
    path = lock_path(experiment)
    regenerate = f"python scripts/freeze_lock.py +experiment={experiment}"
    actual = capture(module, experiment)

    if not path.exists():
        raise FileNotFoundError(
            f"freeze lock is missing: {path}\n"
            f"experiment: {experiment}\n"
            "Every experiment config must carry a lock -- all 22, no exceptions. "
            "Otherwise 'this recipe has no lock' looks exactly like 'this recipe "
            "needs no freezing'.\n"
            f"generate: {regenerate}\n"
            "Then read the first column (requires_grad) and the dtype column in "
            "git diff, and commit the lock alongside the yaml."
        )

    expected_body = lock_body(path.read_text())
    actual_body = lock_body(actual)
    if actual_body == expected_body:
        return

    # Mismatches usually happen on a remote node. If the difference comes from the
    # environment, a local rerun will not reproduce it and this dump is the only
    # evidence there will ever be.
    dump = Path(output_path) / f"freeze_fingerprint.{experiment}.rank{module.global_rank}.txt"
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(actual)

    diff = list(difflib.unified_diff(
        expected_body, actual_body, fromfile=str(path), tofile="measured", lineterm="", n=0,
    ))
    head = "\n".join(diff[:_DIFF_HEAD_LINES])
    more = max(len(diff) - _DIFF_HEAD_LINES, 0)
    raise ValueError(
        "Trainable-parameter fingerprint does not match the lock; refusing to train.\n"
        f"experiment:  {experiment}\n"
        f"lock:        {path}\n"
        f"measured:    {dump}\n"
        f"regenerate:  {regenerate}\n"
        f"lock={structure_hash(expected_body)[:16]}  "
        f"measured={structure_hash(actual_body)[:16]}\n"
        f"--- diff (first {_DIFF_HEAD_LINES} lines, {more} more) ---\n{head}"
    )
