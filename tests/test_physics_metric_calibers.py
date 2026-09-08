"""The pooled (instance-weighted) caliber must be recoverable from means alone.

Ticket 13.2: `val/phys_mae_*` is scene-weighted, tickets 04/05 pool over
instances, and the two disagree by a factor of three on the class-lookup gap.
The `_xn` tags carry the missing sufficient statistic through Lightning, which
can only take means.  Aggregation only -- no model, no dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.physics_metrics import aggregate_rows

TAG = "phys_mae_log10_youngs_modulus"


def _aggregate(scenes, per_batch=2):
    """Aggregate `scenes` the way one validation does: per batch, then over batches."""
    rows = [{"phys_n_matched": float(n), TAG: mae} for n, mae in scenes]
    batches = [
        aggregate_rows(rows[i : i + per_batch]) for i in range(0, len(rows), per_batch)
    ]
    keys = {k for b in batches for k in b}
    return {k: sum(b[k] for b in batches) / len(batches) for k in keys}


def test_pooled_is_recovered_from_the_two_means():
    scenes = [(10, 1.0), (30, 0.5), (10, 1.0), (30, 0.5)]
    agg = _aggregate(scenes)
    pooled = agg[f"{TAG}_xn"] / agg["phys_n_matched"]
    expected = sum(n * mae for n, mae in scenes) / sum(n for n, _ in scenes)
    assert abs(pooled - expected) < 1e-12


def test_the_two_calibers_actually_differ():
    """If they agreed, ticket 13.2 would not have been a question."""
    scenes = [(10, 1.0), (30, 0.5), (10, 1.0), (30, 0.5)]
    agg = _aggregate(scenes)
    pooled = agg[f"{TAG}_xn"] / agg["phys_n_matched"]
    assert abs(agg[TAG] - 0.75) < 1e-12       # scene-weighted
    assert abs(pooled - 0.625) < 1e-12        # instance-weighted


def test_existing_tags_are_untouched():
    """This round's 40 points stay comparable: the old tags keep their values."""
    scenes = [(10, 1.0), (30, 0.5)]
    agg = _aggregate(scenes, per_batch=2)
    assert agg[TAG] == 0.75
    assert agg["phys_n_matched"] == 20.0
