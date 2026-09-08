"""Validation must draw the same frames at every step; training must not.

`fixed_views_and_shape` pins the view *count* and the resolution, not *which*
frames -- those came off the global RNG, which the training loop advances
between validations, so every val point sat on a different draw (ticket 13.1).
This runs against the sampler alone: no dataset, no model, milliseconds.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.view_sampler.view_sampler_bounded_fixed import (
    ViewSamplerBoundedFixed,
    ViewSamplerBoundedFixedCfg,
)

N_VIEWS = 60


def _sampler(stage):
    cfg = ViewSamplerBoundedFixedCfg(
        name="bounded_fixed",
        num_context_views=4,
        num_target_views=0,
        min_distance_between_context_views=45,
        max_distance_between_context_views=192,
        min_distance_to_context_views=0,
        warm_up_steps=0,
        initial_min_distance_between_context_views=0,
        initial_max_distance_between_context_views=0,
        max_img_per_gpu=8,
        min_gap_multiplier=2,
        max_gap_multiplier=6,
    )
    return ViewSamplerBoundedFixed(cfg, stage, False, False, None)


def _draw(sampler, scene, poison):
    """Sample `scene`, having first advanced the global RNG by `poison` draws."""
    torch.manual_seed(poison)
    extrinsics = torch.eye(4).expand(N_VIEWS, 4, 4)
    intrinsics = torch.eye(3).expand(N_VIEWS, 3, 3)
    context, _, _ = sampler.sample(scene, 4, extrinsics, intrinsics)
    return tuple(context.tolist())


def test_val_draw_is_independent_of_the_global_rng():
    sampler = _sampler("val")
    draws = {_draw(sampler, "scene_0007", poison) for poison in range(8)}
    assert len(draws) == 1


def test_val_draw_still_differs_between_scenes():
    sampler = _sampler("val")
    draws = {_draw(sampler, f"scene_{i:04d}", 0) for i in range(16)}
    assert len(draws) > 1, "one seed for all scenes would evaluate 74 copies of one draw"


def test_train_draw_still_jitters():
    sampler = _sampler("train")
    draws = {_draw(sampler, "scene_0007", poison) for poison in range(8)}
    assert len(draws) > 1


def test_context_shape_is_unchanged():
    for stage in ("train", "val"):
        context = _draw(_sampler(stage), "scene_0007", 0)
        assert len(context) == 4
        assert all(0 <= i < N_VIEWS for i in context)


def test_seed_does_not_depend_on_the_process_hash_salt():
    """`hash()` is salted per process; a val draw must survive a restart."""
    sampler = _sampler("val")
    assert sampler.scene_generator("scene_0007").initial_seed() == 175074570330504273
