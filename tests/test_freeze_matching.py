"""`freeze_keywords` is a glob language with an "except": these are its rules.

`BaseModelWrapper.apply_freeze` is called unbound against a stand-in, so this runs in
milliseconds and never builds a 5GB backbone. The real-model evidence -- 192
tensors / 9.44M flipping between the two `phys_query_arm_*` locks, and nothing
else -- lives in the locks themselves and in ticket 09.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.wrapper.base_wrapper import BaseModelWrapper


class _Stand_In:
    """Just enough of a wrapper for `apply_freeze`: names, and the keywords."""

    def __init__(self, names, keywords, frozen_at_construction=()):
        self._params = {
            n: nn.Parameter(torch.zeros(1), requires_grad=n not in frozen_at_construction)
            for n in names
        }
        self.optimizer_cfg = SimpleNamespace(freeze_keywords=list(keywords))
        self.global_rank = 1  # silence the rank-0 logging

    def named_parameters(self):
        return self._params.items()

    def trainable(self):
        return {n for n, p in self._params.items() if p.requires_grad}


NAMES = [
    "aggregator.frame_blocks.0.attn.qkv.linear.weight",
    "aggregator.frame_blocks.0.attn.qkv.lora.lora_A.weight",
    "aggregator.frame_blocks.0.mlp.fc1.weight",
    "aggregator.instance_cross_blocks.0.attn.qkv.weight",
    "query_physgm.mlp.0.weight",
]


def freeze(keywords, **kw):
    stand_in = _Stand_In(NAMES, keywords, **kw)
    BaseModelWrapper.apply_freeze(stand_in)
    return stand_in.trainable()


def test_glob_replaces_bare_substring():
    """The migration's contract: `x` becomes `*x*` and nothing else changes."""
    assert freeze(["*frame_blocks*"]) == {
        "aggregator.instance_cross_blocks.0.attn.qkv.weight",
        "query_physgm.mlp.0.weight",
    }


def test_bare_substring_no_longer_matches():
    """A keyword left un-migrated matches nothing -- and the zero-hit guard says so,
    loudly, rather than silently training a backbone that was meant to be frozen."""
    with pytest.raises(ValueError, match="matched no parameters"):
        freeze(["frame_blocks"])


def test_negation_carves_an_exception_out_of_a_glob():
    """The one recipe bare substrings cannot express: the blocks, except their LoRA."""
    trainable = freeze(["*frame_blocks*", "!*.lora.*"])
    assert "aggregator.frame_blocks.0.attn.qkv.lora.lora_A.weight" in trainable
    assert "aggregator.frame_blocks.0.attn.qkv.linear.weight" not in trainable
    assert "aggregator.frame_blocks.0.mlp.fc1.weight" not in trainable


def test_last_match_wins():
    """Order is the whole disambiguation rule; reversing it reverses the answer."""
    lora = "aggregator.frame_blocks.0.attn.qkv.lora.lora_A.weight"
    assert lora in freeze(["*frame_blocks*", "!*.lora.*"])
    assert lora not in freeze(["!*.lora.*", "*frame_blocks*"])


def test_negation_deselects_it_does_not_unfreeze():
    """`!` narrows what this function selects; it never flips requires_grad back on.
    LoRA freezes the base weights of the layers it wraps at construction time, and
    an un-freezing `!` would silently train base + adapter together."""
    base = "aggregator.frame_blocks.0.attn.qkv.linear.weight"
    assert base not in freeze(["!*frame_blocks*"], frozen_at_construction={base})


def test_zero_hit_guard_covers_negated_keywords_too():
    """A typo in a `!` keyword is exactly as dangerous as one in a plain keyword:
    it silently freezes what was meant to stay trainable."""
    with pytest.raises(ValueError, match=r"!\*\.typo\.\*"):
        freeze(["*frame_blocks*", "!*.typo.*"])
