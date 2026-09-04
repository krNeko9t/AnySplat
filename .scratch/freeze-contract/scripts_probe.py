"""Ticket 01 probe: measure the REAL trainable set of each experiment config.

Builds the model + wrapper on CPU (no weights, no forward), runs
BaseWrapper.setup("fit") -- the only place requires_grad reaches its final
state -- then dumps, per config:
  - per freeze_keyword hit counts (params + numel)
  - actual requires_grad=True set, rolled up by top-level module
  - dtype histogram of the trainable set
  - param_groups assignment + effective lr, replaying configure_optimizers' rule
"""
import json, os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))
os.environ.setdefault("HYDRA_FULL_ERROR", "1")


# --- CPU-probe stubs -------------------------------------------------------
# Compiled / CUDA-only deps that are absent on this box.  None of them
# participates in *parameter construction*; they are imported at module import
# time by rendering / visualisation / clustering code that this probe never
# calls.  Any accidental use raises immediately (see _Stub.__getattr__).
STUBBED_ROOTS = {
    "gsplat", "diff_gaussian_rasterization", "diff_surfel_rasterization",
    "torch_scatter", "pytorch3d", "e3nn", "hdbscan", "open3d", "pycolmap",
    "pillow_heif", "moviepy",
}
import types as _types
import importlib.abc, importlib.machinery


class _StubMod(_types.ModuleType):
    __path__ = []  # make every stub a package so submodules resolve

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        def _boom(*a, **k):
            raise RuntimeError(f"stubbed dependency used: {self.__name__}.{name}")
        _boom.__name__ = name
        return _boom


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in STUBBED_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubMod(spec.name)

    def exec_module(self, module):
        pass


# real package wins if it is actually installed
for _r in list(STUBBED_ROOTS):
    try:
        __import__(_r)
        STUBBED_ROOTS.discard(_r)
    except Exception:
        pass
sys.meta_path.append(_StubFinder())
# ---------------------------------------------------------------------------

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.model.arch import get_model
from src.model.arch.iggt import EncoderIGGTCfg
from src.model.arch.segvggt import EncoderSegVGGTCfg


def _skip_backbone_download():
    """Construct the VGGT backbone randomly instead of pulling 5GB from HF.

    `from_pretrained` = construct + `load_state_dict`; it changes parameter
    *values* only.  Names, shapes, dtypes and requires_grad -- everything this
    probe measures -- are identical either way, and the `.to(torch.bfloat16)`
    cast in the arch files still runs on the result.
    """
    from src.model.vggt.models.vggt import VGGT
    if not getattr(VGGT, "_probe_patched", False):
        VGGT.from_pretrained = classmethod(lambda cls, *a, **k: cls())
        VGGT._probe_patched = True

    # Same argument for the *second* Hub pull: `pretrained_weights: hf:...` on the
    # AnySplat path routes through `init_anysplat_from_hf`, which is
    # `AnySplat(encoder_cfg, decoder_cfg)` + `load_state_dict` -- another ~5GB
    # download that only changes parameter *values*.  Short-circuit to the bare
    # constructor so the instseg_* configs stay CPU/offline like every other one.
    import src.model.arch as _arch
    if not getattr(_arch, "_probe_patched", False):
        from src.model.arch.anysplat import AnySplat as _AnySplat
        _arch.init_anysplat_from_hf = (
            lambda hf_id, encoder_cfg, decoder_cfg, **k: _AnySplat(encoder_cfg, decoder_cfg)
        )
        _arch._probe_patched = True


def build(exp: str, overrides):
    cfg_dir = os.path.abspath(os.path.dirname(__file__) + "/../../config")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg_dict = compose(config_name="main", overrides=[f"+experiment={exp}"] + list(overrides))
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    decoder_cfg = getattr(cfg.model, "decoder", None)
    model = get_model(cfg.model.encoder, decoder_cfg)
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    losses = get_losses(cfg.loss)
    st = StepTracker()
    if isinstance(cfg.model.encoder, EncoderSegVGGTCfg):
        from src.model.wrapper.segvggt_wrapper import SegVGGTWrapper as W
    elif isinstance(cfg.model.encoder, EncoderIGGTCfg):
        from src.model.wrapper.iggt_wrapper import IGGTWrapper as W
    else:
        from src.model.wrapper.anysplat_wrapper import AnySplatWrapper as W
    wrapper = W(cfg.optimizer, cfg.test, cfg.train, model, losses, st)
    return cfg, cfg_dict, wrapper


def probe(exp, overrides):
    cfg, cfg_dict, wrapper = build(exp, overrides)
    ocfg = cfg.optimizer
    freeze_kw = list(ocfg.freeze_keywords or [])

    pre = {n: p.requires_grad for n, p in wrapper.named_parameters()}
    wrapper.setup("fit")
    post = [(n, p) for n, p in wrapper.named_parameters()]

    # keyword hits (same substring rule as base_wrapper.py:512)
    hits = {kw: {"params": 0, "numel": 0} for kw in freeze_kw}
    for n, p in post:
        for kw in freeze_kw:
            if kw in n:
                hits[kw]["params"] += 1
                hits[kw]["numel"] += p.numel()

    # frozen at construction (before setup) -- LoRA base weights, mask_token, ...
    ctor_frozen = sorted(n for n, rg in pre.items() if not rg)

    def _rollup_key(name, depth):
        parts = name.split(".")
        return ".".join(parts[:depth]) if len(parts) > depth else name

    records = {}
    rollup = {}
    rollup_deep = {}
    dtypes = {}
    total_t = total_all = 0
    for n, p in post:
        total_all += p.numel()
        records[n] = [p.numel(), str(p.dtype), bool(p.requires_grad)]
        if not p.requires_grad:
            continue
        total_t += p.numel()
        for key, table in ((_rollup_key(n, 3), rollup), (_rollup_key(n, 5), rollup_deep)):
            table.setdefault(key, {"params": 0, "numel": 0})
            table[key]["params"] += 1
            table[key]["numel"] += p.numel()
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()

    # replay configure_optimizers grouping
    pg_cfg = list(ocfg.param_groups or [])
    groups = []
    if pg_cfg:
        buckets = [{"kw": list(g.keywords), "mult": g.lr_multiplier, "params": 0, "numel": 0, "dtypes": {}} for g in pg_cfg]
        default = {"kw": ["<default>"], "mult": ocfg.backbone_lr_multiplier, "params": 0, "numel": 0, "dtypes": {}}
        for n, p in post:
            if not p.requires_grad:
                continue
            tgt = default
            for b in buckets:
                if any(k in n for k in b["kw"]):
                    tgt = b
                    break
            tgt["params"] += 1; tgt["numel"] += p.numel()
            tgt["dtypes"][str(p.dtype)] = tgt["dtypes"].get(str(p.dtype), 0) + p.numel()
        groups = buckets + [default]
        mode = "param_groups"
    else:
        kws = list(getattr(ocfg, "new_param_keywords", []) or []) or ["gaussian_param_head", "interm"]
        new = {"kw": kws, "mult": 1.0, "params": 0, "numel": 0, "dtypes": {}}
        bb = {"kw": ["<backbone>"], "mult": ocfg.backbone_lr_multiplier, "params": 0, "numel": 0, "dtypes": {}}
        for n, p in post:
            if not p.requires_grad:
                continue
            tgt = new if any(k in n for k in kws) else bb
            tgt["params"] += 1; tgt["numel"] += p.numel()
            tgt["dtypes"][str(p.dtype)] = tgt["dtypes"].get(str(p.dtype), 0) + p.numel()
        groups = [new, bb]
        mode = "legacy new_param_keywords"

    return {
        "experiment": exp,
        "arch": type(wrapper.encoder if hasattr(wrapper, "encoder") else wrapper).__name__,
        "wrapper": type(wrapper).__name__,
        "lr": ocfg.lr,
        "freeze_keywords": freeze_kw,
        "keyword_hits": hits,
        "ctor_frozen_count": len(ctor_frozen),
        "ctor_frozen_sample": ctor_frozen[:8],
        "total_params_numel": total_all,
        "trainable_numel": total_t,
        "trainable_rollup": rollup,
        "trainable_rollup_deep": rollup_deep,
        "params": records,
        "trainable_dtypes": dtypes,
        "group_mode": mode,
        "groups": groups,
        "all_param_dtypes": _all_dtypes(post),
        "trainable_names": [n for n, p in post if p.requires_grad],
    }


def _all_dtypes(post):
    d = {}
    for n, p in post:
        d[str(p.dtype)] = d.get(str(p.dtype), 0) + p.numel()
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--out", default=None)
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--random-backbone", action="store_true",
                    help="build VGGT randomly instead of downloading facebook/VGGT-1B")
    a = ap.parse_args()
    if a.random_backbone:
        _skip_backbone_download()
    r = probe(a.exp, a.override)
    r["random_backbone"] = bool(a.random_backbone)
    out = a.out or f".scratch/freeze-contract/notes/raw/{a.exp}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=1)
    print(f"OK {a.exp}: trainable {r['trainable_numel']/1e6:.1f}M / {r['total_params_numel']/1e6:.1f}M  dtypes={r['trainable_dtypes']}")
