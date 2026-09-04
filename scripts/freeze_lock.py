"""Generate the freeze lock for one experiment config (or all of them).

    python scripts/freeze_lock.py +experiment=<X>      # one recipe
    python scripts/freeze_lock.py --all                # all 22, serially

This is a pure-CPU script on purpose. Driving the check through ``src/main.py``
would mean standing up a Trainer just to reach ``setup()``, binding a CPU-only
job to a GPU node. Here we compose the config, build the model, run
``BaseWrapper.apply_freeze()`` -- the only place ``requires_grad`` reaches its final
state -- and serialise with the *same* ``src.freeze_contract.capture`` the runtime
check uses.

**--all spawns one subprocess per recipe, serially.** Building these models costs
roughly 5GB of fp32 each; looping in-process over 22 of them exhausts memory and
takes the machine down with it. A subprocess returns every byte on exit, and one
recipe that fails to build does not take the other 21 with it.

Weights are never loaded. Every route into weights -- ``pretrained_weights``
pointing at a cluster checkpoint, the ``hf:`` route, and the VGGT-1B backbone pull
inside the arch constructors -- is construct + ``load_state_dict``, so it changes
parameter *values* only. Names, shapes, dtypes and requires_grad, every column of
the fingerprint, are identical either way. Skipping them keeps this script
offline, laptop-sized, and independent of paths that only exist on the cluster.
"""

import argparse
import difflib
import importlib.abc
import importlib.machinery
import os
import subprocess
import sys
import types

REPO_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

# Compiled / CUDA-only dependencies that need not exist on the machine generating
# locks. None of them takes part in parameter construction; they are imported at
# module import time by rendering, visualisation and clustering code this script
# never calls. Any accidental use raises immediately.
STUBBED_ROOTS = {
    "gsplat", "diff_gaussian_rasterization", "diff_surfel_rasterization",
    "torch_scatter", "pytorch3d", "e3nn", "hdbscan", "open3d", "pycolmap",
    "pillow_heif", "moviepy",
}


class _StubModule(types.ModuleType):
    __path__ = []  # make every stub a package so submodules resolve

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def _boom(*args, **kwargs):
            raise RuntimeError(f"stubbed dependency used: {self.__name__}.{name}")

        _boom.__name__ = name
        return _boom


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in STUBBED_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


def _install_stubs():
    for root in list(STUBBED_ROOTS):  # a real install always wins over the stub
        try:
            __import__(root)
            STUBBED_ROOTS.discard(root)
        except Exception:
            pass
    sys.meta_path.append(_StubFinder())


# Emptying pretrained_weights closes all three checkpoint routes at once, at the
# config level rather than by patching each one: get_model() then takes the plain
# constructor branch for IGGT, SegVGGT and AnySplat alike.
NO_WEIGHTS = "model.encoder.pretrained_weights="


def _skip_backbone_download():
    """The VGGT-1B pull lives inside the arch constructors, below the config."""
    from src.model.vggt.models.vggt import VGGT

    VGGT.from_pretrained = classmethod(lambda cls, *a, **k: cls())


def build(experiment, overrides):
    """Compose the config, build the wrapper, and run setup("fit")."""
    from hydra import compose, initialize_config_dir

    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    from src.model.arch import get_model
    from src.model.arch.iggt import EncoderIGGTCfg
    from src.model.arch.segvggt import EncoderSegVGGTCfg

    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO_ROOT, "config")):
        cfg_dict = compose(
            config_name="main",
            overrides=[f"+experiment={experiment}", NO_WEIGHTS] + list(overrides),
        )
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    model = get_model(cfg.model.encoder, getattr(cfg.model, "decoder", None))

    if isinstance(cfg.model.encoder, EncoderSegVGGTCfg):
        from src.model.wrapper.segvggt_wrapper import SegVGGTWrapper as Wrapper
    elif isinstance(cfg.model.encoder, EncoderIGGTCfg):
        from src.model.wrapper.iggt_wrapper import IGGTWrapper as Wrapper
    else:
        from src.model.wrapper.anysplat_wrapper import AnySplatWrapper as Wrapper

    wrapper = Wrapper(
        cfg.optimizer, cfg.test, cfg.train, model, get_losses(cfg.loss), StepTracker()
    )
    wrapper.apply_freeze()
    return wrapper


def generate_one(experiment, overrides):
    from src import freeze_contract

    path = freeze_contract.lock_path(experiment)
    before = path.read_text() if path.exists() else ""
    after = freeze_contract.capture(build(experiment, overrides), experiment)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(after)

    # Print the difference unconditionally, and never ask for confirmation. The
    # signature that matters is the git diff on the committed lock; a terminal
    # prompt would be a second, less informative signature that also deadlocks
    # batch regeneration wherever there is no TTY.
    diff = list(difflib.unified_diff(
        before.splitlines(), after.splitlines(),
        fromfile=f"{path} (old)", tofile=f"{path} (new)", lineterm="",
    ))
    print("\n".join(diff) if diff else f"{path}: unchanged")
    return path


def experiment_names():
    config_dir = os.path.join(REPO_ROOT, "config", "experiment")
    return sorted(
        f[: -len(".yaml")] for f in os.listdir(config_dir) if f.endswith(".yaml")
    )


def generate_all():
    """One subprocess per recipe, serially: peak memory stays at a single model."""
    names = experiment_names()
    failed = []
    for i, name in enumerate(names, 1):
        print(f"\n=== [{i}/{len(names)}] {name} ===", flush=True)
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), f"+experiment={name}"],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            failed.append(name)
    print(f"\n{len(names) - len(failed)}/{len(names)} locks generated.")
    if failed:
        print("failed to build: " + ", ".join(failed))
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="regenerate every lock, serially")
    parser.add_argument("overrides", nargs="*", help="hydra overrides, including +experiment=<X>")
    args = parser.parse_args()

    if args.all:
        return generate_all()

    chosen = [o for o in args.overrides if o.startswith("+experiment=")]
    if len(chosen) != 1:
        parser.error("pass exactly one +experiment=<X> (or --all)")
    experiment = chosen[0].split("=", 1)[1]

    _install_stubs()
    _skip_backbone_download()
    generate_one(experiment, [o for o in args.overrides if o not in chosen])
    return 0


if __name__ == "__main__":
    sys.exit(main())
