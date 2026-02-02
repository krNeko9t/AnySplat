from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .dataset_custom import DatasetCustom, DatasetCustomCfgWrapper
from .types import BatchedExample, DataShim, Stage


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


def worker_init_fn(worker_id: int) -> None:
    # Mirror PyTorch worker seeding behavior; dataset may use numpy.
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    seed = int(info.seed) % (2**32 - 1)
    import random, numpy as np  # local import

    random.seed(seed)
    np.random.seed(seed)


def get_data_shim(encoder: torch.nn.Module) -> DataShim:
    """Collect encoder-provided batch shims (e.g., normalization)."""
    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined(batch: BatchedExample) -> BatchedExample:
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined


DatasetCfgWrapper = DatasetCustomCfgWrapper


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cfgs: list[DatasetCfgWrapper],
        data_loader_cfg: DataLoaderCfg,
        dataset_shim: Callable[[torch.utils.data.Dataset, Stage], torch.utils.data.Dataset] = lambda d, _: d,
    ) -> None:
        super().__init__()
        if len(dataset_cfgs) != 1:
            raise ValueError("InstSeg DataModule currently supports exactly one dataset config.")
        self.dataset_cfgs = dataset_cfgs
        self.data_loader_cfg = data_loader_cfg
        self.dataset_shim = dataset_shim

    def _make_loader(self, stage: Stage) -> DataLoader:
        (wrapper,) = self.dataset_cfgs
        cfg = wrapper.custom
        ds = DatasetCustom(cfg, stage)
        ds = self.dataset_shim(ds, stage)
        stage_cfg = getattr(self.data_loader_cfg, stage)
        generator = None
        if stage_cfg.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(stage_cfg.seed)
        return DataLoader(
            ds,
            batch_size=stage_cfg.batch_size,
            shuffle=(stage == "train"),
            num_workers=stage_cfg.num_workers,
            persistent_workers=(None if stage_cfg.num_workers == 0 else stage_cfg.persistent_workers),
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

    def train_dataloader(self):
        return self._make_loader("train")

    def val_dataloader(self):
        return self._make_loader("val")

    def test_dataloader(self):
        return self._make_loader("test")

