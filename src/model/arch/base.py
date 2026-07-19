"""Encoder 抽象基类:backbone + heads 的组装体,forward 返回 ``EncoderOutput``。"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

from src.dataset.types import BatchedViews, DataShim
from src.model.outputs import EncoderOutput

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
    ) -> EncoderOutput:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
