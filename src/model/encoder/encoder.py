from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn
from dataclasses import dataclass
from src.dataset.types import BatchedViews, DataShim
from ..types import Gaussians
from jaxtyping import Float
from torch import Tensor, nn

from .physics_prediction import PhysicsPrediction
from .physics_property_prediction import PhysicsPropertyPrediction

T = TypeVar("T")


@dataclass
class EncoderOutput:
    gaussians: Gaussians | None
    pred_pose_enc_list: list[Float[Tensor, "batch view 6"]] | None
    pred_context_pose: dict | None
    depth_dict: dict | None
    infos: dict | None
    distill_infos: dict | None
    # Optional instance segmentation embeddings.
    # instance_feat_map: [B, V, N, H, W] (same spatial resolution as decoder output)
    instance_feat_map: Float[Tensor, "batch view n height width"] | None = None
    # gaussian_instance_feat: [B, G, N] (aligned with gaussians order)
    gaussian_instance_feat: Float[Tensor, "batch gaussian n"] | None = None
    # Physics scheme "class" slot. See PhysicsPrediction.
    physics_prediction: PhysicsPrediction | None = None
    # Physics scheme "property" slot. See PhysicsPropertyPrediction.
    physics_property_prediction: PhysicsPropertyPrediction | None = None


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
