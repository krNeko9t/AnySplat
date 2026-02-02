from __future__ import annotations

from typing import Callable, Literal, TypedDict

from jaxtyping import Bool, Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]


class BatchedViews(TypedDict, total=False):
    # Camera.
    extrinsics: Float[Tensor, "batch view 4 4"]  # c2w
    intrinsics: Float[Tensor, "batch view 3 3"]  # normalized K (fx,fy,cx,cy normalized by W/H)
    near: Float[Tensor, "batch view"]
    far: Float[Tensor, "batch view"]
    index: Int64[Tensor, "batch view"]
    overlap: Float[Tensor, "batch view"]

    # RGB.
    image: Float[Tensor, "batch view 3 height width"]  # by convention, in [-1, 1] after shims

    # Optional geometry supervision.
    depth: Float[Tensor, "batch view height width"]
    valid_mask: Bool[Tensor, "batch view height width"]

    # 3D-consistent instance IDs per pixel.
    # Convention: ignore_id (default 0) means background / invalid / unlabeled.
    instance_mask: Int64[Tensor, "batch view height width"]


class BatchedExample(TypedDict, total=False):
    context: BatchedViews
    target: BatchedViews
    scene: list[str]


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "view 4 4"]
    intrinsics: Float[Tensor, "view 3 3"]
    near: Float[Tensor, "view"]
    far: Float[Tensor, "view"]
    index: Int64[Tensor, "view"]
    overlap: Float[Tensor, "view"]

    image: Float[Tensor, "view 3 height width"]
    depth: Float[Tensor, "view height width"]
    valid_mask: Bool[Tensor, "view height width"]
    instance_mask: Int64[Tensor, "view height width"]


class UnbatchedExample(TypedDict, total=False):
    context: UnbatchedViews
    target: UnbatchedViews
    scene: str


DataShim = Callable[[BatchedExample], BatchedExample]
AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews

