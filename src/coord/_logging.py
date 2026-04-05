"""
Runtime coordinate-system logging and assertion helpers.

Enable verbose output with:
    import logging
    logging.getLogger("coord").setLevel(logging.DEBUG)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from torch import Tensor

from .conventions import CameraConvention, ExtrinsicType

if TYPE_CHECKING:
    from .camera_pose import CameraPose

logger = logging.getLogger("coord")


def log_coordinate_op(
    operation: str,
    convention: CameraConvention,
    extrinsic_type: ExtrinsicType,
    shape: Union[tuple, list, "torch.Size"] = (),
    context: str = "",
) -> None:
    """Emit a DEBUG log describing a coordinate-system operation.

    Example output::

        [COORD] load_ply | opencv/c2w | shape=[10000, 4, 4] | scene=garden
    """
    msg = f"[COORD] {operation} | {convention.value}/{extrinsic_type.value}"
    if shape:
        msg += f" | shape={list(shape)}"
    if context:
        msg += f" | {context}"
    logger.debug(msg)


def assert_convention(
    pose: "CameraPose",
    expected_conv: Optional[CameraConvention] = None,
    expected_type: Optional[ExtrinsicType] = None,
    msg: str = "",
) -> None:
    """Raise ``AssertionError`` if *pose* doesn't match expectations."""
    if expected_conv is not None and pose.convention != expected_conv:
        raise AssertionError(
            f"Convention mismatch: expected {expected_conv.value}, "
            f"got {pose.convention.value}. {msg}"
        )
    if expected_type is not None and pose.extrinsic_type != expected_type:
        raise AssertionError(
            f"ExtrinsicType mismatch: expected {expected_type.value}, "
            f"got {pose.extrinsic_type.value}. {msg}"
        )
