import copy
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from src.dataset.shims.bounds_shim import apply_bounds_shim
from src.dataset.shims.normalize_shim import apply_normalize_shim
from src.dataset.shims.patch_shim import apply_patch_shim
from src.dataset.types import BatchedExample, DataShim
from src.geometry.projection import sample_image_grid

from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from src.model.heads.gaussian.vggt_dpt_gs_head import VGGT_DPT_GS_Head
from src.model.vggt.utils.geometry import (
    batchify_unproject_depth_map_to_point_map,
    unproject_depth_map_to_point_map,
)
from src.model.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.utils.geometry import get_rel_pos  # used for model hub
from torch import nn, Tensor
from torch_scatter import scatter_add, scatter_max

logger = logging.getLogger(__name__)

from src.model.types import Gaussians
from src.model.heads.gaussian.gaussian_adapter import (
    GaussianAdapter,
    GaussianAdapterCfg,
    UnifiedGaussianAdapter,
)
from src.model.outputs import EncoderOutput
from .base import Encoder
from src.model.vggt.models.vggt import VGGT
from src.model.heads.instance import PartHead, SamProjector

inf = float("inf")


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class GSHeadParams:
    dec_depth: int = 23
    patch_size: tuple[int, int] = (14, 14)
    enc_embed_dim: int = 2048
    dec_embed_dim: int = 2048
    feature_dim: int = 256
    depth_mode = ("exp", -inf, inf)
    conf_mode = True


@dataclass
class EncoderAnySplatCfg:
    name: Literal["anysplat"]
    anchor_feat_dim: int
    voxel_size: float
    n_offsets: int
    d_feature: int
    add_view: bool
    num_monocular_samples: int
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    pred_pose: bool = True
    gt_pose_to_pts: bool = False
    gs_prune: bool = False
    opacity_threshold: float = 0.001
    gs_keep_ratio: float = 1.0
    pred_head_type: Literal["depth", "point"] = "point"
    freeze_backbone: bool = False
    freeze_module: Literal[
        "all",
        "global",
        "frame",
        "patch_embed",
        "patch_embed+frame",
        "patch_embed+global",
        "global+frame",
        "None",
    ] = "None"
    distill: bool = False
    render_conf: bool = False
    opacity_conf: bool = False
    conf_threshold: float = 0.1
    intermediate_layer_idx: Optional[List[int]] = None
    voxelize: bool = False
    # N: instance embedding dimension. If 0, instance head is disabled.
    instance_feat_dim: int = 0
    # Storage dtype of the aggregator's *parameters* (not the autocast compute
    # dtype, which `trainer.precision` controls independently).
    #
    # "bfloat16" is the upstream AnySplat behaviour and stays the default so
    # existing experiments are bit-identical. It is only safe when the
    # aggregator is frozen: bf16 has an 8-bit mantissa, so a parameter of
    # magnitude |w| has ULP ~= |w| * 2**-8, and an AdamW step whose size is
    # ~lr rounds to a no-op once |w| > lr * 2**8. At the fine-tuning lr this
    # repo uses for pretrained params (base_lr * backbone_lr_multiplier =
    # 2e-5), that kills every weight with |w| > ~5e-3 -- ~70% of a ViT-L
    # weight distribution -- while smaller weights keep updating, so the
    # backbone is silently and *selectively* frozen with no signal in the
    # loss curve. Lightning's "bf16-mixed" does not rescue this: it is
    # autocast only and keeps no fp32 master weights.
    #
    # Set "float32" whenever the aggregator is trainable. Costs ~7.3 GB of
    # extra static memory (params + grads + AdamW moments); pair it with
    # optimizer sharding. See .scratch/instancesplat/tickets/T11.
    aggregator_param_dtype: Literal["bfloat16", "float32"] = "bfloat16"


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderAnySplat(Encoder[EncoderAnySplatCfg]):
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderAnySplatCfg) -> None:
        super().__init__(cfg)
        model_full = VGGT.from_pretrained("facebook/VGGT-1B")
        # model_full = VGGT()
        self.aggregator = model_full.aggregator.to(
            getattr(torch, cfg.aggregator_param_dtype)
        )
        self.freeze_backbone = cfg.freeze_backbone
        self.distill = cfg.distill
        self.pred_pose = cfg.pred_pose

        self.camera_head = model_full.camera_head
        if self.cfg.pred_head_type == "depth":
            self.depth_head = model_full.depth_head
        else:
            self.point_head = model_full.point_head

        # IGGT uses point_head intermediate refinenet features (out2, out3, out4)
        # as cross-attention conditioning for the PartHead.  When using the depth
        # prediction path we still need a point_head to provide those features.
        self._use_point_head_for_part = (
            int(cfg.instance_feat_dim) > 0 and self.cfg.pred_head_type == "depth"
        )
        if self._use_point_head_for_part:
            self.point_head = model_full.point_head

        if self.distill:
            self.distill_aggregator = copy.deepcopy(self.aggregator)
            self.distill_camera_head = copy.deepcopy(self.camera_head)
            self.distill_depth_head = copy.deepcopy(self.depth_head)
            for module in [
                self.distill_aggregator,
                self.distill_camera_head,
                self.distill_depth_head,
            ]:
                for param in module.parameters():
                    param.requires_grad = False
                    param.data = param.data.cpu()

        if self.freeze_backbone:
            # Freeze backbone components
            if self.cfg.pred_head_type == "depth":
                modules_to_freeze = [self.aggregator, self.camera_head, self.depth_head]
                for module in modules_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False
            else:
                for module in [self.aggregator, self.camera_head, self.point_head]:
                    for param in module.parameters():
                        param.requires_grad = False
        else:
            # aggregator freeze
            freeze_module = self.cfg.freeze_module
            if freeze_module == "None":
                pass

            elif freeze_module == "all":
                for param in self.aggregator.parameters():
                    param.requires_grad = False

            else:
                module_pairs = {
                    "patch_embed+frame": ["patch_embed", "frame"],
                    "patch_embed+global": ["patch_embed", "global"],
                    "global+frame": ["global", "frame"],
                }

                if freeze_module in module_pairs:
                    for name, param in self.aggregator.named_parameters():
                        if any(m in name for m in module_pairs[freeze_module]):
                            param.requires_grad = False
                else:
                    for name, param in self.named_parameters():
                        param.requires_grad = (
                            freeze_module not in name and "distill" not in name
                        )

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity
        self.voxel_size = cfg.voxel_size
        self.instance_feat_dim = int(cfg.instance_feat_dim)
        # fake backbone for head parameters
        head_params = GSHeadParams()
        self.gaussian_param_head = VGGT_DPT_GS_Head(
            dim_in=2048,
            patch_size=head_params.patch_size,
            # [opacity+adapter_feats]=raw_gs_dim, +1 conf
            output_dim=self.raw_gs_dim + 1,
            activation="norm_exp",
            conf_activation="expp1",
            features=head_params.feature_dim,
        )
        # IGGT-style instance head: SamProjector (multi-scale adaptor) + PartHead
        # (DPT fusion with cross-attention from point features + window attention).
        self.part_adaptor: SamProjector | None = None
        self.part_head: PartHead | None = None
        if self.instance_feat_dim > 0:
            self.part_adaptor = SamProjector(
                dim_in=2048,
                patch_size=int(head_params.patch_size[0]),
                pos_embed=False,
                out_channels=[256, 256, 256, 256],
            )
            self.part_head = PartHead(
                in_channels=[256, 256, 256, 256],
                features=head_params.feature_dim,
                output_dim=self.instance_feat_dim,
                patch_size=int(head_params.patch_size[0]),
                window_size=8,
            )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def normalize_pts3d(self, pts3ds, valid_masks, original_extrinsics=None):
        # normalize pts_all
        B = pts3ds.shape[0]
        pts3d_norms = []
        scale_factors = []
        for bs in range(B):
            pts3d, valid_mask = pts3ds[bs], valid_masks[bs]
            if original_extrinsics is not None:
                camera_c2w = original_extrinsics[bs]
                first_camera_w2c = (
                    camera_c2w[0].inverse().unsqueeze(0).repeat(pts3d.shape[0], 1, 1)
                )

                pts3d_homo = torch.cat(
                    [pts3d, torch.ones_like(pts3d[:, :, :, :1])], dim=-1
                )
                transformed_pts3d = torch.bmm(
                    first_camera_w2c, pts3d_homo.flatten(1, 2).transpose(1, 2)
                ).transpose(1, 2)[..., :3]
                scene_scale = torch.norm(
                    transformed_pts3d.flatten(0, 1)[valid_mask.flatten(0, 2).bool()],
                    dim=-1,
                ).mean()
            else:
                transformed_pts3d = pts3d[valid_mask]
                dis = transformed_pts3d.norm(dim=-1)
                scene_scale = dis.mean().clip(min=1e-8)
            # pts3d_norm[bs] = pts3d[bs] / scene_scale
            pts3d_norms.append(pts3d / scene_scale)
            scale_factors.append(scene_scale)
        return torch.stack(pts3d_norms, dim=0), torch.stack(scale_factors, dim=0)

    def align_pts_all_with_pts3d(
        self, pts_all, pts3d, valid_mask, original_extrinsics=None
    ):
        # align pts_all with pts3d
        B = pts_all.shape[0]

        # follow vggt's normalization implementation
        pts3d_norm, scale_factor = self.normalize_pts3d(
            pts3d, valid_mask, original_extrinsics
        )  # check if this is correct
        pts_all = pts_all * scale_factor.view(B, 1, 1, 1, 1)

        return pts_all

    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def voxelizaton_with_fusion(self, img_feat, pts3d, voxel_size, conf=None, inst_feat=None):
        # img_feat: B*V, C, H, W
        # pts3d: B*V, 3, H, W
        V, C, H, W = img_feat.shape
        pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )

        # Flatten confidence scores and features
        conf_flat = conf.flatten()  # [B*V*N]
        anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]
        inst_flat = None
        if inst_feat is not None:
            # inst_feat: [V, N, H, W] -> [V*H*W, N]
            inst_flat = inst_feat.permute(0, 2, 3, 1).flatten(0, 2)

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(
            conf_exp, inverse_indices, dim=0
        )  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(
            -1
        )  # [B*V*N, 1]

        # Compute weighted average of positions and features
        weighted_pts = pts3d_flatten * weights
        weighted_feats = anchor_feats_flat.squeeze(1) * weights
        weighted_inst = None
        if inst_flat is not None:
            weighted_inst = inst_flat * weights

        # Aggregate per voxel
        voxel_pts = scatter_add(
            weighted_pts, inverse_indices, dim=0
        )  # [num_unique_voxels, 3]
        voxel_feats = scatter_add(
            weighted_feats, inverse_indices, dim=0
        )  # [num_unique_voxels, feat_dim]
        voxel_inst = None
        if weighted_inst is not None:
            voxel_inst = scatter_add(weighted_inst, inverse_indices, dim=0)  # [num_unique_voxels, N]

        return voxel_pts, voxel_feats, voxel_inst

    def forward(
        self,
        image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        device = image.device
        b, v, _, h, w = image.shape
        distill_infos = {}
        if self.distill:
            distill_image = image.clone().detach()
            for module in [
                self.distill_aggregator,
                self.distill_camera_head,
                self.distill_depth_head,
            ]:
                for param in module.parameters():
                    param.data = param.data.to(device, non_blocking=True)

            with torch.no_grad():
                # Process with bfloat16 precision
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    distill_aggregated_tokens_list, distill_patch_start_idx = (
                        self.distill_aggregator(
                            distill_image.to(torch.bfloat16),
                            intermediate_layer_idx=self.cfg.intermediate_layer_idx,
                        )
                    )

                # Process with default precision
                with torch.amp.autocast("cuda", enabled=False):
                    # Get camera pose information
                    distill_pred_pose_enc_list = self.distill_camera_head(
                        distill_aggregated_tokens_list
                    )
                    last_distill_pred_pose_enc = distill_pred_pose_enc_list[-1]
                    distill_extrinsic, distill_intrinsic = pose_encoding_to_extri_intri(
                        last_distill_pred_pose_enc, image.shape[-2:]
                    )

                    # Get depth information
                    distill_depth_map, distill_depth_conf = self.distill_depth_head(
                        distill_aggregated_tokens_list,
                        images=distill_image,
                        patch_start_idx=distill_patch_start_idx,
                    )

                    # Convert depth to 3D points
                    distill_pts_all = batchify_unproject_depth_map_to_point_map(
                        distill_depth_map, distill_extrinsic, distill_intrinsic
                    )
                # Store results
                distill_infos["pred_pose_enc_list"] = distill_pred_pose_enc_list
                distill_infos["pts_all"] = distill_pts_all
                distill_infos["depth_map"] = distill_depth_map

                conf_threshold = torch.quantile(
                    distill_depth_conf.flatten(2, 3), 0.3, dim=-1, keepdim=True
                )  # Get threshold for each view
                conf_mask = distill_depth_conf > conf_threshold.unsqueeze(-1)
                distill_infos["conf_mask"] = conf_mask

                for module in [
                    self.distill_aggregator,
                    self.distill_camera_head,
                    self.distill_depth_head,
                ]:
                    for param in module.parameters():
                        param.data = param.data.cpu()
                # Clean up to save memory
                del distill_aggregated_tokens_list, distill_patch_start_idx
                del distill_pred_pose_enc_list, last_distill_pred_pose_enc
                del distill_extrinsic, distill_intrinsic
                del distill_depth_map, distill_depth_conf
                torch.cuda.empty_cache()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = self.aggregator(
                image.to(torch.bfloat16),
                intermediate_layer_idx=self.cfg.intermediate_layer_idx,
            )

        with torch.amp.autocast("cuda", enabled=False):
            pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
            last_pred_pose_enc = pred_pose_enc_list[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                last_pred_pose_enc, image.shape[-2:]
            )  # only for debug

            # Whether we need intermediate refinenet features for instance-head
            # point-feature conditioning (aligns with IGGT which always feeds
            # point_head intermediates into the PartHead cross-attention layers).
            _need_point_feat = self.part_head is not None

            if self.cfg.pred_head_type == "point":
                if _need_point_feat:
                    pts_all, pts_conf, point_intermediate = self.point_head(
                        aggregated_tokens_list,
                        images=image,
                        patch_start_idx=patch_start_idx,
                        return_intermediate=True,
                    )
                else:
                    pts_all, pts_conf = self.point_head(
                        aggregated_tokens_list,
                        images=image,
                        patch_start_idx=patch_start_idx,
                    )
                    point_intermediate = None
            elif self.cfg.pred_head_type == "depth":
                depth_map, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=image,
                    patch_start_idx=patch_start_idx,
                )
                pts_all = batchify_unproject_depth_map_to_point_map(
                    depth_map, extrinsic, intrinsic
                )
                # Run the auxiliary point_head to obtain intermediate refinenet
                # features (out2, out3, out4) for PartHead cross-attention
                # conditioning – this mirrors IGGT's architecture.
                if _need_point_feat and self._use_point_head_for_part:
                    _pts, _pts_conf, point_intermediate = self.point_head(
                        aggregated_tokens_list,
                        images=image,
                        patch_start_idx=patch_start_idx,
                        return_intermediate=True,
                    )
                    del _pts, _pts_conf
                else:
                    point_intermediate = None
            else:
                raise ValueError(f"Invalid pred_head_type: {self.cfg.pred_head_type}")

            if self.cfg.render_conf:
                conf_valid = torch.quantile(
                    depth_conf.flatten(0, 1), self.cfg.conf_threshold
                )
                conf_valid_mask = depth_conf > conf_valid
            else:
                conf_valid_mask = torch.ones_like(depth_conf, dtype=torch.bool)

        # dpt style gs_head input format
        out = self.gaussian_param_head(
            aggregated_tokens_list,
            pts_all.flatten(0, 1).permute(0, 3, 1, 2),
            image,
            patch_start_idx=patch_start_idx,
            image_size=(h, w),
        )
        # IGGT-style instance head: SamProjector + PartHead.
        instance_feat_map = None
        if self.part_adaptor is not None and self.part_head is not None:
            adaptor_out, _pos = self.part_adaptor(
                aggregated_tokens_list,
                images=image,
                patch_start_idx=patch_start_idx,
            )
            # adaptor_out is a dict {"res1": ..., "res4": ...} with [B*V, C, H_i, W_i].
            # point_intermediate is a tuple (out2, out3, out4) from point_head's
            # DPT refinenets, used for cross-attention conditioning in PartHead.
            instance_feat_map = self.part_head(
                list(adaptor_out.values()),
                images=image,
                patch_start_idx=patch_start_idx,
                point_feature=list(point_intermediate) if point_intermediate is not None else None,
            )  # [B, V, output_dim, H, W]

        del aggregated_tokens_list, patch_start_idx
        torch.cuda.empty_cache()

        pts_flat = pts_all.flatten(2, 3)
        scene_scale = pts_flat.norm(dim=-1).mean().clip(min=1e-8)

        anchor_feats, conf = out[:, :, : self.raw_gs_dim], out[:, :, self.raw_gs_dim]

        neural_feats_list, neural_pts_list, inst_feats_list = [], [], []
        if self.cfg.voxelize:
            for b_i in range(b):
                neural_pts, neural_feats, voxel_inst = self.voxelizaton_with_fusion(
                    anchor_feats[b_i],
                    pts_all[b_i].permute(0, 3, 1, 2).contiguous(),
                    self.voxel_size,
                    conf=conf[b_i],
                    inst_feat=None if instance_feat_map is None else instance_feat_map[b_i],
                )
                neural_feats_list.append(neural_feats)
                neural_pts_list.append(neural_pts)
                if instance_feat_map is not None:
                    assert voxel_inst is not None
                    inst_feats_list.append(voxel_inst)
        else:
            for b_i in range(b):
                neural_feats_list.append(
                    anchor_feats[b_i].permute(0, 2, 3, 1)[conf_valid_mask[b_i]]
                )
                neural_pts_list.append(pts_all[b_i][conf_valid_mask[b_i]])
                if instance_feat_map is not None:
                    inst_feats_list.append(
                        instance_feat_map[b_i].permute(0, 2, 3, 1)[conf_valid_mask[b_i]]
                    )

        max_voxels = max(f.shape[0] for f in neural_feats_list)
        neural_feats = self.pad_tensor_list(
            neural_feats_list, (max_voxels,), value=-1e10
        )

        neural_pts = self.pad_tensor_list(
            neural_pts_list, (max_voxels,), -1e4
        )  # -1 == invalid voxel
        gaussian_instance_feat = None
        if instance_feat_map is not None:
            gaussian_instance_feat = self.pad_tensor_list(
                inst_feats_list, (max_voxels,), value=0.0
            )

        depths = neural_pts[..., -1].unsqueeze(-1)
        densities = neural_feats[..., 0].sigmoid()

        assert len(densities.shape) == 2, "the shape of densities should be (B, N)"
        assert neural_pts.shape[1] > 1, "the number of voxels should be greater than 1"

        opacity = self.map_pdf_to_opacity(densities, global_step).squeeze(-1)
        if self.cfg.opacity_conf:
            shift = torch.quantile(depth_conf, self.cfg.conf_threshold)
            opacity = opacity * torch.sigmoid(depth_conf - shift)[
                conf_valid_mask
            ].unsqueeze(
                0
            )  # little bit hacky

        # GS Prune, but only works when bs = 1
        # if want to support bs > 1, need to random prune gaussians based on the rank of opacity like LongLRM
        # Note: we not prune gaussians here, but we will try it in the future
        if self.cfg.gs_prune and b == 1:
            opacity_threshold = self.cfg.opacity_threshold
            gaussian_usage = opacity > opacity_threshold  # (B, N)

            logger.info(
                "based on opacity threshold %s, pruned %s gaussians out of %s",
                opacity_threshold,
                gaussian_usage.shape[1] - neural_pts.shape[1],
                gaussian_usage.shape[1],
            )

            if (gaussian_usage.sum() / gaussian_usage.numel()) > self.cfg.gs_keep_ratio:
                # rank by opacity
                num_keep = int(gaussian_usage.shape[1] * self.cfg.gs_keep_ratio)
                idx_sort = opacity.argsort(dim=1, descending=True)
                keep_idx = idx_sort[:, :num_keep]
                gaussian_usage = torch.zeros_like(gaussian_usage, dtype=torch.bool)
                gaussian_usage.scatter_(1, keep_idx, True)

            neural_pts = neural_pts[gaussian_usage].view(b, -1, 3).contiguous()
            depths = depths[gaussian_usage].view(b, -1, 1).contiguous()
            neural_feats = (
                neural_feats[gaussian_usage].view(b, -1, self.raw_gs_dim).contiguous()
            )
            opacity = opacity[gaussian_usage].view(b, -1).contiguous()
            if gaussian_instance_feat is not None:
                gaussian_instance_feat = (
                    gaussian_instance_feat[gaussian_usage]
                    .view(b, -1, self.instance_feat_dim)
                    .contiguous()
                )

            logger.info(
                "finally pruned %s gaussians out of %s",
                gaussian_usage.shape[1] - neural_pts.shape[1],
                gaussian_usage.shape[1],
            )

        gaussians = self.gaussian_adapter.forward(
            neural_pts,
            depths,
            opacity,
            neural_feats[..., 1:].squeeze(2),
        )

        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                pts_all[..., -1].flatten(2, 3).unsqueeze(-1).unsqueeze(-1),
                "b v (h w) srf s -> b v h w srf s",
                h=h,
                w=w,
            )

        infos = {}
        infos["scene_scale"] = scene_scale
        infos["voxelize_ratio"] = densities.shape[1] / (h * w * v)
        extrinsic_padding = (
            torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsic.dtype)
            .view(1, 1, 1, 4)
            .repeat(b, v, 1, 1)
        )
        intrinsic = intrinsic.clone()  # Create a new tensor
        intrinsic = torch.stack(
            [intrinsic[:, :, 0] / w, intrinsic[:, :, 1] / h, intrinsic[:, :, 2]], dim=2
        )

        return EncoderOutput(
            gaussians=gaussians,
            pred_pose_enc_list=pred_pose_enc_list,
            pred_context_pose=dict(
                extrinsic=torch.cat([extrinsic, extrinsic_padding], dim=2).inverse(),
                intrinsic=intrinsic,
            ),
            depth_dict=dict(depth=depth_map, conf_valid_mask=conf_valid_mask),
            infos=infos,
            distill_infos=distill_infos,
            instance_feat_map=instance_feat_map,
            gaussian_instance_feat=gaussian_instance_feat,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim


class AnySplat(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        encoder_cfg: EncoderAnySplatCfg,
        decoder_cfg: DecoderSplattingCUDACfg,
    ):  
        super(AnySplat, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.build_encoder(encoder_cfg)
        self.build_decoder(decoder_cfg)

    def convert_nested_config(self, cfg_dict: dict, target_class: type):
        """Convert nested dictionary config to dataclass instance
        
        Args:
            cfg_dict: Configuration dictionary or already converted object
            target_class: Target dataclass type to convert to
            
        Returns:
            Instance of target_class
        """
        if isinstance(cfg_dict, dict):
            # Convert dict to dataclass
            return target_class(**cfg_dict)
        elif isinstance(cfg_dict, target_class):
            # Already converted, return as is
            return cfg_dict
        elif cfg_dict is None:
            # Handle None case
            return None
        else:
            raise ValueError(f"Cannot convert {type(cfg_dict)} to {target_class}")

    def convert_config_recursively(self, cfg_obj, conversion_map: dict):
        """Convert nested configurations recursively using a conversion map
        
        Args:
            cfg_obj: Configuration object to convert
            conversion_map: Dict mapping field names to their target classes
                           e.g., {'gaussian_adapter': GaussianAdapterCfg}
        
        Returns:
            Converted configuration object
        """
        if not hasattr(cfg_obj, '__dict__'):
            return cfg_obj
            
        cfg_dict = cfg_obj.__dict__.copy()
        
        for field_name, target_class in conversion_map.items():
            if field_name in cfg_dict:
                cfg_dict[field_name] = self.convert_nested_config(
                    cfg_dict[field_name], 
                    target_class
                )
        
        # Return new instance of the same type
        return type(cfg_obj)(**cfg_dict)

    def convert_encoder_config(self, encoder_cfg: EncoderAnySplatCfg) -> EncoderAnySplatCfg:
        """Convert all nested configurations in encoder_cfg"""
        conversion_map = {
            'gaussian_adapter': GaussianAdapterCfg,
            'opacity_mapping': OpacityMappingCfg,
        }
        
        return self.convert_config_recursively(encoder_cfg, conversion_map)

    def build_encoder(self, encoder_cfg: EncoderAnySplatCfg):
        # Convert nested configurations using the helper method
        encoder_cfg = self.convert_encoder_config(encoder_cfg)
        self.encoder = EncoderAnySplat(encoder_cfg)

    def build_decoder(self, decoder_cfg: DecoderSplattingCUDACfg):
        self.decoder = DecoderSplattingCUDA(decoder_cfg)
    
    @torch.no_grad()
    def inference(self,
        context_image: torch.Tensor,
    ):
        self.encoder.distill = False
        encoder_output = self.encoder(context_image, global_step=0, visualization_dump=None)
        gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose
        return gaussians, pred_context_pose
    
    def forward(self, 
        context_image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        near: float = 0.01,
        far: float = 100.0,
    ):
        b, v, c, h, w = context_image.shape
        device = context_image.device
        encoder_output = self.encoder(context_image, global_step, visualization_dump=visualization_dump)
        gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose
        output = self.decoder.forward(
            gaussians,
            pred_context_pose['extrinsic'],
            pred_context_pose["intrinsic"],
            torch.ones(1, v, device=device) * near,
            torch.ones(1, v, device=device) * far,
            (h, w),
            "depth",
        )
        return encoder_output, output
    
