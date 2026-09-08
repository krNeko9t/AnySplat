#!/usr/bin/env python3
"""
Trace 2D feature maps onto pre-trained Gaussian splats (2DGS or 3DGS).

Uses ``diff_surfel_rasterization.trace()`` for **2DGS** (2 scale dims) or
``diff_gaussian_rasterization.trace()`` for **3DGS** (3 scale dims); choose
via ``--trace_backend`` or leave ``auto`` to infer from the PLY.

Both CUDA builds must define the same ``TRACE_CHANNELS`` as
``src.trace_render.trace_rasterize.TRACE_CHANNELS`` (see cuda_rasterizer/config.h).
Feature sources (--feature_source):
  anysplat    : Run AnySplat encoder with instance head (online inference).
  iggt        : Run IGGT model for part features (online inference).
  precomputed : Load pre-computed feature maps from .pt / .npy files.
  gt_idmap    : Load multi-view-consistent GT integer ID maps, encode via
                random embeddings, trace, then decode back to IDs.
  segvggt     : Run SegVGGT for a 128-d field plus its object-query bank.
  iggt_phys   : Like `iggt`, plus the physgm_dpt 32-d dense physics feature from
                the same forward pass, traced onto the same Gaussians.

Legacy args (--run_dir, --feat_dir, --model_type iggt) are still supported
and auto-detected when --feature_source is omitted.

Dependencies (anysplat conda env):
  torch, numpy, Pillow, plyfile, tqdm
  + diff_surfel_rasterization (2DGS trace) and/or diff_gaussian_rasterization (3DGS)
  + src.trace_cameras (project-local)

Example:
  # Precomputed feature maps (e.g. physics features)
  python scripts/trace_instance_to_gaussians.py \
      --source_path scene/ --ply_path scene/point_cloud.ply \
      --feature_source precomputed --feat_dir phys_feats/ --feat_dim 8

  # GT ID map trace
  python scripts/trace_instance_to_gaussians.py \
      --source_path scene/ --ply_path scene/point_cloud.ply \
      --feature_source gt_idmap --idmap_dir scene/id_maps/ \
      --postprocess gt_color,pca

  # AnySplat online inference (legacy args still work)
  python scripts/trace_instance_to_gaussians.py \
      --source_path scene/ --ply_path scene/point_cloud.ply \
      --run_dir output/exp/ --max_views 5

  # IGGT online inference
  python scripts/trace_instance_to_gaussians.py \
      --source_path scene/ --ply_path scene/point_cloud.ply \
      --model_type iggt --iggt_model_path /path/to/iggt_ckpt.pth

  # PhysicsDreamer-style transforms_train.json + IGGT (paths from configs/trace/alocasia.json):
  python scripts/trace_instance_to_gaussians.py \\
      --trace_config configs/trace/alocasia.json

  # Enable DEBUG logs for the ``coord`` logger (coordinate-system boundaries):
  python scripts/trace_instance_to_gaussians.py ... --verbose
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset.physics.types import PROPERTY_NAMES
from src.trace_cameras import TraceCamera, focal2fov, load_trace_cameras
from src.trace_render.trace_rasterize import (
    TRACE_CHANNELS,
    effective_trace_backend,
    load_gaussians_from_ply,
    resolve_trace_backend,
    trace_single_view,
    trace_single_view_chunked,
)

ENCODER_CROP_SIZE = 448  # prepare_encoder_image center-crop size
IGGT_IMAGE_SIZE = (504, 336)  # (W, H) default resize target for IGGT


def _ply_comments_with_convention(plydata: PlyData) -> list[str]:
    """Return PLY header comments, ensuring ``coordinate_convention=`` exists."""
    comments = list(plydata.comments)
    if not any(c.startswith("coordinate_convention=") for c in comments):
        comments.append("coordinate_convention=opencv")
    return comments


def compute_encoder_crop_params(orig_w, orig_h):
    """Match prepare_encoder_image: resize (shorter side 448) then center-crop 448x448.

    Returns (enc_w, enc_h, crop_left, crop_top) for the resized image before crop.
    """
    if orig_w > orig_h:
        enc_h = ENCODER_CROP_SIZE
        enc_w = int(orig_w * (enc_h / orig_h))
    else:
        enc_w = ENCODER_CROP_SIZE
        enc_h = int(orig_h * (enc_w / orig_w))
    crop_left = (enc_w - ENCODER_CROP_SIZE) // 2
    crop_top = (enc_h - ENCODER_CROP_SIZE) // 2
    return enc_w, enc_h, crop_left, crop_top


def create_virtual_crop_camera(full_cam, orig_w, orig_h, fx, fy, cx, cy):
    """Create a TraceCamera for the 448x448 center crop, aligned with encoder output.

    Uses same R,T; intrinsics adjusted for the crop (cx_crop = cx_enc - crop_left, etc).
    """
    enc_w, enc_h, crop_left, crop_top = compute_encoder_crop_params(orig_w, orig_h)

    scale_x = enc_w / orig_w
    scale_y = enc_h / orig_h
    fx_enc = fx * scale_x
    fy_enc = fy * scale_y
    cx_enc = cx * scale_x
    cy_enc = cy * scale_y

    cx_crop = cx_enc - crop_left
    cy_crop = cy_enc - crop_top

    FoVx = focal2fov(fx_enc, ENCODER_CROP_SIZE)
    FoVy = focal2fov(fy_enc, ENCODER_CROP_SIZE)

    return TraceCamera(
        R=full_cam.R, T=full_cam.T,
        FoVx=FoVx, FoVy=FoVy,
        image_name=full_cam.image_name,
        width=ENCODER_CROP_SIZE, height=ENCODER_CROP_SIZE,
        image_path=full_cam.image_path,
    )


def create_virtual_resize_camera(full_cam, orig_w, orig_h, fx, fy, cx, cy,
                                 target_w, target_h):
    """Create a TraceCamera for a resized image (e.g. IGGT's resize preprocessing).

    Scales all intrinsics proportionally to the resize ratio.
    """
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    fx_new = fx * scale_x
    fy_new = fy * scale_y

    FoVx = focal2fov(fx_new, target_w)
    FoVy = focal2fov(fy_new, target_h)

    return TraceCamera(
        R=full_cam.R, T=full_cam.T,
        FoVx=FoVx, FoVy=FoVy,
        image_name=full_cam.image_name,
        width=target_w, height=target_h,
        image_path=full_cam.image_path,
    )


# ---------------------------------------------------------------------------
# Mode A: online model inference
# ---------------------------------------------------------------------------

def load_anysplat_encoder(run_dir, ckpt_path, device="cuda"):
    """Load the AnySplat/IGGT encoder (with instance head) from a training run."""
    from src.model.arch.weight_loading import load_model_from_run

    model = load_model_from_run(run_dir, ckpt_path, device=device)
    encoder = model.encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    feat_dim = int(getattr(encoder, "instance_feat_dim", 0))
    print(f"[model] Loaded ckpt; instance_feat_dim = {feat_dim}")
    return encoder, feat_dim


def prepare_encoder_image(image_path):
    """Resize + center-crop to 448x448, return tensor in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    if width > height:
        new_height = 448
        new_width = int(width * (new_height / height))
    else:
        new_width = 448
        new_height = int(height * (new_width / width))
    img = img.resize((new_width, new_height))
    left = (new_width - 448) // 2
    top = (new_height - 448) // 2
    img = img.crop((left, top, left + 448, top + 448))

    import torchvision.transforms as T
    tensor = T.ToTensor()(img)  # [3, 448, 448] in [0, 1]
    return tensor


@torch.no_grad()
def run_encoder_batch(encoder, image_paths, device="cuda"):
    """
    Run encoder on a batch of images, return instance_feat_map.

    Returns: (feat_map, feat_dim) where feat_map is [V, D, H_feat, W_feat].
    """
    imgs = torch.stack([prepare_encoder_image(p) for p in image_paths])
    imgs = imgs.unsqueeze(0).to(device)  # [1, V, 3, 448, 448]

    enc_out = encoder(imgs, global_step=0, visualization_dump=None)
    feat_map = enc_out.instance_feat_map
    if feat_map is None:
        raise RuntimeError(
            "Encoder returned instance_feat_map=None. "
            "Check that instance_feat_dim > 0 in the checkpoint config."
        )
    # [1, V, D, H, W] -> [V, D, H, W]
    return feat_map[0].float()


# ---------------------------------------------------------------------------
# Mode A (IGGT): online model inference with IGGT
# ---------------------------------------------------------------------------

def load_and_preprocess_images_resize(image_path_list, resize_target_size):
    """Resize all input images to target (W, H), output [V, 3, H, W]."""
    import torchvision.transforms as T

    if not image_path_list:
        raise ValueError("At least 1 image is required")
    if not (isinstance(resize_target_size, (tuple, list)) and len(resize_target_size) == 2):
        raise ValueError("resize_target_size must be (width, height)")

    target_w, target_h = int(resize_target_size[0]), int(resize_target_size[1])
    to_tensor = T.ToTensor()
    images = []
    for image_path in image_path_list:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.BICUBIC)
        images.append(to_tensor(img))
    return torch.stack(images, dim=0)


def load_iggt_model(model_path, device="cuda"):
    """Load the IGGT model for part-feature extraction using the integrated module."""
    from src.model.arch.iggt import EncoderIGGTCfg
    from src.model.arch.iggt import IGGTModel

    cfg = EncoderIGGTCfg(
        name="iggt",
        instance_feat_dim=8,
        pretrained_weights="",
    )
    model = IGGTModel.from_checkpoint(cfg, model_path, device=device)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    feat_dim = 8
    print(f"[IGGT] Loaded model from {model_path}, feat_dim={feat_dim}")
    return model, feat_dim


@torch.no_grad()
def run_iggt_batch(model, image_paths, iggt_image_size, device="cuda"):
    """Run IGGT on a batch of images, return part_feat.

    Returns: feat_map [V, D, H_feat, W_feat] where D=8.
    """
    images = load_and_preprocess_images_resize(
        image_paths, resize_target_size=iggt_image_size
    ).to(device)  # [V, 3, H, W]
    images_batch = images.unsqueeze(0)  # [1, V, 3, H, W]

    enc_out, _ = model(images_batch)
    part_feat = enc_out.instance_feat_map  # [1, V, D, H, W]
    return part_feat[0].float()  # [V, D, H, W]


# ---------------------------------------------------------------------------
# Mode B: load pre-computed feature maps
# ---------------------------------------------------------------------------

def load_precomputed_features(feat_dir, image_names, feat_dim, device="cuda"):
    """
    Load per-view feature maps from feat_dir/{image_name}.pt or .npy.

    Returns list of tensors [D, H_feat, W_feat] on *device*.
    """
    feat_dir = Path(feat_dir)
    feats = []
    for name in image_names:
        pt_path = feat_dir / f"{name}.pt"
        npy_path = feat_dir / f"{name}.npy"
        if pt_path.exists():
            t = torch.load(
                pt_path, map_location=device, weights_only=True,
            ).float()
        elif npy_path.exists():
            arr = np.load(str(npy_path))
            t = torch.from_numpy(arr).float().to(device)
        else:
            raise FileNotFoundError(
                f"Feature map not found: {pt_path} or {npy_path}"
            )
        if t.dim() == 2:
            raise ValueError(f"Feature map must be 3D [D,H,W] or [H,W,D], got {t.shape}")
        if t.shape[0] != feat_dim and t.shape[-1] == feat_dim:
            t = t.permute(2, 0, 1)
        feats.append(t)
    return feats


# ---------------------------------------------------------------------------
# GT ID Map: codec (random-embedding encode/decode) + loader
# ---------------------------------------------------------------------------

class IDMapCodec:
    """Encode discrete integer IDs to random embeddings and decode back.

    Background/ignore is ID 0, mapped to zero vector.
    All other IDs get L2-normalized random vectors of dimension *embed_dim*.
    After tracing the embedding maps and averaging, the original ID is
    recovered via nearest-neighbor lookup against the embedding table.
    """

    def __init__(self, embed_dim=16, seed=42):
        self.embed_dim = embed_dim
        self.seed = seed
        self.id_to_idx = {}      # original_id -> table row index (1-based)
        self.idx_to_id = {0: 0}  # table row index -> original_id
        self.embedding_table = None  # (num_ids+1, embed_dim)

    def fit(self, id_maps):
        """Scan all ID maps to discover unique IDs and build the table."""
        all_ids = set()
        for m in id_maps:
            all_ids.update(m.unique().tolist())
        all_ids.discard(0)
        sorted_ids = sorted(all_ids)
        self.id_to_idx = {uid: i + 1 for i, uid in enumerate(sorted_ids)}
        self.idx_to_id = {0: 0}
        self.idx_to_id.update({i + 1: uid for i, uid in enumerate(sorted_ids)})

        n = len(sorted_ids) + 1  # row 0 = background
        g = torch.Generator().manual_seed(self.seed)
        table = torch.randn(n, self.embed_dim, generator=g)
        table[0] = 0.0
        table[1:] = F.normalize(table[1:].float(), p=2, dim=-1, eps=1e-8)
        self.embedding_table = table
        return self

    def encode(self, id_map):
        """(H, W) int tensor -> (embed_dim, H, W) float tensor."""
        H, W = id_map.shape
        idx_map = torch.zeros(H, W, dtype=torch.long, device=id_map.device)
        for uid, tidx in self.id_to_idx.items():
            idx_map[id_map == uid] = tidx
        emb = self.embedding_table[idx_map.view(-1)]  # (H*W, D)
        return emb.view(H, W, self.embed_dim).permute(2, 0, 1).contiguous()

    @torch.no_grad()
    def decode(self, feat, valid_mask):
        """(G, D) traced features -> (G,) integer IDs via cosine nearest neighbor."""
        table = self.embedding_table.to(feat.device).float()
        feat_n = F.normalize(feat.float(), p=2, dim=-1, eps=1e-8)
        table_n = F.normalize(table, p=2, dim=-1, eps=1e-8)
        sim = feat_n @ table_n.T  # (G, num_ids+1)
        pred_idx = sim.argmax(dim=1)

        ids = torch.zeros(feat.shape[0], dtype=torch.int64, device=feat.device)
        for tidx, uid in self.idx_to_id.items():
            ids[pred_idx == tidx] = uid
        ids[~valid_mask] = 0
        return ids

    @property
    def num_ids(self):
        return len(self.id_to_idx)


def load_gt_idmaps(idmap_dir, image_names):
    """Load per-view integer ID maps from idmap_dir/{name}.npy or image files.

    Returns list of (H, W) int64 tensors.
    """
    idmap_dir = Path(idmap_dir)
    maps = []
    for name in image_names:
        npy_path = idmap_dir / f"{name}.npy"
        png_path = idmap_dir / f"{name}.png"
        if npy_path.exists():
            arr = np.load(str(npy_path)).astype(np.int64)
        elif png_path.exists():
            arr = np.array(Image.open(png_path))
            if arr.ndim == 3:
                arr = arr[..., 0]
            arr = arr.astype(np.int64)
        else:
            candidates = list(idmap_dir.glob(f"{name}.*"))
            if candidates:
                arr = np.array(Image.open(candidates[0]))
                if arr.ndim == 3:
                    arr = arr[..., 0]
                arr = arr.astype(np.int64)
            else:
                raise FileNotFoundError(
                    f"ID map not found for '{name}' in {idmap_dir}"
                )
        maps.append(torch.from_numpy(arr))
    return maps


# ---------------------------------------------------------------------------
# Feature source preparation (one function per source type)
# ---------------------------------------------------------------------------

def prepare_anysplat_features(cam_list, args, device):
    """Load AnySplat encoder, run inference, return feat_maps + trace cameras."""
    print("\n[anysplat] Loading AnySplat encoder ...")
    encoder, feat_dim = load_anysplat_encoder(args.run_dir, args.ckpt, device)
    print(f"  Feature dim = {feat_dim}")

    batch_size = args.encoder_batch_size
    image_paths = [c.image_path for c in cam_list]
    all_feat_maps = []
    print(f"[anysplat] Running encoder on {len(cam_list)} views "
          f"(batch_size={batch_size}) ...")
    for start in range(0, len(cam_list), batch_size):
        end = min(start + batch_size, len(cam_list))
        feat_batch = run_encoder_batch(encoder, image_paths[start:end], device)
        for vi in range(feat_batch.shape[0]):
            all_feat_maps.append(feat_batch[vi])
    del encoder
    torch.cuda.empty_cache()

    if args.trace_crop_aligned:
        trace_cams = [
            create_virtual_crop_camera(
                cam, cam.orig_width, cam.orig_height,
                cam.fx_raw, cam.fy_raw, cam.cx_raw, cam.cy_raw,
            )
            for cam in cam_list
        ]
        print("  Using 448x448 virtual camera (crop-aligned)")
    else:
        trace_cams = list(cam_list)

    return {
        "feat_maps": all_feat_maps,
        "trace_cams": trace_cams,
        "feat_dim": feat_dim,
        "masks": None,
    }


def prepare_iggt_features(cam_list, args, device):
    """Load IGGT model, run inference, return feat_maps + trace cameras."""
    iggt_w, iggt_h = args._iggt_image_size
    print("\n[iggt] Loading IGGT model ...")
    iggt_model, feat_dim = load_iggt_model(args.iggt_model_path, device)
    print(f"  Feature dim = {feat_dim}")

    batch_size = args.encoder_batch_size
    image_paths = [c.image_path for c in cam_list]
    all_feat_maps = []
    print(f"[iggt] Running IGGT on {len(cam_list)} views "
          f"(batch_size={batch_size}, image_size={iggt_w}x{iggt_h}) ...")
    for start in range(0, len(cam_list), batch_size):
        end = min(start + batch_size, len(cam_list))
        feat_batch = run_iggt_batch(
            iggt_model, image_paths[start:end], args._iggt_image_size, device,
        )
        for vi in range(feat_batch.shape[0]):
            all_feat_maps.append(feat_batch[vi])
    del iggt_model
    torch.cuda.empty_cache()

    if args.trace_crop_aligned:
        trace_cams = [
            create_virtual_resize_camera(
                cam, cam.orig_width, cam.orig_height,
                cam.fx_raw, cam.fy_raw, cam.cx_raw, cam.cy_raw,
                iggt_w, iggt_h,
            )
            for cam in cam_list
        ]
        print(f"  Using {iggt_w}x{iggt_h} virtual camera (resize-aligned)")
    else:
        trace_cams = list(cam_list)

    return {
        "feat_maps": all_feat_maps,
        "trace_cams": trace_cams,
        "feat_dim": feat_dim,
        "masks": None,
    }


def prepare_precomputed_features(cam_list, args, device):
    """Load pre-computed feature maps from disk."""
    feat_dim = args.feat_dim
    print(f"\n[precomputed] Loading features from {args.feat_dir}")
    feat_maps = load_precomputed_features(
        args.feat_dir, [c.image_name for c in cam_list], feat_dim, device,
    )
    print(f"  Loaded {len(feat_maps)} feature maps, dim={feat_dim}")

    return {
        "feat_maps": feat_maps,
        "trace_cams": list(cam_list),
        "feat_dim": feat_dim,
        "masks": None,
    }


def prepare_gt_idmap_features(cam_list, args, device):
    """Load GT ID maps, encode to random embeddings, return feat_maps + masks."""
    embed_dim = args.id_embed_dim
    print(f"\n[gt_idmap] Loading ID maps from {args.idmap_dir}")
    id_maps = load_gt_idmaps(args.idmap_dir, [c.image_name for c in cam_list])
    print(f"  Loaded {len(id_maps)} ID maps")

    codec = IDMapCodec(embed_dim=embed_dim, seed=args.id_embed_seed)
    codec.fit(id_maps)
    print(f"  Unique IDs: {codec.num_ids}, embed_dim={embed_dim}")

    feat_maps = []
    masks = []
    for idmap in id_maps:
        feat_maps.append(codec.encode(idmap).to(device))
        masks.append((idmap != 0).to(torch.int32).to(device))

    return {
        "feat_maps": feat_maps,
        "trace_cams": list(cam_list),
        "feat_dim": embed_dim,
        "masks": masks,
        "id_codec": codec,
    }


# ---------------------------------------------------------------------------
# Mode E (SegVGGT): 128-d feature field + the object-query bank that reads it
# ---------------------------------------------------------------------------

# Operating point, pinned by ticket 05 of the segvggt-trace-3dgs map. Neither of
# these is a tunable: 252x448 is where ticket 01 read "the feature map does not
# drift", and 4 is the training view count (view sweep degrades monotonically:
# 4 -> 0.613, 8 -> 0.606, 12 -> 0.555, 24 -> 0.561). Do NOT inherit IGGT's 48.
SEGVGGT_TRACE_WH = (448, 252)
SEGVGGT_BATCH_SIZE = 4


def load_segvggt_model(ckpt_path, device="cuda"):
    """Load the phys-on-query SegVGGT checkpoint (arm_b_lora) for inference."""
    from src.model.arch.segvggt import EncoderSegVGGTCfg, SegVGGTModel

    # Mirrors the arm_b_lora training config (.hydra/config.yaml). pretrained_weights
    # stays empty: from_checkpoint loads the whole fine-tuned state dict, and
    # _assert_query_physgm_coverage verifies the physics head landed.
    cfg = EncoderSegVGGTCfg(
        name="segvggt",
        pretrained_weights="",
        num_semantic_classes=200,
        non_instance_classes=["wall", "floor"],
        phys_scheme="query_physgm",
        physgm_hidden=64,
    )
    model = SegVGGTModel.from_checkpoint(cfg, ckpt_path, device="cpu")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"[segvggt] Loaded {ckpt_path}")
    return model


def _assert_segvggt_preprocess_identity(cam_list):
    """Guard the one assumption the whole camera story rests on: plain full-image resize.

    ``segvggt_infer.load_and_preprocess`` silently crops in two cases -- portrait
    frames (centre square crop) and frames of unequal height (``im[:min_h]``, which
    crops from the bottom only and therefore shifts the principal point). Ticket 05
    showed ``TraceCamera`` cannot express either: it stores FoVx/FoVy only, a
    symmetric frustum with no principal point. A crop would not raise, it would just
    produce a smeared field and silently invalidate everything downstream -- so it
    raises here instead.
    """
    sizes = {(int(c.orig_width), int(c.orig_height)) for c in cam_list}
    if len(sizes) != 1:
        detail = ", ".join(
            f"{c.image_name}={int(c.orig_width)}x{int(c.orig_height)}"
            for c in cam_list[:8]
        )
        raise ValueError(
            f"segvggt feature source requires all frames to share one size; got "
            f"{len(sizes)} distinct sizes {sorted(sizes)} (first few: {detail}). "
            "Mixed sizes make load_and_preprocess crop to the minimum height, which "
            "shifts the principal point -- TraceCamera cannot represent that."
        )
    w, h = sizes.pop()
    if h > w:
        raise ValueError(
            f"segvggt feature source requires landscape frames (h <= w); got "
            f"{w}x{h} (e.g. {cam_list[0].image_name}). Portrait frames get a centre "
            "square crop, which changes the field of view TraceCamera assumes."
        )
    return w, h


def load_segvggt_images(image_paths, target_wh, device="cuda"):
    """Full-image resize to ``target_wh`` -> [1, V, 3, H, W] in [0, 1].

    Deliberately *not* segvggt_infer.load_and_preprocess: that one rounds the height
    to a multiple of 14 and may crop. Ticket 05 fixed the operating point at an exact
    252x448, and a plain full-image resize is a no-op for TraceCamera
    (``focal2fov(f * s, W * s) == focal2fov(f, W)`` per axis, so anisotropy is fine).
    """
    import torchvision.transforms as T

    target_w, target_h = int(target_wh[0]), int(target_wh[1])
    to_tensor = T.ToTensor()
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(to_tensor(img.resize((target_w, target_h), Image.Resampling.BICUBIC)))
    return torch.stack(imgs, dim=0).unsqueeze(0).to(device)


@torch.no_grad()
def run_segvggt_batch(model, image_paths, target_wh, device="cuda"):
    """One SegVGGT forward over a view batch.

    Returns ``(feat_maps [V, 128, h, w], query_bank_entry)``. The bank entry carries
    everything a query needs to be used *outside* this batch: the 128-d projected
    query (the vector the field is dotted with), the physics readout, the score and
    the ``query_idx`` join key. DETR slot ids do not survive across batches, so the
    batch index is recorded too.
    """
    from src.model.arch.segvggt_decode import decode_instances

    images = load_segvggt_images(image_paths, target_wh, device)
    enc_out, _ = model(images)
    pred = enc_out.segvggt_prediction

    feat = pred.feature_map[0].float()             # [V, h, w, 128]
    qm = pred.query_masks[0]                       # [Q, V, h, w]
    ql = pred.query_class_logits[0]                # [Q, C+1]
    Q = qm.shape[0]

    # class_agnostic is not a flag here: map fact F3 -- the joint recipe trained with
    # class_agnostic: true, so this checkpoint predicts objectness only.
    binary, scores, _labels, query_idx = decode_instances(
        ql, qm.reshape(Q, -1), class_agnostic=True,
    )

    # Area fraction of each candidate's 2D mask over this batch's views. Ticket 04:
    # one slot (234 on bench, in every batch) decodes the background as a single
    # 67%-of-frame blob, and in 3D it then claims most of the scene. It is a "stuff"
    # query, and this is where it is cheapest to recognise -- a 2D property of the
    # decode, not something a 3D threshold should have to be tuned around.
    mask_frac = binary.float().mean(dim=1)                     # [N]

    # The 128-d projection is exactly what produced the mask logits above, so
    # ``q_proj @ gau_feat.T`` on the traced field is the 3D continuation of that
    # einsum (map: "trace is alpha*T weighted accumulation, the dot product is linear").
    proj = model.encoder.model.aggregator.instance_queries_proj
    q_proj = proj(pred.query_embed[0].float())[query_idx]     # [N, 128]

    mu = pred.query_phys_mu[0].float()[query_idx]             # [N, 3] model space
    var = pred.query_phys_var[0].float()[query_idx]

    entry = {
        "query_proj": q_proj.cpu(),
        "phys_mu_model": mu.cpu(),
        "phys_var_model": var.cpu(),
        "scores": scores.cpu(),
        "query_idx": query_idx.cpu(),
        "mask_frac": mask_frac.cpu(),
        "view_names": [Path(p).stem for p in image_paths],
    }
    return feat.permute(0, 3, 1, 2).contiguous(), entry


def prepare_segvggt_features(cam_list, args, device):
    """Run SegVGGT over all views in batches; return the 128-d field + query bank."""
    from src.dataset.physics.parsers import physgm_denormalize
    from src.model.arch.segvggt_decode import format_physics_table

    w, h = _assert_segvggt_preprocess_identity(cam_list)
    target_wh = args._segvggt_image_size
    print(f"\n[segvggt] {len(cam_list)} frames, all {w}x{h} (landscape) -> "
          f"{target_wh[0]}x{target_wh[1]}")

    model = load_segvggt_model(args.segvggt_ckpt, device)
    batch_size = args.encoder_batch_size
    image_paths = [c.image_path for c in cam_list]

    all_feat_maps = []
    batches = []
    n_batches = (len(cam_list) + batch_size - 1) // batch_size
    print(f"[segvggt] Running {n_batches} batches (encoder_batch_size={batch_size}) ...")
    for start in tqdm(range(0, len(cam_list), batch_size), desc="SegVGGT"):
        end = min(start + batch_size, len(cam_list))
        feat_batch, entry = run_segvggt_batch(
            model, image_paths[start:end], target_wh, device,
        )
        for vi in range(feat_batch.shape[0]):
            all_feat_maps.append(feat_batch[vi])
        entry["batch"] = len(batches)
        entry["view_range"] = (start, end)
        batches.append(entry)

    feat_dim = all_feat_maps[0].shape[0]
    del model
    torch.cuda.empty_cache()

    # Pool the per-batch banks into Q_all. Ticket 04 owns the 3D dedup that turns
    # these overlapping hypotheses into instances; this only concatenates and keeps
    # the provenance (batch id + query_idx) so nothing is silently merged here.
    query_bank = {
        "query_proj": torch.cat([b["query_proj"] for b in batches], dim=0),
        "phys_mu_model": torch.cat([b["phys_mu_model"] for b in batches], dim=0),
        "phys_var_model": torch.cat([b["phys_var_model"] for b in batches], dim=0),
        "scores": torch.cat([b["scores"] for b in batches], dim=0),
        "query_idx": torch.cat([b["query_idx"] for b in batches], dim=0),
        "mask_frac": torch.cat([b["mask_frac"] for b in batches], dim=0),
        "batch_id": torch.cat([
            torch.full((len(b["scores"]),), b["batch"], dtype=torch.long)
            for b in batches
        ], dim=0),
        "view_names": [b["view_names"] for b in batches],
    }
    query_bank["phys_si"] = physgm_denormalize(query_bank["phys_mu_model"])
    query_bank["property_names"] = list(PROPERTY_NAMES)

    n_q = query_bank["scores"].shape[0]
    print(f"[segvggt] feat_dim={feat_dim}, Q_all={n_q} rows over {n_batches} batches "
          f"({n_q / max(n_batches, 1):.1f} per batch)")
    print("[segvggt] per-query physics (SI units, +- is the model-space std):\n"
          + format_physics_table(
              query_bank["phys_si"], query_bank["phys_var_model"],
              query_bank["scores"], query_bank["property_names"],
          ))

    # Ticket 05: full-image resize is a no-op for TraceCamera, so use the scene's own
    # cameras. Neither trace_crop_aligned helper applies (create_virtual_crop_camera
    # in particular computes cx_crop and never passes it on).
    if args.trace_crop_aligned:
        raise ValueError(
            "segvggt feature source requires --no_trace_crop_aligned: a full-image "
            "resize is already a no-op for TraceCamera, and neither virtual-camera "
            "helper models this preprocessing."
        )

    return {
        "feat_maps": all_feat_maps,
        "trace_cams": list(cam_list),
        "feat_dim": feat_dim,
        "masks": None,
        "query_bank": query_bank,
    }


# ---------------------------------------------------------------------------
# Mode F (iggt_phys): IGGT instance feat (8-d) + physgm_dpt physics feat (32-d)
# from ONE forward pass, traced onto ONE set of Gaussians.
# ---------------------------------------------------------------------------
#
# Ticket 04 of the iggt-phys-pipeline map.  This source is a strict superset of
# `iggt`: the primary stream it returns is byte-for-byte the same computation as
# prepare_iggt_features (same weights, same preprocessing, same default image
# size), and the 32-d physics map rides along as an *aux* stream so both land on
# the same Gaussians and the same cameras by construction.
#
# The 32-d map is the input of PhysGMDenseReadout, i.e. the tensor the training
# recipe pools with `pool_one_sample` before the per-property MLP
# (physgm_dense_readout.py:68-90).  Pooling it in 3D instead of 2D is exactly
# what ticket 01 decided (candidate A: "the head does not change one byte").

IGGT_PHYS_FEAT_DIM = 32
IGGT_PHYS_HIDDEN = 64


def _strip_lightning_model_prefix(sd: dict) -> dict:
    """Drop the LightningModule wrapper prefix (`model.`) from checkpoint keys."""
    return {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in sd.items()
    }


def load_iggt_phys_weights(model, ckpt_path, device="cuda"):
    """Overlay a trained `physics_scheme.*` state dict onto a built IGGT model.

    The base IGGT checkpoint has no physics head, so `from_checkpoint` leaves
    `physics_scheme` freshly initialised.  This copies the physics path (and only
    the physics path) out of a Lightning checkpoint from experiment
    `physgm_dpt_iggt` and asserts full coverage -- a silently half-loaded head
    would produce plausible-looking numbers that mean nothing.

    Returns a short provenance string.
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    step = None
    src = None
    if isinstance(raw, dict) and "state_dict" in raw:
        step = raw.get("global_step")
        src = raw.get("source_ckpt")
        raw = raw["state_dict"]
    raw = {k.replace("module.", "", 1): v for k, v in raw.items()}
    raw = _strip_lightning_model_prefix(raw)

    model_sd = model.state_dict()
    want = [k for k in model_sd if "physics_scheme" in k]
    if not want:
        raise ValueError("Model was built without a physics_scheme; nothing to load.")

    aligned = {}
    missing, mismatched = [], []
    for k in want:
        if k not in raw:
            missing.append(k)
            continue
        if tuple(raw[k].shape) != tuple(model_sd[k].shape):
            mismatched.append((k, tuple(raw[k].shape), tuple(model_sd[k].shape)))
            continue
        aligned[k] = raw[k]
    if missing or mismatched:
        raise ValueError(
            f"physics head checkpoint {ckpt_path} does not cover the built head: "
            f"{len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(mismatched)} shape-mismatched (e.g. {mismatched[:2]}). "
            "Refusing to run with a partially initialised physics head."
        )
    model.load_state_dict(aligned, strict=False)
    prov = f"{ckpt_path} ({len(aligned)} physics_scheme tensors"
    if step is not None:
        prov += f", global_step={step}"
    if src is not None:
        prov += f", from {src}"
    prov += ")"
    print(f"[iggt_phys] Loaded physics head from {prov}")
    return prov


def load_iggt_phys_model(model_path, phys_ckpt, device="cuda"):
    """Build IGGT with the `physgm_dpt` physics scheme attached.

    Returns ``(model, capture, provenance)`` where *capture* is a dict that the
    forward hook fills with the dense 32-d physics feature map.
    """
    from src.model.arch.iggt import EncoderIGGTCfg, IGGTModel

    cfg = EncoderIGGTCfg(
        name="iggt",
        instance_feat_dim=8,
        pretrained_weights="",
        phys_scheme="physgm_dpt",
        phys_feat_dim=IGGT_PHYS_FEAT_DIM,
        physgm_hidden=IGGT_PHYS_HIDDEN,
        phys_use_point_feat=True,
        phys_use_window_cross_attn=True,
    )
    model = IGGTModel.from_checkpoint(cfg, model_path, device=device)

    if phys_ckpt is not None:
        prov = load_iggt_phys_weights(model, phys_ckpt, device=device)
    else:
        prov = "RANDOMLY INITIALISED (no --iggt_phys_ckpt given)"
        print(
            "[iggt_phys] WARNING: no physics checkpoint given -- the 32-d physics "
            "stream comes from a freshly initialised head.  The plumbing is exact; "
            "the values are not meaningful."
        )

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # The dense feature map never reaches EncoderOutput: PhysGMDPTBundle keeps it
    # local and only publishes the pooled (mu, var).  A forward hook on the DPT
    # head is the non-invasive way to get it -- no model code is touched, so the
    # tensor is provably the same one the readout pools.
    capture: dict[str, torch.Tensor] = {}
    model.encoder.physics_scheme.physics_head.register_forward_hook(
        lambda _m, _i, out: capture.__setitem__("feat_map", out)
    )
    print(
        f"[iggt_phys] Loaded IGGT from {model_path}; "
        f"instance_feat_dim=8, phys_feat_dim={IGGT_PHYS_FEAT_DIM}"
    )
    return model, capture, prov


@torch.no_grad()
def run_iggt_phys_batch(model, capture, image_paths, image_size, device="cuda"):
    """One IGGT forward → (instance_feat [V,8,H,W], physics_feat [V,32,H,W])."""
    images = load_and_preprocess_images_resize(
        image_paths, resize_target_size=image_size
    ).to(device)
    images_batch = images.unsqueeze(0)  # [1, V, 3, H, W]

    # instance_mask is all-`ignore_id`: PhysGMDPTBundle short-circuits to the
    # empty-pool branch (the DPT head still runs, so the hook still fires), so we
    # pay for the dense features and nothing else.  Pooling is ours to do in 3D.
    b, v, _, h, w = images_batch.shape
    dummy_mask = torch.zeros(b, v, h, w, dtype=torch.long, device=device)

    capture.pop("feat_map", None)
    enc_out, _ = model(images_batch, instance_mask=dummy_mask)
    inst_feat = enc_out.instance_feat_map[0].float()  # [V, 8, H, W]

    phys_feat = capture.pop("feat_map", None)
    if phys_feat is None:
        raise RuntimeError(
            "physics_head forward hook never fired -- the physgm_dpt bundle did "
            "not run its DPT head this pass."
        )
    phys_feat = phys_feat[0].float()  # [V, 32, H, W]

    if phys_feat.shape[0] != inst_feat.shape[0]:
        raise RuntimeError(
            f"view count mismatch between streams: instance {inst_feat.shape} "
            f"vs physics {phys_feat.shape}"
        )
    return inst_feat, phys_feat


def prepare_iggt_phys_features(cam_list, args, device):
    """IGGT instance feat + physgm_dpt dense physics feat from one forward pass."""
    w_in, h_in = args._iggt_phys_image_size
    print("\n[iggt_phys] Loading IGGT + physgm_dpt physics head ...")
    model, capture, phys_prov = load_iggt_phys_model(
        args.iggt_model_path, args.iggt_phys_ckpt, device,
    )

    batch_size = args.encoder_batch_size
    image_paths = [c.image_path for c in cam_list]
    inst_maps, phys_maps = [], []
    print(f"[iggt_phys] Running IGGT on {len(cam_list)} views "
          f"(batch_size={batch_size}, image_size={w_in}x{h_in}) ...")
    for start in tqdm(range(0, len(cam_list), batch_size), desc="IGGT+phys"):
        end = min(start + batch_size, len(cam_list))
        inst_b, phys_b = run_iggt_phys_batch(
            model, capture, image_paths[start:end], args._iggt_phys_image_size, device,
        )
        for vi in range(inst_b.shape[0]):
            inst_maps.append(inst_b[vi].cpu())
            phys_maps.append(phys_b[vi].cpu())
    del model, capture
    torch.cuda.empty_cache()

    if len(inst_maps) != len(cam_list) or len(phys_maps) != len(cam_list):
        raise RuntimeError(
            f"stream/camera count mismatch: {len(inst_maps)} instance maps, "
            f"{len(phys_maps)} physics maps, {len(cam_list)} cameras"
        )

    # Same forward, same views: the two streams must agree on the raster grid, or
    # they are not describing the same pixels and the shared-Gaussian claim is void.
    for i, (a, b) in enumerate(zip(inst_maps, phys_maps)):
        if a.shape[1:] != b.shape[1:]:
            raise RuntimeError(
                f"view {i}: instance map {tuple(a.shape)} and physics map "
                f"{tuple(b.shape)} disagree on spatial size"
            )

    # Ticket 05 of the segvggt map: a full-image resize is a no-op for TraceCamera,
    # so `trace_crop_aligned` is only meaningful when the encoder crops.  Kept
    # identical to prepare_iggt_features so the instance stream is comparable.
    if args.trace_crop_aligned:
        trace_cams = [
            create_virtual_resize_camera(
                cam, cam.orig_width, cam.orig_height,
                cam.fx_raw, cam.fy_raw, cam.cx_raw, cam.cy_raw,
                w_in, h_in,
            )
            for cam in cam_list
        ]
        print(f"  Using {w_in}x{h_in} virtual camera (resize-aligned)")
    else:
        trace_cams = list(cam_list)

    return {
        "feat_maps": inst_maps,
        "trace_cams": trace_cams,
        "feat_dim": inst_maps[0].shape[0],
        "masks": None,
        "primary_name": "inst",
        "aux_feat_maps": {"phys": phys_maps},
        "provenance": {
            "iggt_ckpt": args.iggt_model_path,
            "phys_ckpt": phys_prov,
            "image_size_wh": (w_in, h_in),
            "encoder_batch_size": batch_size,
            "phys_feat_dim": IGGT_PHYS_FEAT_DIM,
            "physgm_hidden": IGGT_PHYS_HIDDEN,
        },
    }


FEATURE_PREP = {
    "anysplat": prepare_anysplat_features,
    "iggt": prepare_iggt_features,
    "precomputed": prepare_precomputed_features,
    "gt_idmap": prepare_gt_idmap_features,
    "segvggt": prepare_segvggt_features,
    "iggt_phys": prepare_iggt_phys_features,
}


# ---------------------------------------------------------------------------
# Post-processing: PCA visualization, clustering, PLY export
# ---------------------------------------------------------------------------

def load_xyz_from_ply(ply_path):
    """Load only xyz positions from a PLY file (lightweight for postprocess)."""
    plydata = PlyData.read(ply_path)
    v = plydata.elements[0]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)
    return xyz.astype(np.float32)


@torch.no_grad()
def knn_smooth_gaussians(xyz_np, feat, k=20):
    """Smooth [G, D] Gaussian features via 3D KNN averaging (cKDTree).

    Aligns with iggt_idmap.knn_smooth_features / instance_viz._knn_smooth_features_impl.
    """
    from scipy.spatial import cKDTree

    G, D = feat.shape
    feat_np = feat.cpu().float().numpy()

    print(f"[knn] Building cKDTree for {G:,} points (k={k}) ...")
    tree = cKDTree(xyz_np)
    _, indices = tree.query(xyz_np, k=k + 1)
    indices = indices[:, 1:]

    print("[knn] Averaging neighbour features ...")
    smoothed = feat_np[indices].mean(axis=1)
    return torch.from_numpy(smoothed).to(device=feat.device, dtype=feat.dtype)


@torch.no_grad()
def pca_colorize_gaussians(feat, valid_mask, low_p=0.02, high_p=0.98):
    """PCA-reduce [G, D] instance features to [G, 3] uint8 RGB.

    Aligns with iggt_idmap.pca_visualize: L2 normalize, center only (no std),
    stride-subsampled quantile for large tensors.

    Only valid Gaussians (valid_mask == True) are used to fit PCA;
    invalid ones get black (0, 0, 0).
    """
    G, D = feat.shape
    feat_f = F.normalize(feat.float(), p=2, dim=-1, eps=1e-8)

    valid_feat = feat_f[valid_mask]
    if valid_feat.shape[0] < 10:
        print("[pca] Warning: fewer than 10 valid Gaussians, using all")
        valid_feat = feat_f

    if D < 3:
        pad = torch.zeros(feat_f.shape[0], 3 - D, device=feat_f.device)
        feat_f = torch.cat([feat_f, pad], dim=1)
        pad_v = torch.zeros(valid_feat.shape[0], 3 - D, device=valid_feat.device)
        valid_feat = torch.cat([valid_feat, pad_v], dim=1)

    mean = valid_feat.mean(dim=0, keepdim=True)
    feat_centered = feat_f - mean
    valid_centered = valid_feat - mean

    try:
        _, _, v = torch.pca_lowrank(valid_centered, q=min(valid_centered.shape[1], 256))
        proj = feat_centered @ v[:, :3]  # [G, 3]
    except Exception as e:
        print(f"[pca] PCA failed ({e}), returning black")
        return np.zeros((G, 3), dtype=np.uint8)

    proj_valid = proj[valid_mask]
    max_quantile_elems = 1_000_000
    for i in range(3):
        ch = proj_valid[:, i].flatten()
        n = ch.numel()
        if n > max_quantile_elems:
            step = max(1, n // max_quantile_elems)
            ch_sub = ch[::step]
        else:
            ch_sub = ch
        v_lo = torch.quantile(ch_sub, low_p)
        v_hi = torch.quantile(ch_sub, high_p)
        if v_hi > v_lo:
            proj[:, i] = (proj[:, i] - v_lo) / (v_hi - v_lo)
        else:
            proj[:, i] = 0.5
    proj = proj.clamp(0, 1)

    rgb_f = proj.cpu().numpy()
    rgb_u8 = (rgb_f * 255).astype(np.uint8)
    rgb_u8[~valid_mask.cpu().numpy()] = 0
    return rgb_u8


@torch.no_grad()
def cluster_gaussians(feat, valid_mask, algo, **kwargs):
    """Cluster per-Gaussian features and return (labels [G] int32, centers).

    Labels: 0 = invalid/unassigned, 1..K = cluster IDs.
    Features are L2-normalized before clustering.
    """
    G, D = feat.shape
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return np.zeros(G, dtype=np.int32), np.zeros((0, D), dtype=np.float32)

    feat_n = F.normalize(feat.float(), p=2, dim=-1, eps=1e-8)

    if algo == "kmeans":
        return _cluster_kmeans(feat_n, valid_idx, G, D, **kwargs)
    elif algo == "dbscan":
        return _cluster_dbscan(feat_n, valid_idx, G, D, **kwargs)
    elif algo == "hdbscan":
        return _cluster_hdbscan(feat_n, valid_idx, G, D, **kwargs)
    else:
        raise ValueError(f"Unknown clustering algo: {algo}")


def _cluster_kmeans(feat_n, valid_idx, G, D, k=20, iters=30, seed=0, max_points=200000):
    from src.instseg.kmeans import kmeans_torch

    S = min(max_points, valid_idx.numel())
    g = torch.Generator(device=feat_n.device)
    g.manual_seed(seed)
    perm = torch.randperm(valid_idx.numel(), generator=g, device=feat_n.device)[:S]
    x = feat_n[valid_idx[perm]]
    _, centers = kmeans_torch(x, k=k, num_iters=iters, seed=seed)

    centers_n = F.normalize(centers.float(), p=2, dim=-1, eps=1e-8)
    pred = (feat_n @ centers_n.T).argmax(dim=1).to(torch.int64)

    valid_bool = torch.zeros(G, dtype=torch.bool, device=feat_n.device)
    valid_bool[valid_idx] = True
    pred[~valid_bool] = -1

    labels = torch.zeros(G, dtype=torch.int32, device=feat_n.device)
    labels[pred >= 0] = (pred[pred >= 0] + 1).to(torch.int32)
    return labels.cpu().numpy(), centers.cpu().numpy()


def _cluster_dbscan(feat_n, valid_idx, G, D, eps=0.3, min_samples=10, max_points=200000):
    from sklearn.cluster import DBSCAN

    x_np = feat_n[valid_idx].cpu().numpy()
    if x_np.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx_sub = rng.choice(x_np.shape[0], max_points, replace=False)
        sub_labels = DBSCAN(
            eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1,
        ).fit_predict(x_np[idx_sub])
        mask_clustered = sub_labels >= 0
        n_sub_clusters = int(sub_labels.max() + 1) if mask_clustered.any() else 0
        if n_sub_clusters >= 2:
            from sklearn.neighbors import NearestCentroid
            nc = NearestCentroid()
            nc.fit(x_np[idx_sub][mask_clustered], sub_labels[mask_clustered])
            db_labels = nc.predict(x_np)
        elif n_sub_clusters == 1:
            db_labels = np.zeros(x_np.shape[0], dtype=np.int32)
        else:
            db_labels = np.full(x_np.shape[0], -1, dtype=np.int32)
    else:
        db_labels = DBSCAN(
            eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1,
        ).fit_predict(x_np)

    labels = np.zeros(G, dtype=np.int32)
    valid_np = valid_idx.cpu().numpy()
    is_cluster = db_labels >= 0
    if np.any(is_cluster):
        labels[valid_np[is_cluster]] = (db_labels[is_cluster] + 1).astype(np.int32)

    n_clusters = int(db_labels[is_cluster].max() + 1) if np.any(is_cluster) else 0
    centers = np.zeros((n_clusters, feat_n.shape[1]), dtype=np.float32)
    for cid in range(n_clusters):
        mem = valid_np[db_labels == cid]
        c = feat_n[torch.from_numpy(mem).to(feat_n.device)].float().mean(0)
        centers[cid] = F.normalize(c, p=2, dim=-1, eps=1e-8).cpu().numpy()
    return labels, centers


def _cluster_hdbscan(feat_n, valid_idx, G, D,
                     min_cluster_size=50, min_samples=10, max_points=200000,
                     cluster_selection_epsilon=0.06):
    from src.instseg.hdbscan_assign import hdbscan_assign

    x_np = feat_n[valid_idx].cpu().numpy()
    valid_labels = hdbscan_assign(
        x_np,
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        max_points=max_points,
        rng_seed=0,
    )

    labels = np.zeros(G, dtype=np.int32)
    valid_np = valid_idx.cpu().numpy()
    labels[valid_np] = valid_labels + 1

    n_clusters = int(valid_labels.max()) + 1
    centers = np.zeros((n_clusters, feat_n.shape[1]), dtype=np.float32)
    for cid in range(n_clusters):
        mem = valid_np[valid_labels == cid]
        c = feat_n[torch.from_numpy(mem).to(feat_n.device)].float().mean(0)
        centers[cid] = F.normalize(c, p=2, dim=-1, eps=1e-8).cpu().numpy()
    return labels, centers


def save_colored_pointcloud(xyz_np, rgb_u8, path):
    """Write a simple (x,y,z,red,green,blue) PLY file."""
    from plyfile import PlyData as PD, PlyElement as PE

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    verts = np.empty(len(xyz_np), dtype=dtype)
    verts["x"], verts["y"], verts["z"] = xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    PD(
        [PE.describe(verts, "vertex")],
        comments=["coordinate_convention=opencv"],
    ).write(str(path))
    print(f"  Saved PLY: {path}")


def save_colored_gaussians_ply(ply_path, rgb_u8, out_path):
    """Clone the original 3DGS PLY but replace SH DC with given RGB colors.

    All other attributes (scales, rotations, opacities, f_rest, etc.) are
    preserved verbatim, so the output can be rendered in any 3DGS viewer.
    Header comments (including ``coordinate_convention=``) are preserved from
    the source file when present.
    """
    plydata = PlyData.read(ply_path)
    v = plydata.elements[0]
    comments = _ply_comments_with_convention(plydata)

    C0 = 0.28209479177387814
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    sh_dc = (rgb_f - 0.5) / C0  # [G, 3]

    data = v.data.copy()
    data["f_dc_0"] = sh_dc[:, 0].astype(np.float32)
    data["f_dc_1"] = sh_dc[:, 1].astype(np.float32)
    data["f_dc_2"] = sh_dc[:, 2].astype(np.float32)

    from plyfile import PlyData as PD, PlyElement as PE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    PD([PE.describe(data, "vertex")], comments=comments).write(str(out_path))
    print(f"  Saved 3DGS PLY: {out_path}")


def write_ply_vertex_rows(
    vdata,
    row_mask: np.ndarray,
    out_path: str,
    *,
    comments: list[str] | None = None,
) -> bool:
    """Write a subset of 3DGS / PLY vertices (structured array) to a new PLY.

    Preserves all vertex properties from the source file. Returns False if
    the subset is empty (no file written).
    """
    from plyfile import PlyData as PD, PlyElement as PE

    sub = vdata[row_mask]
    if sub.shape[0] == 0:
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if comments is not None:
        PD([PE.describe(sub, "vertex")], comments=comments).write(str(out_path))
    else:
        PD([PE.describe(sub, "vertex")]).write(str(out_path))
    return True


def _postprocess_algo_tag(args, algo: str) -> str:
    """Match scripts/instseg_infer._algo_tag for seg3d_split directory names."""
    if algo == "kmeans":
        return f"kmeans_k{args.k}"
    if algo == "dbscan":
        return f"dbscan_eps{args.dbscan_eps}_ms{args.dbscan_min_samples}"
    if algo == "hdbscan":
        return (
            f"hdbscan_mcs{args.hdbscan_min_cluster_size}"
            f"_ms{args.hdbscan_min_samples}"
        )
    return algo


def _run_gt_instance_split(
    ply_path: str,
    gaussian_ids,
    post_dir: str,
    ignore_id: int = 0,
) -> None:
    """Export one PLY per GT instance id.

    By convention, ID 0 is background/ignore and is not exported.
    """
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    vdata = plydata.elements[0].data
    comments = _ply_comments_with_convention(plydata)
    G = gaussian_ids.shape[0]
    if len(vdata) != G:
        raise AssertionError(
            f"PLY has {len(vdata)} vertices but gaussian_ids length is {G}"
        )

    ids_np = gaussian_ids.cpu().numpy().astype(np.int32)
    split_dir = os.path.join(post_dir, "gt_split")
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, "gt_ids.npy"), ids_np)

    n_saved = 0
    for iid in np.unique(ids_np):
        if int(iid) == int(ignore_id):
            continue
        mask = ids_np == iid
        path = os.path.join(split_dir, f"instance_{int(iid):05d}.ply")
        if write_ply_vertex_rows(vdata, mask, path, comments=comments):
            n_saved += 1

    print(
        f"[gt_split] Saved {n_saved} instance PLYs "
        f"(ignore_id={ignore_id}, ids 0..{int(ids_np.max())}) -> {split_dir}"
    )


def run_postprocess(feat, num_ray, ply_path, output_dir, args,
                    gaussian_ids=None):
    """Dispatch post-processing: PCA, clustering, and/or GT-ID coloring."""
    from src.visualization.instance_viz import make_color_lut

    post_str = getattr(args, "postprocess", None) or ""
    modes = [s.strip() for s in post_str.split(",") if s.strip()]
    want_seg3d_split = getattr(args, "postprocess_seg3d_split", False)
    want_gt_split = getattr(args, "postprocess_gt_split", False)

    if not modes and not want_gt_split and not want_seg3d_split:
        return

    valid_mask = num_ray > 0
    n_valid = valid_mask.sum().item()
    G = feat.shape[0]
    print(f"\n[postprocess] {G:,} Gaussians, {n_valid:,} valid ({100*n_valid/G:.1f}%)")
    if modes:
        print(f"[postprocess] Modes: {modes}")
    else:
        print("[postprocess] Modes: (none; gt_split only)")

    xyz_np = load_xyz_from_ply(ply_path)
    assert xyz_np.shape[0] == G, (
        f"PLY has {xyz_np.shape[0]} points but feat has {G} Gaussians"
    )

    post_dir = os.path.join(output_dir, "postprocess")

    if not modes:
        if want_gt_split:
            if gaussian_ids is None:
                print(
                    "\n[gt_split] Skipped: no gaussian_ids. "
                    "(Use --feature_source gt_idmap, or --load_traced with a .pt "
                    "that contains gaussian_ids.)"
                )
            else:
                print("\n--- GT instance PLY split ---")
                _run_gt_instance_split(ply_path, gaussian_ids, post_dir)
        if want_seg3d_split:
            print(
                "\n[seg3d_split] Skipped: add kmeans, dbscan, or hdbscan to --postprocess"
            )
        print(f"\n[postprocess] Done -> {post_dir}")
        return

    feat_gpu = feat.cuda() if not feat.is_cuda else feat
    valid_gpu = valid_mask.cuda() if not valid_mask.is_cuda else valid_mask

    knn_k = getattr(args, "postprocess_knn_k", 0)
    if knn_k > 0:
        print(f"\n--- 3D KNN smoothing (k={knn_k}) ---")
        feat_gpu = knn_smooth_gaussians(xyz_np, feat_gpu, k=knn_k)

    if "gt_color" in modes:
        if gaussian_ids is None:
            print("\n[gt_color] Skipped: no decoded IDs (only for gt_idmap source)")
        else:
            print("\n--- GT ID coloring ---")
            ids_np = gaussian_ids.cpu().numpy().astype(np.int32)
            max_id = int(ids_np.max())
            n_assigned = int((ids_np > 0).sum())
            print(f"  Max ID: {max_id}, assigned: {n_assigned:,}/{n_valid:,}")

            lut = make_color_lut(max_id + 1, seed=args.palette_seed)
            rgb_gt = lut[np.clip(ids_np, 0, max_id)]
            rgb_gt[ids_np == 0] = 0

            gt_dir = os.path.join(post_dir, "gt_color")
            save_colored_pointcloud(xyz_np, rgb_gt,
                                    os.path.join(gt_dir, "colored_pointcloud.ply"))
            save_colored_gaussians_ply(ply_path, rgb_gt,
                                       os.path.join(gt_dir, "colored_gaussians.ply"))
            np.save(os.path.join(gt_dir, "gt_ids.npy"), ids_np)
            print(f"  Saved IDs: {os.path.join(gt_dir, 'gt_ids.npy')}")

    if "pca" in modes:
        print("\n--- PCA visualization ---")
        rgb_pca = pca_colorize_gaussians(feat_gpu, valid_gpu)
        save_colored_pointcloud(xyz_np, rgb_pca,
                                os.path.join(post_dir, "pca_pointcloud.ply"))
        save_colored_gaussians_ply(ply_path, rgb_pca,
                                   os.path.join(post_dir, "pca_gaussians.ply"))

    for algo in ["kmeans", "dbscan", "hdbscan"]:
        if algo not in modes:
            continue
        print(f"\n--- Clustering: {algo} ---")
        if algo == "kmeans":
            kw = dict(k=args.k, iters=args.kmeans_iters,
                      seed=args.seed, max_points=args.max_cluster_pts)
        elif algo == "dbscan":
            kw = dict(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples,
                      max_points=args.max_cluster_pts)
        else:
            kw = dict(min_cluster_size=args.hdbscan_min_cluster_size,
                      min_samples=args.hdbscan_min_samples,
                      max_points=args.max_cluster_pts,
                      cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon)

        labels, centers = cluster_gaussians(feat_gpu, valid_gpu, algo, **kw)
        max_label = int(labels.max()) if labels.size > 0 else 0
        n_clustered = int((labels > 0).sum())
        print(f"  Clusters: {max_label}, assigned: {n_clustered:,}/{n_valid:,}")

        lut = make_color_lut(max_label + 1, seed=args.palette_seed)
        rgb_cluster = lut[np.clip(labels, 0, max_label)]  # [G, 3] uint8
        rgb_cluster[labels == 0] = 0

        algo_dir = os.path.join(post_dir, algo)
        save_colored_pointcloud(xyz_np, rgb_cluster,
                                os.path.join(algo_dir, "colored_pointcloud.ply"))
        save_colored_gaussians_ply(ply_path, rgb_cluster,
                                   os.path.join(algo_dir, "colored_gaussians.ply"))
        np.save(os.path.join(algo_dir, "cluster_labels.npy"), labels)
        print(f"  Saved labels: {os.path.join(algo_dir, 'cluster_labels.npy')}")

        if want_seg3d_split:
            from plyfile import PlyData

            tag = _postprocess_algo_tag(args, algo)
            split_dir = os.path.join(post_dir, "seg3d_split", tag)
            os.makedirs(split_dir, exist_ok=True)
            np.save(os.path.join(split_dir, "cluster_labels.npy"), labels)
            plydata = PlyData.read(ply_path)
            vdata = plydata.elements[0].data
            comments = _ply_comments_with_convention(plydata)
            n_saved = 0
            for cid in range(1, max_label + 1):
                mask = labels == cid
                if not mask.any():
                    continue
                out_p = os.path.join(split_dir, f"cluster_{cid - 1:03d}.ply")
                if write_ply_vertex_rows(vdata, mask, out_p, comments=comments):
                    n_saved += 1
            m0 = labels == 0
            if m0.any():
                write_ply_vertex_rows(
                    vdata,
                    m0,
                    os.path.join(split_dir, "unclustered.ply"),
                    comments=comments,
                )
            print(
                f"  [seg3d_split] algo={tag} saved {n_saved} cluster PLYs "
                f"(max_label={max_label})"
                f"{', unclustered.ply' if m0.any() else ''} -> {split_dir}"
            )

    if want_seg3d_split and not any(
        a in modes for a in ("kmeans", "dbscan", "hdbscan")
    ):
        print(
            "\n[seg3d_split] Skipped: add kmeans, dbscan, or hdbscan to --postprocess"
        )

    if want_gt_split:
        if gaussian_ids is None:
            print(
                "\n[gt_split] Skipped: no gaussian_ids. "
                "(Use --feature_source gt_idmap, or --load_traced with a .pt "
                "that contains gaussian_ids.)"
            )
        else:
            print("\n--- GT instance PLY split ---")
            _run_gt_instance_split(ply_path, gaussian_ids, post_dir)

    print(f"\n[postprocess] Done -> {post_dir}")


def render_colored_views(source_path, ply_path, output_dir, args):
    """Render original + PCA + cluster PLYs from random views and save images."""
    import random

    n_views = args.render_views
    if n_views <= 0:
        return

    post_dir = os.path.join(output_dir, "postprocess")
    render_dir = os.path.join(post_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    plys = {"original": ply_path}
    if os.path.isfile(os.path.join(post_dir, "pca_gaussians.ply")):
        plys["pca"] = os.path.join(post_dir, "pca_gaussians.ply")
    gt_color_path = os.path.join(post_dir, "gt_color", "colored_gaussians.ply")
    if os.path.isfile(gt_color_path):
        plys["gt_color"] = gt_color_path
    for algo in ["kmeans", "dbscan", "hdbscan"]:
        path = os.path.join(post_dir, algo, "colored_gaussians.ply")
        if os.path.isfile(path):
            plys[algo] = path

    if len(plys) <= 1:
        print("[render] No PCA/cluster PLYs found, skip rendering")
        return

    print(f"\n[render] Loading cameras from {source_path} ...")
    cameras = load_trace_cameras(
        args.camera_backend,
        source_path=os.path.abspath(source_path),
        images_folder=args.images_folder,
        resolution=args.render_resolution,
        transforms_json=args.transforms_json,
        transforms_axis=args.transforms_axis,
    )
    n_avail = min(n_views, len(cameras))
    random.seed(args.seed)
    selected = random.sample(cameras, n_avail)
    print(f"  Selected {n_avail} views: {[c.image_name for c in selected]}")

    bg = torch.tensor([1.0, 1.0, 1.0], device="cuda")

    for ply_name, ply_path_i in plys.items():
        print(f"  Rendering {ply_name} ...")
        means, quats, scales, opacities, colors = load_gaussians_from_ply(ply_path_i)

        for cam in selected:
            H, W = cam.image_height, cam.image_width
            img_sem = torch.zeros(H, W, TRACE_CHANNELS, device="cuda")
            img_mask = torch.ones(H, W, dtype=torch.int32, device="cuda")

            rb = getattr(args, "resolved_trace_backend", None)
            if rb is None:
                rb = resolve_trace_backend("auto", scales.shape[1])

            with torch.no_grad():
                _, _, _, out_color = trace_single_view(
                    means, quats, scales, opacities, colors,
                    img_sem, img_mask, cam, bg,
                    rb,
                )

            img_np = (
                out_color.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)
            Image.fromarray(img_np).save(
                os.path.join(render_dir, f"{cam.image_name}_{ply_name}.png")
            )

    # Comparison grids (all PLYs side by side)
    ply_names = list(plys.keys())
    print(f"  Creating comparison grids ({' | '.join(ply_names)}) ...")
    for cam in selected:
        imgs = []
        for ply_name in ply_names:
            path = os.path.join(render_dir, f"{cam.image_name}_{ply_name}.png")
            imgs.append(np.array(Image.open(path)))
        grid = np.concatenate(imgs, axis=1)
        Image.fromarray(grid).save(
            os.path.join(render_dir, f"{cam.image_name}_compare.png")
        )

    print(f"  Saved to {render_dir}")


# ---------------------------------------------------------------------------
# Feature source resolution (auto-detect from legacy args when needed)
# ---------------------------------------------------------------------------

def _resolve_feature_source(args, parser):
    """Determine feature_source from explicit flag or legacy args."""
    if args.feature_source is not None:
        return args.feature_source
    if args.idmap_dir is not None:
        return "gt_idmap"
    if args.model_type == "iggt":
        return "iggt"
    if args.run_dir is not None:
        return "anysplat"
    if args.feat_dir is not None:
        return "precomputed"
    parser.error(
        "Cannot determine feature source. Use --feature_source or provide "
        "--run_dir (anysplat), --model_type iggt (iggt), --feat_dir "
        "(precomputed), or --idmap_dir (gt_idmap)."
    )


def _validate_source_args(feature_source, args, parser):
    """Validate that required args are present for the chosen source."""
    if feature_source == "anysplat":
        if args.run_dir is None:
            parser.error("--run_dir is required for feature_source=anysplat")
        if args.ckpt is None:
            args.ckpt = str(Path(args.run_dir) / "checkpoints" / "last.ckpt")
            if not os.path.exists(args.ckpt):
                parser.error(
                    f"--ckpt not specified and default {args.ckpt} not found"
                )
    elif feature_source == "iggt":
        if args.iggt_model_path is None:
            parser.error(
                "--iggt_model_path is required for feature_source=iggt"
            )
    elif feature_source == "iggt_phys":
        if args.iggt_model_path is None:
            parser.error(
                "--iggt_model_path is required for feature_source=iggt_phys"
            )
    elif feature_source == "precomputed":
        if args.feat_dir is None:
            parser.error(
                "--feat_dir is required for feature_source=precomputed"
            )
    elif feature_source == "gt_idmap":
        if args.idmap_dir is None:
            parser.error(
                "--idmap_dir is required for feature_source=gt_idmap"
            )


def _strip_trace_config_from_argv(argv):
    """Remove ``--trace_config PATH`` pairs and return JSON defaults dict."""
    cfg = {}
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--trace_config" and i + 1 < len(argv):
            with open(argv[i + 1], encoding="utf-8") as f:
                cfg = json.load(f)
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out, cfg


def _filter_trace_config_defaults(parser: argparse.ArgumentParser, cfg: dict) -> dict:
    """Only pass keys that match argparse destinations."""
    dests = {
        a.dest
        for a in parser._actions
        if getattr(a, "dest", None) not in (None, argparse.SUPPRESS)
    }
    return {k: v for k, v in cfg.items() if k in dests}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace 2D features onto 2DGS (surfel) or 3DGS Gaussians"
    )
    g_scene = parser.add_argument_group("Scene")
    g_scene.add_argument("--source_path", "-s", default=None)
    g_scene.add_argument("--ply_path", "-p", default=None)
    g_scene.add_argument("--images_folder", default="images")
    g_scene.add_argument("--resolution", "-r", type=int, default=1)
    g_scene.add_argument("--output_dir", "-o", default="trace_output")
    g_scene.add_argument(
        "--camera_backend",
        default="colmap",
        choices=["colmap", "nerf_transforms"],
        help="Camera source: COLMAP sparse/ or NeRF-style transforms JSON",
    )
    g_scene.add_argument(
        "--transforms_json",
        default=None,
        help="Path to transforms_train.json (nerf_transforms; "
             "default: transforms_train.json then transforms.json under source_path)",
    )
    g_scene.add_argument(
        "--transforms_axis",
        default="nerfstudio",
        choices=["nerfstudio", "opencv"],
        help="Meaning of transform_matrix rows/cols for nerf_transforms "
             "(see src.trace_cameras.load_transforms_json)",
    )

    g_feat = parser.add_argument_group("Feature source")
    g_feat.add_argument(
        "--feature_source", default=None,
        choices=["anysplat", "iggt", "precomputed", "gt_idmap", "segvggt",
                 "iggt_phys"],
        help="Feature source type (auto-detected from legacy args if omitted)",
    )
    g_feat.add_argument("--feat_dir", default=None,
                        help="precomputed: directory of .pt/.npy feature maps")
    g_feat.add_argument("--feat_dim", type=int, default=8,
                        help="Feature dimension (precomputed; others read from model)")
    g_feat.add_argument("--run_dir", default=None,
                        help="anysplat: Hydra run directory")
    g_feat.add_argument("--ckpt", default=None,
                        help="anysplat: Lightning checkpoint path "
                             "(default: run_dir/checkpoints/last.ckpt)")
    g_feat.add_argument("--encoder_batch_size", type=int, default=4,
                        help="anysplat/iggt/segvggt: views per forward pass "
                             "(segvggt: leave at 4 -- the training view count; "
                             "quality degrades monotonically above it)")
    g_feat.add_argument("--segvggt_ckpt", default=None,
                        help="segvggt: Lightning .ckpt of the phys-on-query run "
                             "(arm_b_lora)")
    g_feat.add_argument("--segvggt_image_size", default="448,252",
                        help="segvggt: W,H fed to the encoder (default 448,252 = the "
                             "training operating point; not a tunable)")
    g_feat.add_argument("--model_type", choices=["anysplat", "iggt"],
                        default="anysplat",
                        help="Legacy model selector (prefer --feature_source)")
    g_feat.add_argument("--iggt_model_path", default=None,
                        help="iggt: path to IGGT checkpoint")
    g_feat.add_argument("--iggt_image_size", default="504,336",
                        help="iggt: resize target as W,H (default: 504,336)")
    g_feat.add_argument("--iggt_phys_ckpt", default=None,
                        help="iggt_phys: Lightning .ckpt of the physgm_dpt_iggt run "
                             "(supplies physics_scheme.*; omit to run a freshly "
                             "initialised head -- plumbing only, values meaningless)")
    g_feat.add_argument("--iggt_phys_image_size", default=None,
                        help="iggt_phys: resize target as W,H "
                             "(default: follow --iggt_image_size, so the instance "
                             "stream matches feature_source=iggt exactly)")
    g_feat.add_argument("--idmap_dir", default=None,
                        help="gt_idmap: directory of integer ID map files "
                             "(.npy or single-channel images)")
    g_feat.add_argument("--id_embed_dim", type=int, default=16,
                        help="gt_idmap: random embedding dimension (default: 16)")
    g_feat.add_argument("--id_embed_seed", type=int, default=42,
                        help="gt_idmap: random seed for embedding table")

    g_run = parser.add_argument_group("Runtime")
    g_run.add_argument("--max_views", type=int, default=None)
    g_run.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging for the coord logger (coordinate-system boundaries)",
    )
    g_run.add_argument("--white_background", action="store_true")
    g_run.add_argument("--save_render", action="store_true",
                       help="Save per-view trace RGB renders")
    g_run.add_argument("--trace_blend_mass", action="store_true", default=False,
                       help="Trace one extra constant-1.0 channel per view and save "
                            "it as `blend_mass [N]`.  The kernel accumulates "
                            "gau_sem[g] = sum_r w_gr * img_sem[r] with per-ray blend "
                            "weights, while num_ray[g] is an unweighted hit count, so "
                            "feat = gau_sem/num_ray carries a per-Gaussian scale of "
                            "blend_mass/num_ray (bench: median 0.012).  Saving it "
                            "makes that scale removable later without re-tracing.")
    g_run.add_argument("--dump_feat_maps", default=None,
                       help="Directory to write the per-view 2D feature maps "
                            "(fp16, <dir>/<stream>/<image_name>.pt) that were fed "
                            "to trace.  Needed to compare a 2D pooling protocol "
                            "against the 3D one without a second forward pass.")
    g_run.add_argument("--trace_crop_aligned", action="store_true", default=True,
                       help="anysplat/iggt: trace with virtual camera aligned "
                            "to encoder preprocessing (default: True)")
    g_run.add_argument("--no_trace_crop_aligned", dest="trace_crop_aligned",
                       action="store_false",
                       help="Disable trace_crop_aligned (use full camera res)")
    g_run.add_argument(
        "--trace_backend",
        default="auto",
        choices=["auto", "surfel", "3dgs"],
        help="Trace rasterizer: surfel=diff_surfel_rasterization (2 scale_*), "
             "3dgs=diff_gaussian_rasterization (3 scale_*), auto=infer from PLY",
    )

    g_post = parser.add_argument_group("Post-processing")
    g_post.add_argument("--postprocess", default=None,
                        help="Comma-separated: pca, gt_color, kmeans, "
                             "dbscan, hdbscan")
    g_post.add_argument("--load_traced", default=None,
                        help="Load saved .pt file, skip trace, "
                             "run postprocess only")
    g_post.add_argument("--k", type=int, default=20,
                        help="K-means clusters (default: 20)")
    g_post.add_argument("--kmeans_iters", type=int, default=30)
    g_post.add_argument("--max_cluster_pts", type=int, default=200_000,
                        help="Max points sampled for clustering (default: 200000)")
    g_post.add_argument("--dbscan_eps", type=float, default=0.3)
    g_post.add_argument("--dbscan_min_samples", type=int, default=10)
    g_post.add_argument("--hdbscan_min_cluster_size", type=int, default=50)
    g_post.add_argument("--hdbscan_min_samples", type=int, default=10)
    g_post.add_argument("--hdbscan_cluster_selection_epsilon", type=float,
                        default=0.06,
                        help="HDBSCAN cluster_selection_epsilon "
                             "(aligned with iggt_idmap.py --eps, default: 0.06)")
    g_post.add_argument("--postprocess_knn_k", type=int, default=0,
                        help="3D KNN smoothing before PCA/clustering "
                             "(0=disabled; iggt_idmap uses 20)")
    g_post.add_argument("--seed", type=int, default=0)
    g_post.add_argument("--palette_seed", type=int, default=0)
    g_post.add_argument("--render_views", type=int, default=0,
                        help="Render N random views of PLYs (0=skip)")
    g_post.add_argument("--render_resolution", type=int, default=4,
                        help="Resolution divisor for render (1/2/4/8, default 4)")
    g_post.add_argument(
        "--postprocess_seg3d_split", action="store_true", default=False,
        help="After clustering (kmeans/dbscan/hdbscan in --postprocess), "
             "also export per-cluster 3DGS PLYs under postprocess/seg3d_split/"
             "<tag>/; writes unclustered.ply when label==0 exists "
             "(same idea as instseg_infer seg3d_split)",
    )
    g_post.add_argument(
        "--postprocess_gt_split", action="store_true", default=False,
        help="When gaussian_ids exist (gt_idmap or .pt with gaussian_ids), "
             "export instance_XXXXX.ply per id under postprocess/gt_split/ "
             "(id==0 is background/ignore and is skipped)",
    )

    return parser


def main():
    argv_rest, cfg_file = _strip_trace_config_from_argv(sys.argv[1:])
    parser = build_arg_parser()
    parser.set_defaults(**_filter_trace_config_defaults(parser, cfg_file))
    args = parser.parse_args(argv_rest)

    if args.verbose:
        _coord_log = logging.getLogger("coord")
        _coord_log.setLevel(logging.DEBUG)
        if not _coord_log.handlers:
            _h = logging.StreamHandler(sys.stderr)
            _h.setLevel(logging.DEBUG)
            _h.setFormatter(logging.Formatter("%(message)s"))
            _coord_log.addHandler(_h)
        _coord_log.propagate = False

    # Parse IGGT image size
    iggt_w, iggt_h = [int(x) for x in args.iggt_image_size.split(",")]
    args._iggt_image_size = (iggt_w, iggt_h)
    sv_w, sv_h = [int(x) for x in args.segvggt_image_size.split(",")]
    args._segvggt_image_size = (sv_w, sv_h)
    # iggt_phys defaults to the iggt operating point on purpose: the instance
    # stream must stay comparable with feature_source=iggt (the acceptance
    # control group), so this is opt-in, not a silent new default.
    _ip_size = args.iggt_phys_image_size or args.iggt_image_size
    ip_w, ip_h = [int(x) for x in _ip_size.split(",")]
    args._iggt_phys_image_size = (ip_w, ip_h)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ==================================================================
    # Postprocess-only mode: load saved .pt and skip trace
    # ==================================================================
    if args.load_traced is not None:
        print(f"[load_traced] Loading {args.load_traced} ...")
        ckpt = torch.load(args.load_traced, map_location="cpu",
                          weights_only=True)
        final_feat = ckpt["feat"]       # [G, D]
        num_ray = ckpt["num_ray"]       # [G]
        ply_path = args.ply_path or ckpt.get("ply_path", None)
        if ply_path is None:
            parser.error("--ply_path is required (not found in .pt either)")
        ply_path = os.path.abspath(ply_path)

        G = final_feat.shape[0]
        n_valid = (num_ray > 0).sum().item()
        print(f"  Loaded feat [{G}, {final_feat.shape[1]}], "
              f"valid: {n_valid:,}/{G:,} ({100*n_valid/G:.1f}%)")

        if (
            args.postprocess is None
            and not args.postprocess_gt_split
            and not args.postprocess_seg3d_split
        ):
            parser.error(
                "--postprocess is required in --load_traced mode "
                "(unless --postprocess_gt_split or --postprocess_seg3d_split only)"
            )

        gaussian_ids = None
        if "gaussian_ids" in ckpt:
            gaussian_ids = ckpt["gaussian_ids"]

        run_postprocess(final_feat, num_ray, ply_path, output_dir, args,
                        gaussian_ids=gaussian_ids)

        if args.render_views > 0:
            source_path = args.source_path or ckpt.get("source_path")
            if source_path is None:
                print("[render] Skipped: --source_path required for "
                      "render_views")
            else:
                for attr in ("camera_backend", "transforms_json",
                             "transforms_axis"):
                    if attr in ckpt and ckpt[attr] is not None:
                        setattr(args, attr, ckpt[attr])
                _, _, sc_r, _, _ = load_gaussians_from_ply(ply_path)
                try:
                    args.resolved_trace_backend = effective_trace_backend(
                        args.trace_backend,
                        int(sc_r.shape[1]),
                        ckpt.get("trace_backend"),
                    )
                except ValueError as e:
                    parser.error(str(e))
                render_colored_views(
                    os.path.abspath(source_path), ply_path, output_dir, args
                )
        return

    # ==================================================================
    # Normal trace mode
    # ==================================================================
    if args.source_path is None or args.ply_path is None:
        parser.error("--source_path and --ply_path are required for trace mode")

    source_path = os.path.abspath(args.source_path)
    ply_path = os.path.abspath(args.ply_path)

    feature_source = _resolve_feature_source(args, parser)
    _validate_source_args(feature_source, args, parser)
    print(f"Feature source: {feature_source}")

    device = "cuda"
    bg_val = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    bg_color = torch.tensor(bg_val, dtype=torch.float32, device=device)

    # ---- 1. Load Gaussians ----
    means, quats, scales, opacities, colors = load_gaussians_from_ply(ply_path)
    N = means.shape[0]
    n_scales = int(scales.shape[1])
    try:
        args.resolved_trace_backend = resolve_trace_backend(
            args.trace_backend, n_scales,
        )
    except ValueError as e:
        parser.error(str(e))
    print(
        f"  trace_backend: {args.resolved_trace_backend} "
        f"(requested={args.trace_backend}, scales_ndim={n_scales})"
    )

    # ---- 2. Load cameras ----
    print(f"Loading cameras ({args.camera_backend}) ...")
    cameras = load_trace_cameras(
        args.camera_backend,
        source_path=source_path,
        images_folder=args.images_folder,
        resolution=args.resolution,
        transforms_json=args.transforms_json,
        transforms_axis=args.transforms_axis,
    )
    print(f"  {len(cameras)} cameras loaded")

    cam_list = cameras
    if args.max_views is not None:
        cam_list = cameras[:args.max_views]
        print(f"  Using first {len(cam_list)} views")

    # ---- 3. Prepare feature source ----
    prep_fn = FEATURE_PREP.get(feature_source)
    if prep_fn is None:
        parser.error(f"Unknown feature source: {feature_source}")
    prep = prep_fn(cam_list, args, device)

    feat_maps = prep["feat_maps"]
    trace_cams = prep["trace_cams"]
    feat_dim = prep["feat_dim"]
    masks = prep.get("masks")
    id_codec = prep.get("id_codec")
    query_bank = prep.get("query_bank")
    # Optional extra feature streams that must land on the SAME Gaussians and the
    # SAME cameras as the primary one (iggt-phys-pipeline ticket 04).  Each entry
    # is name -> list of [D, H, W], one per camera in cam_list.
    aux_feat_maps = prep.get("aux_feat_maps") or {}
    primary_name = prep.get("primary_name")
    provenance = prep.get("provenance")

    for name, amaps in aux_feat_maps.items():
        if len(amaps) != len(cam_list):
            parser.error(
                f"aux stream {name!r} has {len(amaps)} feature maps for "
                f"{len(cam_list)} cameras"
            )

    if args.save_render:
        os.makedirs(os.path.join(output_dir, "renders"), exist_ok=True)
    if args.dump_feat_maps:
        for name in [primary_name or "primary", *aux_feat_maps]:
            os.makedirs(os.path.join(args.dump_feat_maps, name), exist_ok=True)

    # ---- 4. Trace loop (unified across all sources) ----
    sum_gau_sem = torch.zeros(N, feat_dim, device=device, dtype=torch.float32)
    sum_num_ray = torch.zeros(N, device=device, dtype=torch.float32)
    aux_sum = {
        name: torch.zeros(N, amaps[0].shape[0], device=device, dtype=torch.float32)
        for name, amaps in aux_feat_maps.items()
    }
    sum_blend_mass = (
        torch.zeros(N, device=device, dtype=torch.float32)
        if args.trace_blend_mass else None
    )

    n_passes = (feat_dim + TRACE_CHANNELS - 1) // TRACE_CHANNELS
    aux_passes = sum(
        (amaps[0].shape[0] + TRACE_CHANNELS - 1) // TRACE_CHANNELS
        for amaps in aux_feat_maps.values()
    )
    print(f"\nTracing {len(cam_list)} views (feat_dim={feat_dim}, "
          f"TRACE_CHANNELS={TRACE_CHANNELS} x {n_passes} pass(es), "
          f"backend={args.resolved_trace_backend}) ...")
    if aux_feat_maps:
        dims = ", ".join(f"{n}={a[0].shape[0]}d" for n, a in aux_feat_maps.items())
        print(f"  + {len(aux_feat_maps)} aux stream(s) ({dims}) "
              f"= {aux_passes} more pass(es); {n_passes + aux_passes} total per view")

    for idx, cam in enumerate(tqdm(cam_list, desc="Trace")):
        trace_cam = trace_cams[idx]
        feat_2d = feat_maps[idx].to(device)
        H, W = trace_cam.image_height, trace_cam.image_width

        if feat_2d.shape[1] != H or feat_2d.shape[2] != W:
            feat_2d = F.interpolate(
                feat_2d.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        feat_hwc = feat_2d.permute(1, 2, 0).contiguous()

        if masks is not None:
            img_mask = masks[idx]
            if img_mask.shape[0] != H or img_mask.shape[1] != W:
                img_mask = F.interpolate(
                    img_mask.float().unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="nearest",
                ).squeeze(0).squeeze(0)
            img_mask = img_mask.to(device=device, dtype=torch.int32)
        else:
            img_mask = torch.ones(H, W, dtype=torch.int32, device=device)

        gau_sem, num_ray, radii, out_color = trace_single_view_chunked(
            means, quats, scales, opacities, colors,
            feat_hwc, img_mask, trace_cam, bg_color,
            args.resolved_trace_backend,
        )

        sum_gau_sem += gau_sem
        sum_num_ray += num_ray.float()

        # ---- 4b. Aux streams: same Gaussians, same camera, same mask ----
        # The invariant is enforced, not assumed.  num_ray is an integer per-Gaussian
        # hit count that depends only on geometry + camera + img_mask (never on the
        # feature values), and it is bit-exact across passes -- so an exact compare
        # is the right test here, and it is the strongest available proof that this
        # aux stream landed on the same Gaussians through the same camera.
        for name, amaps in aux_feat_maps.items():
            aux_2d = amaps[idx].to(device)
            if aux_2d.shape[1] != H or aux_2d.shape[2] != W:
                aux_2d = F.interpolate(
                    aux_2d.unsqueeze(0), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            aux_gau, aux_num_ray, _, _ = trace_single_view_chunked(
                means, quats, scales, opacities, colors,
                aux_2d.permute(1, 2, 0).contiguous(), img_mask, trace_cam, bg_color,
                args.resolved_trace_backend,
            )
            if not torch.equal(aux_num_ray, num_ray):
                n_diff = (aux_num_ray != num_ray).sum().item()
                raise RuntimeError(
                    f"view {idx} ({cam.image_name}): aux stream {name!r} traced a "
                    f"different Gaussian/ray set than the primary stream "
                    f"({n_diff:,} of {num_ray.numel():,} Gaussians differ in "
                    "num_ray).  The two streams are NOT on the same Gaussians / "
                    "cameras; refusing to write a .pt that claims they are."
                )
            aux_sum[name] += aux_gau

        if sum_blend_mass is not None:
            ones = torch.ones(H, W, 1, device=device, dtype=torch.float32)
            bm_gau, bm_num_ray, _, _ = trace_single_view_chunked(
                means, quats, scales, opacities, colors,
                ones, img_mask, trace_cam, bg_color, args.resolved_trace_backend,
            )
            if not torch.equal(bm_num_ray, num_ray):
                raise RuntimeError(
                    f"view {idx} ({cam.image_name}): the blend-mass pass traced a "
                    "different ray set than the primary stream."
                )
            sum_blend_mass += bm_gau[:, 0]

        if args.dump_feat_maps:
            torch.save(
                feat_maps[idx].detach().half().cpu(),
                os.path.join(args.dump_feat_maps, primary_name or "primary",
                             f"{cam.image_name}.pt"),
            )
            for name, amaps in aux_feat_maps.items():
                torch.save(
                    amaps[idx].detach().half().cpu(),
                    os.path.join(args.dump_feat_maps, name, f"{cam.image_name}.pt"),
                )

        if args.save_render:
            render_np = (
                out_color.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255
            ).astype(np.uint8)
            Image.fromarray(render_np).save(
                os.path.join(output_dir, "renders", f"{cam.image_name}.png")
            )

    # ---- 5. Normalize ----
    valid_mask = sum_num_ray > 0
    final_feat = torch.zeros_like(sum_gau_sem)
    final_feat[valid_mask] = (
        sum_gau_sem[valid_mask] / sum_num_ray[valid_mask].unsqueeze(-1)
    )
    # Same denominator, by construction: sum_num_ray is shared, which is only
    # legitimate because the assert above proved every aux pass saw the same rays.
    aux_final = {}
    for name, s in aux_sum.items():
        t = torch.zeros_like(s)
        t[valid_mask] = s[valid_mask] / sum_num_ray[valid_mask].unsqueeze(-1)
        aux_final[name] = t

    # ---- 6. Source-specific post-decode ----
    gaussian_ids = None
    if feature_source == "gt_idmap" and id_codec is not None:
        gaussian_ids = id_codec.decode(final_feat, valid_mask)
        n_assigned = (gaussian_ids > 0).sum().item()
        print(f"\n[gt_idmap decode] Recovered IDs for {n_assigned:,} Gaussians "
              f"({codec_ids_str(id_codec)})")

    # ---- 7. Print results and save ----
    n_valid = valid_mask.sum().item()
    print(f"\n========== Results ==========")
    print(f"  Feature source:    {feature_source}")
    print(f"  Trace backend:     {args.resolved_trace_backend}")
    print(f"  Gaussians total:   {N:,}")
    print(f"  Gaussians traced:  {n_valid:,} ({100 * n_valid / N:.1f}%)")
    print(f"  Feature dim:       {feat_dim}"
          + (f" ({primary_name})" if primary_name else ""))
    for name, t in aux_final.items():
        print(f"  Aux stream:        {name} [{t.shape[0]:,}, {t.shape[1]}]")
    print(f"  Views used:        {len(cam_list)}")

    source_label = feature_source if args.feature_source else "instance"
    out_path_pt = os.path.join(output_dir,
                               f"gaussian_{source_label}_feat.pt")
    save_dict = {
        "feat": final_feat.cpu(),
        "num_ray": sum_num_ray.cpu(),
        "feat_unnorm": sum_gau_sem.cpu(),
        "feat_dim": feat_dim,
        "n_views": len(cam_list),
        "ply_path": ply_path,
        "source_path": source_path,
        "feature_source": feature_source,
        "camera_backend": args.camera_backend,
        "transforms_json": args.transforms_json,
        "transforms_axis": args.transforms_axis,
        "trace_backend": args.resolved_trace_backend,
        "scales_ndim": n_scales,
    }
    if query_bank is not None:
        save_dict["query_bank"] = query_bank
    if provenance is not None:
        save_dict["provenance"] = provenance
    if aux_final or primary_name:
        # `gaussian_index` is the shared Gaussian index of every stream in this
        # file: row i of the compacted view is Gaussian gaussian_index[i].  All
        # gau_*_feat / num_ray arrays here are full length N, indexed the same way.
        save_dict["valid_mask"] = valid_mask.cpu()
        save_dict["gaussian_index"] = valid_mask.nonzero(as_tuple=True)[0].cpu()
    if primary_name:
        # Alias, not a copy: torch.save dedups shared storages within one file.
        save_dict[f"gau_{primary_name}_feat"] = save_dict["feat"]
    for name, t in aux_final.items():
        save_dict[f"gau_{name}_feat"] = t.cpu()
        save_dict[f"{name}_feat_dim"] = int(t.shape[1])
    if sum_blend_mass is not None:
        save_dict["blend_mass"] = sum_blend_mass.cpu()
        r = (sum_blend_mass[valid_mask] / sum_num_ray[valid_mask])
        print(f"  blend_mass/num_ray:  p05={r.quantile(0.05):.4f} "
              f"p50={r.quantile(0.5):.4f} p95={r.quantile(0.95):.4f} "
              f"max={r.max():.4f}  (1.0 would mean an unweighted ray mean)")
    if gaussian_ids is not None:
        save_dict["gaussian_ids"] = gaussian_ids.cpu()
    torch.save(save_dict, out_path_pt)
    print(f"  Saved to {out_path_pt}")

    out_path_npy = os.path.join(output_dir,
                                f"gaussian_{source_label}_feat.npy")
    np.save(out_path_npy, final_feat.cpu().numpy())
    print(f"  Saved to {out_path_npy}")

    if gaussian_ids is not None:
        ids_path = os.path.join(output_dir, "gaussian_gt_ids.npy")
        np.save(ids_path, gaussian_ids.cpu().numpy())
        print(f"  Saved decoded IDs to {ids_path}")

    if n_valid > 0:
        feat_norms = final_feat[valid_mask].norm(dim=1)
        print(f"\n  Feature stats (valid Gaussians):")
        print(f"    norm  min={feat_norms.min():.4f}  "
              f"mean={feat_norms.mean():.4f}  "
              f"max={feat_norms.max():.4f}")
        print(f"    num_ray  min={sum_num_ray[valid_mask].min():.0f}  "
              f"mean={sum_num_ray[valid_mask].mean():.1f}  "
              f"max={sum_num_ray[valid_mask].max():.0f}")

    # ---- 8. Post-processing (optional) ----
    if (
        args.postprocess
        or args.postprocess_gt_split
        or args.postprocess_seg3d_split
    ):
        run_postprocess(final_feat, sum_num_ray, ply_path, output_dir, args,
                        gaussian_ids=gaussian_ids)

    # ---- 9. Render colored views (optional) ----
    if args.render_views > 0:
        render_colored_views(source_path, ply_path, output_dir, args)


def codec_ids_str(codec):
    """Short summary string for IDMapCodec."""
    return f"{codec.num_ids} unique IDs, embed_dim={codec.embed_dim}"


if __name__ == "__main__":
    main()
