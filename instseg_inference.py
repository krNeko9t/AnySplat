from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from src.instseg.export import export_gaussian_instance_embedding
from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.model.ply_export import export_ply
from src.utils.image import process_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    ap.add_argument("--out_dir", type=str, default="outputs/instseg_infer")
    ap.add_argument("--hf_model", type=str, default="lhjiang/anysplat")
    ap.add_argument("--instance_feat_dim", type=int, default=16)
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base model, then attach an independent instance head by rebuilding the
    # model with `instance_feat_dim > 0` and loading weights non-strictly.
    base = AnySplat.from_pretrained(args.hf_model)
    encoder_cfg = deepcopy(base.encoder_cfg)
    encoder_cfg.instance_feat_dim = int(args.instance_feat_dim)
    encoder_cfg.pretrained_weights = ""  # avoid any config-based re-init
    model = AnySplat(encoder_cfg, deepcopy(base.decoder_cfg))
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    allowed_missing_prefixes = (
        "encoder.instance_head.",
        "encoder.instance_head_proj.",
    )
    bad_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
    print("[instseg_inference] Initialized from HF weights")
    print(
        f"[instseg_inference] missing_keys={len(missing)} "
        f"(allowed={len(missing) - len(bad_missing)}, unexpected={len(bad_missing)}), "
        f"unexpected_keys={len(unexpected)}"
    )
    if bad_missing:
        prefixes = {}
        for k in bad_missing:
            p = k.split(".", 2)[:2]
            p = ".".join(p) + "."
            prefixes[p] = prefixes.get(p, 0) + 1
        top = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:15]
        print("[instseg_inference] unexpected missing key prefixes (top):")
        for p, c in top:
            print(f"  - {p}: {c}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    imgs = [process_image(str(p)) for p in image_paths]  # [-1,1], [3,448,448]
    imgs = torch.stack(imgs, dim=0).unsqueeze(0).to(device)  # [1,V,3,H,W]
    b, v, _, h, w = imgs.shape
    imgs_01 = (imgs + 1) * 0.5

    # Encoder forward: get gaussians + gs-wise instance embedding.
    with torch.no_grad():
        encoder_output = model.encoder(imgs_01, global_step=0, visualization_dump=None)
    gaussians = encoder_output.gaussians
    pred_context_pose = encoder_output.pred_context_pose

    # Export PLY.
    ply_path = out_dir / "gaussians.ply"
    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        ply_path,
        save_sh_dc_only=True,
    )

    # Export pose video.
    if pred_context_pose is not None:
        save_interpolated_video(
            pred_context_pose["extrinsic"],
            pred_context_pose["intrinsic"],
            b,
            h,
            w,
            gaussians,
            str(out_dir),
            model.decoder,
        )

    # Export gs-wise embedding.
    if encoder_output.gaussian_instance_feat is not None:
        emb_path = export_gaussian_instance_embedding(
            out_dir / "gaussian_instance_embedding.pt",
            gaussians,
            encoder_output.gaussian_instance_feat,
            meta={"image_dir": str(image_dir), "hf_model": args.hf_model},
        )
        print(f"Saved gs-wise embedding to {emb_path}")
    else:
        print("gaussian_instance_feat is None. Did you set model.encoder.instance_feat_dim > 0?")


if __name__ == "__main__":
    main()

