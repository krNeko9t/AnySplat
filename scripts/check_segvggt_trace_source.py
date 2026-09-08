"""03 号票判据：segvggt feature source 的三条检查。

① bank 里的 query_proj 与 traced field 是不是同一个空间
   （用模型自己的 einsum 重建 query_masks，必须逐位相等）；
② feature map 的 PCA 伪彩色贴回原图，边界对不对得上（05 号票的判据在这里复查）；
③ traced field 与 Q_all 点积出得来 3D logit，量级是否可用（结论交 04 号票）。

先跑一遍 trace（--feature_source segvggt）产出 .pt，再跑本脚本。
路径走环境变量：SEGVGGT_TRACE_OUT / SEGVGGT_SCENE / SEGVGGT_CKPT。
"""
import sys, numpy as np, torch, cv2
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import os
TRACE_OUT = Path(os.environ["SEGVGGT_TRACE_OUT"])   # trace 的 -o 目录
S = Path(os.environ.get("SEGVGGT_SCENE",
    "/home/liaoyuanjun/projects/instascene_preprocessed_data/3dovs/bench"))
CK = os.environ.get("SEGVGGT_CKPT",
    "/mnt/storage_pool/liaoyuanjun/runs/exp_phys_query_arm_b_lora/"
    "2026-09-07_17-09-03/checkpoints/epoch_114-step_20000.ckpt")
OUT = TRACE_OUT / "check03"; OUT.mkdir(parents=True, exist_ok=True)

import trace_instance_to_gaussians as T
from src.model.arch.segvggt_decode import decode_instances

paths = sorted(p for p in (S / "images").iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))[:4]
model = T.load_segvggt_model(CK, "cuda")

# ---------- ① bank 的 query_proj 与 field 是不是同一个空间 ----------
imgs = T.load_segvggt_images(paths, T.SEGVGGT_TRACE_WH, "cuda")
with torch.no_grad():
    enc, _ = model(imgs)
pred = enc.segvggt_prediction
feat = pred.feature_map[0].float()                 # [V,h,w,128]
qm = pred.query_masks[0]                           # [Q,V,h,w]
Q, V, h, w = qm.shape
proj = model.encoder.model.aggregator.instance_queries_proj
q_all = proj(pred.query_embed[0].float())          # [Q,128]
recon = torch.einsum("qd,vhwd->qvhw", q_all, feat)
d = (recon - qm.float()).abs()
print(f"① q_proj·feat 重建 query_masks: shape={tuple(qm.shape)} "
      f"maxabs={d.max().item():.3e} 相对={d.max().item()/qm.float().abs().max().item():.2e}")
ok1 = d.max().item() / qm.float().abs().max().item() < 1e-4

# ---------- ② PCA 伪彩色贴回原图 ----------
f0 = feat[0].reshape(-1, 128).cpu().numpy()
f0 = f0 - f0.mean(0)
u, s, vt = np.linalg.svd(f0, full_matrices=False)
pc = (f0 @ vt[:3].T).reshape(h, w, 3)
pc = (pc - pc.min((0, 1))) / (np.ptp(pc, axis=(0, 1)) + 1e-9)
bgr = cv2.imread(str(paths[0])); H, W = bgr.shape[:2]
pc_up = cv2.resize((pc * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(OUT / "pca.png"), pc_up[:, :, ::-1])
cv2.imwrite(str(OUT / "overlay.png"), cv2.addWeighted(pc_up[:, :, ::-1], 0.5, bgr, 0.5, 0))
print(f"② feature 栅格 {h}x{w} vs 原图 {H}x{W}（比例 {W/w:.2f}/{H/h:.2f}）"
      f" -> {OUT/'overlay.png'}")

# ---------- 2D mask 对 sam/mask 的 IoU（复现 05 号票的 0.6133）----------
_bin, sc, _lb, qi = decode_instances(pred.query_class_logits[0], qm.reshape(Q, -1),
                                     class_agnostic=True)
pm = _bin.view(-1, V, h, w)[:, 0].numpy()
gt_dir = S / "sam" / "mask"
gts = sorted(gt_dir.glob("*.png"))
gt = cv2.imread(str(gts[0]), cv2.IMREAD_UNCHANGED)
ids = [i for i in np.unique(gt) if i != 0]
ious = []
for i in ids:
    g = cv2.resize((gt == i).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    best = max(((p & g).sum() / max((p | g).sum(), 1) for p in pm.astype(bool)), default=0.0)
    ious.append(best)
print(f"② 逐视角 mask vs sam/mask（{len(ids)} 实例）mean best IoU = {np.mean(ious):.4f} "
      f"(05 号票在同一运行点读到 0.6133)")

# ---------- ③ 3D 侧：traced field 与 Q_all 点积 ----------
d = torch.load(TRACE_OUT / "gaussian_segvggt_feat.pt", map_location="cpu",
               weights_only=False)
gf = d["feat"].float(); qb = d["query_bank"]
print(f"③ gau_feat {tuple(gf.shape)}  Q_all {tuple(qb['query_proj'].shape)}  "
      f"batches={int(qb['batch_id'].max())+1}  traced={int((d['num_ray']>0).sum()):,}")
logits = gf @ qb["query_proj"].T                    # [P, M]
sel = logits > 0
cnt = sel.sum(0)
print(f"③ logits {tuple(logits.shape)}  每个 query 认领的高斯数: "
      f"min={int(cnt.min())} p50={int(cnt.median())} max={int(cnt.max())} "
      f"（总高斯 {gf.shape[0]:,}）")
claimed = sel.any(1).sum().item()
print(f"③ 被至少一个 query 认领的高斯 = {claimed:,} ({100*claimed/gf.shape[0]:.1f}%)")
print("\n① PASS" if ok1 else "\n① FAIL")
