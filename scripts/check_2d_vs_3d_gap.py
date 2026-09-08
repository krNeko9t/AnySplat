"""04 号票复盘：分割差，差在模型还是差在我们的 3D 链路？

同一个视角上量三样东西，逐个 GT 实例对齐：

  (a) GT（``sam/mask``，756x1008，8 个实例——注意其中 7=地板、8=墙是 stuff）
  (b) **模型自己的 2D mask**（该视角所在那一批 decode 出来的，已按 mask_frac 去掉 stuff 槽）
  (c) **我们的 3D 实例投回该视角**（逐实例 one-hot 上色重渲染，按 alpha 加权投票 argmax）

(b) 是天花板，(c) 是交付物。(b) 低 ⇒ 模型的问题；(b)-(c) 的差 ⇒ trace/池化/阈值的问题。

用法：SEGVGGT_VIEW=02 PYTHONNOUSERSITE=1 python scripts/check_2d_vs_3d_gap.py
"""
import os, sys, json, numpy as np, torch, cv2
from pathlib import Path
R=Path.cwd(); sys.path.insert(0,str(R)); sys.path.insert(0,str(R/"scripts"))
import trace_instance_to_gaussians as T
from src.trace_cameras import load_trace_cameras
from src.trace_render.trace_rasterize import TRACE_CHANNELS, load_gaussians_from_ply, trace_single_view
from src.model.arch.segvggt_decode import decode_instances

S=Path(os.environ.get("SEGVGGT_SCENE","/home/liaoyuanjun/projects/instascene_preprocessed_data/3dovs/bench"))
OUT=Path(os.environ.get("SEGVGGT_TRACE_OUT",".scratch/segvggt-trace-3dgs/out/bench")); VIEW=os.environ.get("SEGVGGT_VIEW","02")
CK="/mnt/storage_pool/liaoyuanjun/runs/exp_phys_query_arm_b_lora/2026-09-07_17-09-03/checkpoints/epoch_114-step_20000.ckpt"
paths=sorted(p for p in (S/"images").iterdir() if p.suffix.lower() in (".jpg",".png",".jpeg"))
vi=[p.stem for p in paths].index(VIEW); b0=(vi//4)*4
print(f"view {VIEW} -> index {vi}, batch views {[p.stem for p in paths[b0:b0+4]]}")

gt=cv2.imread(str(sorted((S/"sam"/"mask").glob("*.png"))[vi]), cv2.IMREAD_UNCHANGED)
gids=[i for i in np.unique(gt) if i!=0]; H,W=gt.shape[:2]
print(f"GT {gt.shape} {len(gids)} instances, sizes {[int((gt==i).sum()) for i in gids]}")

def iou_report(name, pred_masks):
    """pred_masks: list of bool [H,W] at GT resolution."""
    rows=[]
    for i in gids:
        g=(gt==i)
        best=max(((p&g).sum()/max((p|g).sum(),1), j) for j,p in enumerate(pred_masks)) if pred_masks else (0,-1)
        rows.append((int(i), int(g.sum()), best[0], best[1]))
    m=np.mean([r[2] for r in rows])
    print(f"\n== {name}: mean best IoU = {m:.4f} ({len(pred_masks)} preds)")
    for r in rows: print(f"   gt{r[0]:>3} ({r[1]:>7,} px)  bestIoU={r[2]:.3f} -> pred {r[3]}")
    return m

# ---------- (b) 模型自己的 2D mask ----------
m=T.load_segvggt_model(CK,"cuda")
imgs=T.load_segvggt_images(paths[b0:b0+4], T.SEGVGGT_TRACE_WH, "cuda")
with torch.no_grad(): enc,_=m(imgs)
pred=enc.segvggt_prediction; qm=pred.query_masks[0]; Q,V,h,w=qm.shape
binm,sc2,_,qidx=decode_instances(pred.query_class_logits[0], qm.reshape(Q,-1), class_agnostic=True)
mf=binm.float().mean(1)
sel=[j for j in range(len(qidx)) if mf[j]<=0.3]
p2d=[cv2.resize(binm.view(-1,V,h,w)[j,vi-b0].numpy().astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool) for j in sel]
print(f"\nbatch queries: " + " ".join(f"q{int(qidx[j])}:{mf[j]*100:.1f}%" for j in range(len(qidx))))
iou2d=iou_report("(b) 模型 2D mask（本批，已去 stuff）", p2d)
del m; torch.cuda.empty_cache()

# ---------- (c) 我们的 3D 实例投回该视角 ----------
z=np.load(OUT/"instances/instances.npz", allow_pickle=True)
lab=z["labels"]; K=int(z["n_instances"]); print(f"\n3D: {K} instances, labelled {int((lab>=0).sum()):,}")
cams=load_trace_cameras("colmap", source_path=str(S), images_folder="images", resolution=1)
cam=[c for c in cams if c.image_name==VIEW][0]
print(f"render cam {cam.image_name} {cam.image_height}x{cam.image_width}")
ply=str(S/"point_cloud.ply")
means,quats,scales,opac,_=load_gaussians_from_ply(ply)
bg=torch.zeros(3,device="cuda")
acc=[]
for k in range(K):
    col=torch.zeros(means.shape[0],3,device="cuda"); col[torch.from_numpy(lab==k).cuda()]=1.0
    ish=torch.zeros(cam.image_height,cam.image_width,TRACE_CHANNELS,device="cuda")
    im=torch.ones(cam.image_height,cam.image_width,dtype=torch.int32,device="cuda")
    with torch.no_grad(): *_,oc=trace_single_view(means,quats,scales,opac,col,ish,im,cam,bg,"surfel")
    acc.append(oc[0].cpu().numpy())
A=np.stack(acc)                                  # [K,H,W] alpha-weighted vote
print(f"rendered {A.shape}, GT {gt.shape}")
if A.shape[1:]!=(H,W):
    A=np.stack([cv2.resize(a,(W,H),interpolation=cv2.INTER_LINEAR) for a in A])
p3d=[(A.argmax(0)==k)&(A.max(0)>0.25) for k in range(K)]
for k in range(K): print(f"   inst{k}: {p3d[k].sum():>8,} px projected")
iou3d=iou_report("(c) 我们的 3D 实例投回 2D", p3d)
print(f"\n>>> 全部 8 个 GT: 2D 天花板 {iou2d:.4f} -> 3D {iou3d:.4f} "
      f"(损失 {100*(iou2d-iou3d)/max(iou2d,1e-9):.1f}%)")

# gt7=地板 gt8=墙 是 stuff，class-agnostic 的这条链路本来就不做它们；物体单独再报一次
thing=[j for j,i in enumerate(gids) if int(i) not in (7,8)]
def sub(rows): return float(np.mean([rows[j] for j in thing]))
r2=[max(((p&(gt==i)).sum()/max((p|(gt==i)).sum(),1)) for p in p2d) for i in gids]
r3=[max(((p&(gt==i)).sum()/max((p|(gt==i)).sum(),1)) for p in p3d) for i in gids]
print(f">>> 只算 6 个物体: 2D {sub(r2):.4f} -> 3D {sub(r3):.4f} "
      f"(损失 {100*(sub(r2)-sub(r3))/max(sub(r2),1e-9):.1f}%)")
print(f">>> 3D 反而更好的 GT: {[int(gids[j]) for j in range(len(gids)) if r3[j]>r2[j]]}")

# 三联图：GT / 模型 2D / 我们的 3D
def paint(masks, base):
    o=base.copy(); rng=np.random.default_rng(1)
    for m in masks:
        c=rng.integers(60,255,3).tolist(); o[m]=(0.45*o[m]+0.55*np.array(c)).astype(np.uint8)
    return o
img=cv2.imread(str(paths[vi]))
panel=np.concatenate([paint([gt==i for i in gids], img), paint(p2d, img), paint(p3d, img)], axis=1)
fig=OUT/"gap"; fig.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(fig/f"{VIEW}_gt_2d_3d.png"), panel)
print(f"\n三联图（GT / 模型 2D / 我们的 3D）-> {fig/f'{VIEW}_gt_2d_3d.png'}")
