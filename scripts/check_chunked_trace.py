"""检查 trace_single_view_chunked 与单趟 trace 等价（02 号票判据）：分趟 trace == 一趟 trace（到 kernel 自身底噪为止）+ 墙钟开销。

trace kernel 用 atomicAdd 累加，浮点加法顺序每次都不同 ⇒ **同一条路跑两遍都不逐位相等**。
所以判据不是 torch.equal，而是「新旧之差 <= 同一条路跑两遍之差」这个控制组。
"""
import os, sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.trace_cameras import load_trace_cameras
from src.trace_render.trace_rasterize import (
    TRACE_CHANNELS, load_gaussians_from_ply, resolve_trace_backend,
    trace_single_view, trace_single_view_chunked)

SCENE = os.environ.get(
    "CHUNKED_TRACE_SCENE",
    "/home/liaoyuanjun/projects/instascene_preprocessed_data/3dovs/bench",
)
means, quats, scales, opacities, colors = load_gaussians_from_ply(f"{SCENE}/point_cloud.ply")
backend = resolve_trace_backend("auto", scales.shape[1])
cams = load_trace_cameras("colmap", source_path=SCENE, images_folder="images", resolution=1)
cam = cams[0]; H, W = cam.image_height, cam.image_width
dev = means.device; bg = torch.zeros(3, device=dev)
img_mask = torch.ones(H, W, dtype=torch.int32, device=dev)
g = torch.Generator().manual_seed(0)
print(f"backend={backend} P={means.shape[0]:,} cam={cam.image_name} {H}x{W} views={len(cams)}")

def old_path(feat_hwc):
    """被替换掉的那条路：pad 到 TRACE_CHANNELS，一趟，切回去。"""
    d = feat_hwc.shape[2]; assert d <= TRACE_CHANNELS
    if d < TRACE_CHANNELS:
        feat_hwc = torch.cat([feat_hwc, torch.zeros(
            H, W, TRACE_CHANNELS - d, device=dev, dtype=torch.float32)], dim=2)
    gs, nr, rd, oc = trace_single_view(means, quats, scales, opacities, colors,
                                       feat_hwc.contiguous(), img_mask, cam, bg, backend)
    return gs[:, :d], nr, rd, oc

def new_path(feat_hwc):
    return trace_single_view_chunked(means, quats, scales, opacities, colors,
                                     feat_hwc, img_mask, cam, bg, backend)

def stats(x, y):
    d = (x - y).abs()
    return d.max().item(), d.mean().item()

# ---- 控制组：同一条路两遍，量底噪 ----
f20 = torch.rand(H, W, 20, generator=g).to(dev)
c1, c2 = old_path(f20)[0], old_path(f20)[0]
noise_max, noise_mean = stats(c1, c2)
scale = c1.abs().max().item()
print(f"\n[控制组] 同一条路两遍: maxabs={noise_max:.3e} mean={noise_mean:.3e} "
      f"(gau_sem absmax={scale:.3e}) ⇒ 底噪/量级 = {noise_max/scale:.2e}")

fail = 0
def check(tag, a, b, ref_max):
    global fail
    m, mn = stats(a, b)
    ok = m <= ref_max * 4  # 底噪本身 run-to-run 也在 2~4e-3 之间浮动，留 4× 余量
    print(f"{tag}: maxabs={m:.3e} mean={mn:.3e} 底噪={ref_max:.3e} -> {'PASS' if ok else 'FAIL'}")
    fail += not ok

# ---- 判据 1: 低维（bench_gt_idmap 用 id_embed_dim=20）新旧一致 ----
for D in (16, 20):
    feat = torch.rand(H, W, D, generator=g).to(dev)
    a_gs, a_nr, _, a_oc = old_path(feat)
    b_gs, b_nr, _, b_oc = new_path(feat)
    assert b_gs.shape == (means.shape[0], D), b_gs.shape
    check(f"[D={D:3d}] 新 vs 旧 gau_sem", a_gs, b_gs, noise_max)
    print(f"          num_ray 逐位相等={torch.equal(a_nr,b_nr)}  "
          f"out_color 逐位相等={torch.equal(a_oc,b_oc)}")
    fail += not torch.equal(a_nr, b_nr)

# ---- 判据 2: 128 维分 7 趟 ----
D = 128
feat = torch.rand(H, W, D, generator=g).to(dev)
n_pass = (D + TRACE_CHANNELS - 1) // TRACE_CHANNELS
c_gs, c_nr, _, _ = new_path(feat)
assert c_gs.shape == (means.shape[0], D), c_gs.shape
print(f"\n[D={D}] passes={n_pass} gau_sem shape={tuple(c_gs.shape)}")
manual = torch.cat([old_path(feat[:, :, s:min(s+TRACE_CHANNELS, D)].contiguous())[0]
                    for s in range(0, D, TRACE_CHANNELS)], dim=1)
check(f"[D={D}] 分趟 vs 手工 {n_pass} 段拼接", manual, c_gs, noise_max)

# num_ray 只累加一次（本票最容易埋的坑）
ok_nr = torch.equal(a_nr, c_nr)
print(f"[D={D}] num_ray 与 D=20 那趟逐位相等={ok_nr}  sum={c_nr.float().sum().item():.0f}  "
      f"(若误累加 {n_pass} 次会是 {c_nr.float().sum().item()*n_pass:.0f})")
fail += not ok_nr

# ---- 判据 3: 墙钟 ----
def bench(fn, n=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
t1 = bench(lambda: new_path(f20)); t7 = bench(lambda: new_path(feat))
print(f"\n墙钟: 1 趟(D=20)={t1:.1f} ms/view  7 趟(D=128)={t7:.1f} ms/view  比值={t7/t1:.2f}×")
print(f"外推: bench 36 视角={t7*36/1000:.1f}s  garden 185 视角={t7*185/1000:.1f}s")
print("\nRESULT:", "PASS" if fail == 0 else f"FAIL ({fail})")
sys.exit(fail != 0)
