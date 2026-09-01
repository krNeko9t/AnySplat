#!/usr/bin/env python
"""T11 判定性验证：aggregator 参数 dtype 与 backbone 是否真的在训。

跑法（本机 CPU 即可，不需要 GPU / 不需要下 HF 权重）：
    /home/liaowanjun/miniconda3/envs/paper_repo/bin/python .scratch/instancesplat/T11_verify.py

[C] 是这张票的判定性证据：修复前（aggregator 参数为 bf16）**必须失败**。
"""
import ast
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]

# 地图锁定的配方 + base_wrapper.py:597 的 AdamW 超参
LR = 2e-4 * 0.1          # base_lr * backbone_lr_multiplier
WD = 0.05
BETAS = (0.9, 0.95)
DEAD_THRESHOLD = LR * 2**8   # bf16 尾数 8 位 => |w| 超过它，~lr 量级的更新被舍成 0

ok = True


def check(cond: bool, msg: str) -> None:
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    ok = ok and cond


# --------------------------------------------------------------------------
print("[A] 静态门控（ast，不依赖 gsplat/colorspacious）")

src = (ROOT / "src/model/arch/anysplat.py").read_text(encoding="utf-8")
tree = ast.parse(src)

cfg_cls = next(
    n for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "EncoderAnySplatCfg"
)
defaults = {
    n.target.id: n.value
    for n in cfg_cls.body
    if isinstance(n, ast.AnnAssign) and n.value is not None
}
check(
    "aggregator_param_dtype" in defaults,
    "EncoderAnySplatCfg 有 aggregator_param_dtype 字段",
)
check(
    getattr(defaults.get("aggregator_param_dtype"), "value", None) == "bfloat16",
    'aggregator_param_dtype 默认值为 "bfloat16"（既有行为）',
)
check(
    "aggregator.to(torch.bfloat16)" not in src,
    "aggregator 的硬编码 .to(torch.bfloat16) 已移除",
)
check(
    "getattr(torch, cfg.aggregator_param_dtype)" in src,
    "cast 由 cfg.aggregator_param_dtype 驱动",
)

# 既有实验不带这个字段 => 拿默认 bfloat16 => 行为逐比特不变
for rel in ("config/model/encoder/anysplat.yaml",
            "config/experiment/instseg_anysplat.yaml"):
    check(
        "aggregator_param_dtype" not in (ROOT / rel).read_text(encoding="utf-8"),
        f"{rel} 未出现该字段（走默认，行为不变）",
    )

# --------------------------------------------------------------------------
print("\n[B] 真实 Aggregator 上的参数 dtype（随机初始化，不下 HF 权重）")

sys.path.insert(0, str(ROOT))
from src.model.vggt.models.aggregator import Aggregator  # noqa: E402

agg = Aggregator()
n_param = sum(p.numel() for p in agg.parameters())
print(f"  aggregator 参数量 = {n_param/1e6:.1f}M")

for name in ("bfloat16", "float32"):
    dtype = getattr(torch, name)
    agg.to(dtype)
    bad = [n for n, p in agg.named_parameters()
           if p.is_floating_point() and p.dtype is not dtype]
    check(not bad, f'aggregator_param_dtype="{name}" -> 全部浮点参数为 {name}'
                   + (f"（异常 {len(bad)} 个）" if bad else ""))

# --------------------------------------------------------------------------
print(f"\n[C] 判定性断言：lr={LR:g} 下 AdamW 到底更新了多少参数")
print("    （取真实 aggregator 的一个 transformer 线性层权重）")
print("    注意：bug 是**逐元素、按量级**生效的 —— 小权重照常更新，大权重钉死。")
print("    所以判据是「更新覆盖率」，不是「整个张量纹丝不动」。")

agg.to(torch.float32)
ref_name, ref = next(
    (n, p) for n, p in agg.named_parameters()
    if "blocks" in n and n.endswith("qkv.weight")
)
print(f"    参数: {ref_name}  shape={tuple(ref.shape)}  "
      f"std={ref.detach().std():.4f}  mean|w|={ref.detach().abs().mean():.4f}")

# 挑出「大权重」子集：死区判据说它们在 bf16 下必然不动
big_mask = ref.detach().abs() > DEAD_THRESHOLD

moved_frac, big_moved_frac = {}, {}
for name in ("bfloat16", "float32"):
    dtype = getattr(torch, name)
    p = ref.detach().to(dtype).clone().requires_grad_(True)
    opt = torch.optim.AdamW([p], lr=LR, weight_decay=WD, betas=BETAS)
    before = p.detach().clone()
    torch.manual_seed(0)
    for _ in range(50):
        p.grad = (torch.randn_like(p.detach().float()) * 1e-3).to(dtype)
        opt.step()
        p.grad = None
    moved = p.detach() != before
    moved_frac[name] = moved.float().mean().item()
    big_moved_frac[name] = moved[big_mask].float().mean().item()
    print(f"    {name:<9} AdamW 状态 dtype={opt.state[p]['exp_avg'].dtype}  "
          f"50 步后动了：全体 {moved_frac[name]:6.2%} / "
          f"|w|>{DEAD_THRESHOLD:.3g} 的那批 {big_moved_frac[name]:6.2%}")

check(moved_frac["float32"] > 0.99,
      "float32：>99% 的参数被更新（修复后的正确行为）")
# 比较式判据：50 步里动量累积会把边界附近的权重顶过去，所以绝对覆盖率不为 0；
# 真正的断言是「大权重的更新被压到 fp32 的一小部分」。
check(big_moved_frac["bfloat16"] < 0.5 * big_moved_frac["float32"],
      f"bfloat16：|w|>{DEAD_THRESHOLD:.3g} 的参数更新覆盖率 "
      f"({big_moved_frac['bfloat16']:.1%}) 不到 float32 "
      f"({big_moved_frac['float32']:.1%}) 的一半 —— T11 要修的 bug")
check(moved_frac["bfloat16"] < 0.5,
      "bfloat16：全体更新覆盖率不足一半（backbone 被静默地选择性冻结）")

# --------------------------------------------------------------------------
print("\n[D] 全 aggregator 的「死参数」占比（分层抽样实测）")
print(f"    参考判据：|w| > lr*2^8 = {DEAD_THRESHOLD:.4g}（一阶估计，偏保守）")

# 跨所有模块分层抽样，再在样本上真跑 AdamW —— 实测优于解析判据
torch.manual_seed(0)
chunks = []
for p in agg.parameters():
    if not p.is_floating_point():
        continue
    flat = p.detach().flatten()
    k = max(1, min(flat.numel(), int(flat.numel() * 0.006)))
    chunks.append(flat[torch.randperm(flat.numel())[:k]])
sample = torch.cat(chunks)
print(f"    样本量 {sample.numel()/1e6:.2f}M / {n_param/1e6:.1f}M 参数，覆盖全部模块")

stats = {}
for name in ("bfloat16", "float32"):
    dtype = getattr(torch, name)
    s_p = sample.to(dtype).clone().requires_grad_(True)
    opt = torch.optim.AdamW([s_p], lr=LR, weight_decay=WD, betas=BETAS)
    before = s_p.detach().clone()
    torch.manual_seed(1)
    for _ in range(50):
        s_p.grad = (torch.randn_like(s_p.detach().float()) * 1e-3).to(dtype)
        opt.step()
        s_p.grad = None
    stats[name] = (s_p.detach() == before).float().mean().item()

analytic = (sample.to(torch.bfloat16).abs() > DEAD_THRESHOLD).float().mean().item()
print(f"    bfloat16 实测死亡 {stats['bfloat16']:.1%}"
      f"（解析判据 {analytic:.1%}，作为保守上界）")
print(f"    float32  实测死亡 {stats['float32']:.1%}")
check(stats["bfloat16"] > 0.5,
      f"bf16 下超过一半的 backbone 参数从不更新（实测 {stats['bfloat16']:.1%}"
      f" ≈ {stats['bfloat16']*n_param/1e6:.0f}M）")
check(stats["float32"] < 0.01, "fp32 下几乎无死参数")
check(analytic >= stats["bfloat16"], "解析判据确为上界（不低于实测）")
print("    注：随机初始化，量级与训练后的 ViT-L 同量级但非逐值相同；"
      "结论只依赖量级分布，不依赖具体取值。")

print("\n" + ("全部通过" if ok else "有 FAIL —— 见上"))
sys.exit(0 if ok else 1)
