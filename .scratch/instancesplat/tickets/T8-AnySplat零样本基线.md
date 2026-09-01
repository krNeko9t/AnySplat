---
id: T8
title: AnySplat 零样本基线
type: wayfinder:task
status: closed
assignee: codebuddy
blocked-by: []
---

## Question

**HITL：需要人在集群上跑。**

测出 `hf:lhjiang/anysplat` 预训练权重在我们的 val 集（`processed_scannetpp_v2` +
`processed_re10k` 的 held-out 部分）上的**零样本重建指标**：PSNR / SSIM / LPIPS。

为什么必须有这个数：地图的验收门是**绝对**的（disc 路线已注销，没有相对基线）。
「重建不退化」这一条如果没有起点数字，训完根本无从判断——InstanceSplat 加了三个新 loss
并解冻了 backbone，重建质量掉下去是完全可能的，论文 Table 1 里 AnySplat 的 PSNR
（20.76 / 20.73 / 21.14 / 21.53）就是拿来做这个对照的。

要做的：

1. 确定 val 划分（哪些场景 held-out），**写死并记录**，之后所有评测都用同一批
2. 用现有 `AnySplatWrapper` 的 validation 路径跑一遍（`instance_feat_dim: 0`，纯重建）
3. 按视角数 2 / 4 / 8 分别报，和论文 Table 1 的行对齐着看

**产出**：一张表（视角数 × PSNR/SSIM/LPIPS）+ val 场景列表的落盘路径。
这张表会成为 T10 判读结果时的「不退化」门槛。

**注意**：这张票和 T1–T7 完全无关，任何时候都能跑，**建议第一个动手**——它是唯一一张
不依赖任何代码改动、又必须在集群上排队的票。

## R3 关闭后追加：顺手带两个探针（零代码）

这张票是**唯一一张不依赖任何代码改动、又必须在集群上排队**的票，人已经在 GPU 前面了，
顺手把 R3 findings 里两个「只能实测」的数字取回来，能消掉后面一大片不确定性：

1. **`voxelize_ratio`** —— R3 的显存估算里**不确定度最大的一项**（区间 `[0.16, 0.6]`，
   压缩比主要由视角间重叠决定）。仓库**已经在打这个点**：`anysplat.py:645` →
   `anysplat_wrapper.py:142`，**零代码，日志里读数就行**。
   按视角数 2 / 4 / 8 各记一次（正好和本票的三档对齐）。

2. **峰值显存**（`torch.cuda.max_memory_allocated`）按同样三档记一次。
   这是零样本、`instance_feat_dim: 0`、且 backbone 冻结的下界，不等于训练峰值——
   但它能把 R3 那张分项表里的「静态 + 激活」基数**钉死一个真实锚点**，
   让 T12 做完之后的估算不再全靠推算。

这两个数会直接回填 [R3](R3-40G显存预算.md) 的估算区间，并决定
[T12](T12-DPT栈梯度检查点与前向瘦身.md) 做完后是否还需要降视角数。
**它们不改变本票的验收产出**（PSNR/SSIM/LPIPS 表），只是搭便车。

## 解决

在集群（8×A100，conda env `anysplat`）上完成。产物：
- **脚本（协议载体，T10 复评必须原样复用）**：`scripts/zeroshot_baseline_anysplat.py`
- 主表：`.scratch/instancesplat/t8_zeroshot/`（`results.json` 逐场景逐指标含 context_indices / `summary.md` / `run.log`）
- 224×448 对照：`.scratch/instancesplat/t8_zeroshot_224/`（运行日志在 `.scratch/instancesplat/t8_zeroshot_224.log`）

### 1. val 划分（写死）

`/mnt/storage_pool/liaoyuanjun/data/InsScene-15K/manifest_val_instancesplat.jsonl`
——50 场景 = 25 `spp_` + 25 `re10k_`（清单同目录 `val_scene_ids_instancesplat.txt`），
路径已验证可解析、可加载。

⚠️ **泄漏警告（T7 必须处理）**：这 50 个场景 id **全部**存在于训练 manifest
`manifest_instancesplat_spp_re10k.jsonl`（overlap=50/50）。T7 写训练配置时必须
把 val 场景从训练侧剔除（生成 train-only manifest 或过滤），否则 T10 的
「不退化」对照被污染。

### 2. 锁定的评测协议

- 纯重建：N 个 context view 输入、在预测位姿下重渲染（与 `AnySplatWrapper.validation_step`
  同一前向路径，`num_target_views=0`）
- 视角采样：`ViewSamplerBoundedFixed`，stage=val，repo 默认 gap 参数
  （min/max distance 12/24、multipliers 3/5），N∈{2,4,8}，
  `seed=20260901` 按 (scene, N) 播种；采样到的 `context_indices` 已随 results.json 落盘
- 分辨率：长边 448，高按宽高比五档 bin {0.5, 0.625, 0.75, 0.875, 1.0} snap
  （spp 690×920 → 336×448；re10k 360×640 → 280×448）——与地图锁定配方一致
- 模型：`hf:lhjiang/anysplat`，`instance_feat_dim=0`，`pred_head_type=depth`、
  `anchor_feat_dim=128`（**对齐 HF 发布 config.json**；repo yaml 默认 point/83 与发布
  权重不符——脚本内 override，未改任何 yaml），`voxel_size=0.002`
- 前向 bf16 autocast（与 trainer `bf16-mixed` 一致），指标 fp32

### 3. 结果表（bin 分辨率，主表）

| 视角数 | 子集 | PSNR↑ | SSIM↑ | LPIPS↓ | voxelize_ratio | 峰值显存(GiB) |
|---|---|---|---|---|---|---|
| 2 | all | 30.29 | 0.9370 | 0.0730 | 1.000 | 3.77 |
| 2 | spp | 29.31 | 0.9415 | 0.0764 | 1.000 | |
| 2 | re10k | 31.27 | 0.9325 | 0.0696 | 1.000 | |
| 4 | all | 27.98 | 0.9091 | 0.1067 | 1.000 | 4.50 |
| 4 | spp | 25.66 | 0.8926 | 0.1342 | 1.000 | |
| 4 | re10k | 30.30 | 0.9256 | 0.0793 | 1.000 | |
| 8 | all | 27.03 | 0.8935 | 0.1168 | 1.000 | 6.11 |
| 8 | spp | 24.44 | 0.8709 | 0.1487 | 1.000 | |
| 8 | re10k | 29.62 | 0.9161 | 0.0849 | 1.000 | |

224×448（零样本模型原生训练分辨率）对照：all = 31.43 / 28.49 / 27.52 dB（N=2/4/8）。

### 4. 两个探针（R3 回填）

- **`voxelize_ratio` = 1.000，处处成立**（三档视角 × 两子集 × 两分辨率，零合并）。
  R3 估算区间 [0.16, 0.6] **被证伪**：`voxel_size=0.002` 在 VGGT canonical 尺度下
  小到体素化实际是 no-op，GS 数 = h·w·v **线性于视角数**（8 视角 @336×448 → 1.20M）。
  显存预算不能指望体素化压缩；「要不要调 voxel_size」从优化项变成了「要么接受 no-op、
  要么显著调大才可能真的合并」。
- **峰值显存**（推理、b=1、backbone 冻结的下界）：bin 分辨率 3.77 / 4.50 / 6.11 GiB
  @N=2/4/8；224 分辨率 3.47 / 3.96 / 5.01 GiB。前向步时 N=8 约 0.3 s（warm）。

### 5. 现象与判读要点（给 T10）

- **PSNR 随视角数单调下降**（30.29→27.98→27.03）。224 对照同样下降（31.43→28.49→27.52），
  说明这是 pose-free 多视角对齐误差随视角累积的真实效应，不是分辨率 OOD
  （分辨率 OOD 只贡献 ~0.5–1 dB，spp @2 视角最明显：32.02→29.31，-2.7 dB）。
  与论文 Table 1 的 NVS 趋势（视角越多越好）方向相反——协议不同（那边评 held-out
  新视角），不矛盾。
- 论文 Table 1 AnySplat 的 20.76/20.73/21.14/21.53 是 ScanNet held-out **NVS** 数字，
  与本表（重建、InsScene val、bin 分辨率）**不可直接对比**，只作量级参考。
- T10 判「不退化」时：同协议（同脚本、同 seed、同 bin）下逐视角数对照本表。

### 6. 顺带发现

- `manifest_instancesplat_spp_re10k.jsonl` 的 re10k 5138 条路径**全部**带
  `processed_re10k/` 前缀且文件存在 → [T13](T13-修复re10k-manifest路径.md) 担心的
  路径 bug 在**本机现有 manifest 产物上已不存在**（manifest 已按正确前缀重新生成过）；
  T13 剩余工作是脚本侧修复 + 复验。
- LPIPS(vgg) 依赖的 `vgg16-397923af.pth` 此前两次下载失败只剩 `.partial`，
  已用本机代理 `http://127.0.0.1:51390` 补齐（553 MB，`~/.cache/torch/hub/checkpoints/`）。
  `lhjiang/anysplat` 与 `facebook/VGGT-1B` 权重在 HF_HOME 缓存中完整，无需再下载。
