# 01 — 实测：每份 config 的真实可训集合

> 产出票：`.scratch/freeze-contract/issues/01-measure-current-trainable-sets.md`
> 测量日期：2026-09-04 · 分支 `fix_freeze` @ `b8d943a` · 本机 CPU（无 GPU）

## 怎么测的（口径，复现用）

探针：`.scratch/freeze-contract/scripts_probe.py`，渲染：`.scratch/freeze-contract/scripts_report.py`。

```bash
conda run -n paper_repo python .scratch/freeze-contract/scripts_probe.py <experiment> \
    -o model.encoder.pretrained_weights=""
conda run -n paper_repo python .scratch/freeze-contract/scripts_report.py   # 重新渲染本文件
```

探针复刻 `src/main.py` 的真实路径：hydra `compose(+experiment=X)` → `load_typed_root_config`
→ `get_model` → 按 `EncoderSegVGGTCfg` / `EncoderIGGTCfg` 分发 wrapper → **`wrapper.setup("fit")`**
（`requires_grad` 达到终态的唯一位置，`base_wrapper.py:501-528`）→ 逐参数快照。
`param_groups` 一栏是就地**重放** `configure_optimizers`（`base_wrapper.py:508-596`）的分组规则，
不构造 optimizer，因此不受 CPU/显存限制。不跑 forward，不加载权重。

**四点口径偏差，读数时须知**：

1. **`pretrained_weights` 一律置空**。本机挂不到 `/mnt/storage_pool`、`/mnt/shared-storage-gpfs2`。
   加载权重只改**值**不改 `requires_grad` / dtype / 形状（`weight_loading.py` 是 `load_state_dict`
   路径），所以对本票的四项测量无影响。**唯一例外**是 `segvggt_scannet` 本来就写着
   `pretrained_weights: ""`——它的实测即线上实测。
2. **IGGT 五份用 `--random-backbone` 建 VGGT 底座**：`iggt.py:79` 的
   `VGGT.from_pretrained("facebook/VGGT-1B")` 被换成 `VGGT()`。`from_pretrained` =
   构造 + `load_state_dict`，只改**值**；名字、形状、dtype、`requires_grad` 都不变，
   而 `iggt.py:80` 的 `.to(torch.bfloat16)` 照样在结果上执行——所以本票测的四项不受影响。
   这么做只是为了不下那 5GB 权重（本机没有缓存，实测下行 ~110MB/min）。
   SegVGGT 四份不走这条路径（`segvggt.py` 自建 backbone），全程真实构造。
3. **单卡 CPU、无 DDP**。`global_rank` 恒 0；`freeze_keywords` 的应用与 rank 无关。
4. **缺失的编译期依赖被打桩**（`gsplat` / `diff_*_rasterization` / `torch_scatter` / `pytorch3d`
   / `e3nn` / `hdbscan` / `open3d` / `pycolmap` / `pillow_heif` / `moviepy`）。全部是渲染/可视化/
   聚类的 import-time 依赖，不参与参数构造；打桩对象一旦被真正调用会立刻 `RuntimeError`，
   全程没有触发。为跑通探针，`paper_repo` env 额外装了纯 python 包：`hydra-core lightning
   beartype timm tabulate colorama wandb matplotlib plyfile trimesh lpips scikit-video colorspacious`。

逐参数原始记录（name → numel / dtype / requires_grad）在 `raw/<experiment>.json` 的 `params` 字段，
就是 03 号票要设计的那个指纹的**样本数据**。

## 总表

| experiment | 总参数 | 可训 | 占比 | 可训集合 dtype | `<default>` 组 |
|---|---:|---:|---:|---|---|
| `segvggt_finetune_agnostic` | 1654.3M | **486.879M** | 29.4% | 全 fp32 | legacy 模式，全量落 backbone 组 |
| `segvggt_agnostic_phys_joint` | 1654.5M | **487.079M** | 29.4% | 全 fp32 | 486.879M（非空） |
| `segvggt_physgm` | 1654.5M | **0.200M** | 0.01% | 全 fp32 | legacy 模式，全量落 backbone 组 |
| `segvggt_scannet` | 1654.3M | **1148.356M** | 69.4% | 全 fp32 | 652.046M（非空） |
| `instseg_iggt` | 1216.2M | **90.936M** | 7.5% | 全 fp32 | **空** |
| `phys_iggt` | 1230.2M | **79.324M** | 6.4% | 全 fp32 | **空** |
| `phys_prop_iggt` | 1230.2M | **14.015M** | 1.1% | 全 fp32 | **空** |
| `physgm_iggt` | 1216.6M | **0.400M** | 0.03% | 全 fp32 | **空** |
| `physgm_dpt_iggt` | 1230.2M | **14.019M** | 1.1% | 全 fp32 | **空** |

九份配方全部通过 `freeze_keywords` 的零命中护栏（无空关键词）。

## 与 `docs/repo_knowledge.md` 声称意图的逐份对照

| experiment | 文档声称 | 实测 | 结论 |
|---|---|---|---|
| `segvggt_finetune_agnostic` | `:118` 487M 可训；`patch_embed`→344/304M、`instance_`→1011/454M、`semantic_head`→62/32.6M、`lora_`→192/9.44M | 486.879M；344/304.373M；（该配方不冻 `instance_`，命中数由 `segvggt_physgm` 侧核对：1011/454.251M）；`semantic_head` 62/32.628M | **一致** |
| `segvggt_agnostic_phys_joint` | `:126` 可训 = `instance_` + `semantic_head` + `query_physgm`；readout「提到 ~1e-4」 | 487.079M，构成正确；**readout 实际 5.00e-4，pretrained 侧 1.00e-4** | **不一致（见 F1）** |
| `segvggt_physgm` | `:148` 冻死分割只训 ~0.2M readout | 0.200M，只有 `query_physgm.decoders.{0,1,2}` | **一致** |
| `segvggt_scannet` | `:118` 冻结集只有 `[patch_embed]`，可训 1148M；LoRA 只冻被包裹的 attention 基座（202M），MLP/norm（403M）仍可训 | 1148.356M；构造期冻结 193 张量；`frame_blocks`/`global_blocks` 各 307.083M 中各 206.322M 可训（差额 100.761M×2 = 201.5M 即 LoRA 基座） | **一致** |
| `instseg_iggt` | 无专门段落；yaml 冻 `[aggregator, camera_head]` | 90.936M = `part_adaptor`+`part_head`(25.627M) + `point_head`+`depth_head`(65.309M) | **一致** |
| `phys_iggt` | 无专门段落；yaml 冻 `[aggregator, camera_head, part_adaptor, part_head]` | 79.324M = `physics_scheme`(14.015M) + `point_head`+`depth_head`(65.309M) | **一致**（但见 F4） |
| `phys_prop_iggt` | 同上再冻 `point_head, depth_head` | 14.015M，只有 `physics_scheme` | **一致** |
| `physgm_iggt` | 同上 | 0.400M，只有 `physics_scheme.physgm_readout.decoders` | **一致** |
| `physgm_dpt_iggt` | 同上 | 14.019M = `physics_head` + `physgm_dense_readout` | **一致** |

`repo_knowledge.md:222` 的一句**与代码不符**，见 F5。

## 发现（只记录，不修 —— 修法归 04/06 号票）

### F1 · `segvggt_agnostic_phys_joint` 的 lr 比注释声称的高 5 倍

`config/experiment/segvggt_agnostic_phys_joint.yaml:66` 写着

```yaml
    - keywords: [query_physgm]
      lr_multiplier: 5.0   # 2e-5 * 5 = 1e-4
```

但同一段的 `:49` 是 `lr: 1.0e-4`，不是注释假设的 `2e-5`。实际结果：

- `query_physgm`（15 张量 / 0.200M，随机 init）→ **5.00e-4**，注释想要 1e-4；
- pretrained 的 `instance_` + `semantic_head`（1073 张量 / 486.879M）→ **1.00e-4**，
  而 `:47-48` 自陈「stays cool (same as corrected stage-1)」，stage-1 是 2e-5。

两侧都偏热 5 倍。这是**纯数值型的意图-事实偏差**：`freeze_keywords` 一个字没错、
可训集合完全正确、零命中护栏也不响——现有的两道护栏结构上抓不到它。
**这是 03 号票「指纹里要不要放 `param_groups` 归属 + 实际 lr」的第一个实物论据。**

### F2 · `register_token` 关键词多冻了一个 DINO 张量（当前良性）

裸子串匹配（`base_wrapper.py:512`）下，关键词 `register_token` 在三份 segvggt 配方里
命中的是 **2 个张量 / 12288 参数**，而不是地图事实底座记的 8192：

```
model.encoder.model.aggregator.register_token            8192
model.encoder.model.aggregator.patch_embed.register_tokens  4096   ← 多出来的
```

多出来的那个是 DINOv2 自带的 register tokens，它同时被 `patch_embed` 命中，
所以**今天完全良性**（本来就要冻）。但它证明：keyword 的命中数**不能**当作
「我冻的就是我想冻的那个」的证据。九份配方里这是唯一一处多关键词重叠。
→ 03 号票「指纹粒度」：rollup 级别看不出这个，逐参数级别才看得出。

### F3 · bf16 静默冻结在这九份配方里**一次都没发生**，但离发生只差一行

IGGT 路线五份配方的 aggregator 都是 bf16（`iggt.py:80` 无条件 `.to(torch.bfloat16)`，
909.112M 参数），但五份**全部**把 `aggregator` 写进了 `freeze_keywords`，所以：

- 九份配方的**可训集合无一例外全是 fp32**；
- 06 号票候选判据「进了优化器的参数里有非 fp32 的」在今天**零次触发**。

也就是说：这个判据现在是**纯前瞻性**的护栏，不是在修一个现存 bug。它的价值在于，
任何人把 `aggregator` 从 `freeze_keywords` 里拿掉（比如为了放开 LoRA、或换一个
只冻部分 block 的配方），909M bf16 参数就会**静默**进 AdamW —— 而 `requires_grad`
全程为真、零命中护栏不响、loss 曲线看不出来。07 号票补的 dtype 开关是这条路的前置。

SegVGGT 路线（`arch/segvggt.py`）无永久 cast，四份配方全模型 fp32，与地图事实底座一致。

### F4 · `phys_iggt` 在物理阶段仍然训练 `point_head` / `depth_head`

`phys_iggt` 冻 `[aggregator, camera_head, part_adaptor, part_head]`，几何的
`point_head` + `depth_head`（124 张量 / 65.309M）**仍可训**，且被显式写进
`param_groups` 的 0.1 倍组（3.00e-6）——看起来是刻意的，不是漏冻。

但它意味着：跑 `phys_iggt` 之后的 ckpt，其几何与输入 ckpt **不再逐位相同**。
同一条链上的 `phys_prop_iggt` / `physgm_iggt` / `physgm_dpt_iggt` 则把这两个头冻死。
同一路线的相邻配方对「几何该不该动」给出了相反的答案。
**这不是本票能定的事，交给 05 号票**（跨 stage 关系的约束形式）。

### F5 · `repo_knowledge.md:222` 关于 `freeze_module` 的一句是错的

文档写：`EncoderAnySplat` 自带 `freeze_backbone`/`freeze_module`，「**仅在 freeze_keywords
为空时生效**」。代码里没有这个条件：`anysplat.py:157-193` 在 `__init__` 里无条件执行，
而 `BaseWrapper.setup()` 在**之后**才跑。两者是顺序关系，不是互斥关系。

方向还反了一层：`:191-193` 的兜底分支是**无条件赋值**

```python
param.requires_grad = (freeze_module not in name and "distill" not in name)
```

它会把构造期已冻的参数**解冻**，而 `freeze_keywords` 是只冻不解冻的增量语义，
不会把它们冻回去（除非关键词恰好命中）。今天没咬人只是因为用 `freeze_module` 的
5 份 config 一份都没设 `freeze_keywords`（地图事实底座已核实）。
→ 直接喂给 **02 号票**的方案 2 处置。

### F6 · 五份 IGGT 配方的 `backbone_lr_multiplier` 是死配置

`instseg_iggt` / `phys_iggt` / `phys_prop_iggt` / `physgm_iggt` / `physgm_dpt_iggt`
的 `param_groups` 覆盖了全部可训参数，`<default>` 组**恒为空**，
所以 `backbone_lr_multiplier: 0.1` 在这五份里对任何参数都不起作用。
（`segvggt_scannet` / `segvggt_agnostic_phys_joint` 的 default 组非空，是活的。）
无害，但它是「配方里写了、实际没生效」的一类——如果指纹带 lr，这类会被顺带照出来。

### F7 · legacy 分支会造一个空的 param group

`segvggt_finetune_agnostic` 与 `segvggt_physgm` 没写 `param_groups`，走
`base_wrapper.py:572-595` 的 legacy 路径，`new_param_keywords` 回落到默认的
`["gaussian_param_head", "interm"]`——这两个词在 SegVGGT 上**零命中**，于是
`param_dicts[0]` 是个空组，全部可训参数落进 backbone 组
（两份的 `backbone_lr_multiplier` 都是 1.0，所以实际 lr 就是 base lr，行为正确）。
空 param group 对 AdamW 合法（已验证），只是 `LearningRateMonitor` 会给一个
不训任何东西的组打 lr 日志。**行为无误，记录备查。**

## 顺带产出：stage 链的集合关系（数字，交给 05 号票判定）

以逐参数名集合计算（`raw/*.json` 的 `params` 字段）：

- stage-1 `segvggt_finetune_agnostic` 训了 **1073** 个张量；
- stage-2 `segvggt_physgm` 的**冻结集完全覆盖**它：`trained(stage1) \ frozen(stage2)` = **0**；
- `trained(stage1) ∩ trained(stage2)` = **0**；
- joint `segvggt_agnostic_phys_joint` 训了 **1088** 个张量，
  `trained(joint) ⊇ trained(stage1)` 成立，差集正是 15 个 `query_physgm.decoders.*`。

即：今天这条链是**自洽的**。05 号票要定的是用什么形式把这个自洽性**锁住**，
以及 joint 这种「不是任何 stage 的续作」的配方该被约束成什么。

## 每份配方的完整读数

### segvggt_finetune_agnostic

- wrapper `SegVGGTWrapper` · base lr `2e-05` · 总参数 1654.258M · **可训 486.879M** (29.4%)
- 构造期就已冻结（`setup()` 之前）：**193** 个参数张量

**(1) freeze_keywords 命中**

| keyword | 命中张量 | 命中参数量 |
|---|---:|---:|
| `patch_embed` | 344 | 304.373M |
| `frame_blocks` | 528 | 307.083M |
| `global_blocks` | 528 | 307.083M |
| `camera_head` | 69 | 216.175M |
| `depth_head` | 62 | 32.655M |
| `camera_token` | 1 | 0.002M |
| `register_token` | 2 | 0.012M |

**(2) 实际 requires_grad=True 的集合（按模块 rollup）**

| 模块 | 张量 | 参数量 |
|---|---:|---:|
| `model.encoder.model.aggregator.instance_cross_blocks` | 576 | 302.414M |
| `model.encoder.model.aggregator.instance_query_self_attn` | 432 | 151.296M |
| `model.encoder.model.semantic_head.scratch` | 46 | 15.318M |
| `model.encoder.model.semantic_head.resize_layers` | 6 | 11.536M |
| `model.encoder.model.semantic_head.projects` | 8 | 5.770M |
| `model.encoder.model.aggregator.instance_query_token` | 1 | 0.410M |
| `model.encoder.model.aggregator.instance_queries_proj` | 2 | 0.131M |
| `model.encoder.model.semantic_head.norm` | 2 | 0.004M |
| **合计** | | **486.879M** |

**(3) dtype**

- 可训集合：float32 486.879M
- 全模型：float32 1654.258M

**(4) param_groups 归属与实际 lr**（`legacy new_param_keywords`）

| 组 | keywords | lr_multiplier | 实际 lr | 张量 | 参数量 | dtype |
|---|---|---:|---:|---:|---:|---|
| | `gaussian_param_head`, `interm` | 1.0 | 2.00e-05 | 0 | 0.000M | — |
| | `<backbone>` | 1.0 | 2.00e-05 | 1073 | 486.879M | float32 486.879M |

### segvggt_agnostic_phys_joint

- wrapper `SegVGGTWrapper` · base lr `0.0001` · 总参数 1654.458M · **可训 487.079M** (29.4%)
- 构造期就已冻结（`setup()` 之前）：**193** 个参数张量

**(1) freeze_keywords 命中**

| keyword | 命中张量 | 命中参数量 |
|---|---:|---:|
| `patch_embed` | 344 | 304.373M |
| `frame_blocks` | 528 | 307.083M |
| `global_blocks` | 528 | 307.083M |
| `camera_head` | 69 | 216.175M |
| `depth_head` | 62 | 32.655M |
| `camera_token` | 1 | 0.002M |
| `register_token` | 2 | 0.012M |

**(2) 实际 requires_grad=True 的集合（按模块 rollup）**

| 模块 | 张量 | 参数量 |
|---|---:|---:|
| `model.encoder.model.aggregator.instance_cross_blocks` | 576 | 302.414M |
| `model.encoder.model.aggregator.instance_query_self_attn` | 432 | 151.296M |
| `model.encoder.model.semantic_head.scratch` | 46 | 15.318M |
| `model.encoder.model.semantic_head.resize_layers` | 6 | 11.536M |
| `model.encoder.model.semantic_head.projects` | 8 | 5.770M |
| `model.encoder.model.aggregator.instance_query_token` | 1 | 0.410M |
| `model.encoder.model.aggregator.instance_queries_proj` | 2 | 0.131M |
| `model.encoder.query_physgm.decoders.0` | 5 | 0.067M |
| `model.encoder.query_physgm.decoders.1` | 5 | 0.067M |
| `model.encoder.query_physgm.decoders.2` | 5 | 0.067M |
| `model.encoder.model.semantic_head.norm` | 2 | 0.004M |
| **合计** | | **487.079M** |

**(3) dtype**

- 可训集合：float32 487.079M
- 全模型：float32 1654.458M

**(4) param_groups 归属与实际 lr**（`param_groups`）

| 组 | keywords | lr_multiplier | 实际 lr | 张量 | 参数量 | dtype |
|---|---|---:|---:|---:|---:|---|
| | `query_physgm` | 5.0 | 5.00e-04 | 15 | 0.200M | float32 0.200M |
| | `<default>` | 1.0 | 1.00e-04 | 1073 | 486.879M | float32 486.879M |

### segvggt_physgm

- wrapper `SegVGGTWrapper` · base lr `0.0001` · 总参数 1654.458M · **可训 0.200M** (0.0%)
- 构造期就已冻结（`setup()` 之前）：**193** 个参数张量

**(1) freeze_keywords 命中**

| keyword | 命中张量 | 命中参数量 |
|---|---:|---:|
| `patch_embed` | 344 | 304.373M |
| `frame_blocks` | 528 | 307.083M |
| `global_blocks` | 528 | 307.083M |
| `camera_head` | 69 | 216.175M |
| `depth_head` | 62 | 32.655M |
| `semantic_head` | 62 | 32.628M |
| `instance_` | 1011 | 454.251M |
| `camera_token` | 1 | 0.002M |
| `register_token` | 2 | 0.012M |

**(2) 实际 requires_grad=True 的集合（按模块 rollup）**

| 模块 | 张量 | 参数量 |
|---|---:|---:|
| `model.encoder.query_physgm.decoders.0` | 5 | 0.067M |
| `model.encoder.query_physgm.decoders.1` | 5 | 0.067M |
| `model.encoder.query_physgm.decoders.2` | 5 | 0.067M |
| **合计** | | **0.200M** |

**(3) dtype**

- 可训集合：float32 0.200M
- 全模型：float32 1654.458M

**(4) param_groups 归属与实际 lr**（`legacy new_param_keywords`）

| 组 | keywords | lr_multiplier | 实际 lr | 张量 | 参数量 | dtype |
|---|---|---:|---:|---:|---:|---|
| | `gaussian_param_head`, `interm` | 1.0 | 1.00e-04 | 0 | 0.000M | — |
| | `<backbone>` | 1.0 | 1.00e-04 | 15 | 0.200M | float32 0.200M |

### segvggt_scannet

- wrapper `SegVGGTWrapper` · base lr `6e-05` · 总参数 1654.252M · **可训 1148.356M** (69.4%)
- 构造期就已冻结（`setup()` 之前）：**193** 个参数张量

**(1) freeze_keywords 命中**

| keyword | 命中张量 | 命中参数量 |
|---|---:|---:|
| `patch_embed` | 344 | 304.373M |

**(2) 实际 requires_grad=True 的集合（按模块 rollup）**

| 模块 | 张量 | 参数量 |
|---|---:|---:|
| `model.encoder.model.aggregator.instance_cross_blocks` | 576 | 302.414M |
| `model.encoder.model.aggregator.frame_blocks` | 432 | 206.322M |
| `model.encoder.model.aggregator.global_blocks` | 432 | 206.322M |
| `model.encoder.model.camera_head.trunk` | 56 | 201.449M |
| `model.encoder.model.aggregator.instance_query_self_attn` | 432 | 151.296M |
| `model.encoder.model.depth_head.scratch` | 46 | 15.344M |
| `model.encoder.model.semantic_head.scratch` | 46 | 15.312M |
| `model.encoder.model.camera_head.poseLN_modulation` | 2 | 12.589M |
| `model.encoder.model.depth_head.resize_layers` | 6 | 11.536M |
| `model.encoder.model.semantic_head.resize_layers` | 6 | 11.536M |
| `model.encoder.model.depth_head.projects` | 8 | 5.770M |
| `model.encoder.model.semantic_head.projects` | 8 | 5.770M |
| `model.encoder.model.camera_head.pose_branch` | 4 | 2.107M |
| `model.encoder.model.aggregator.instance_query_token` | 1 | 0.410M |
| `model.encoder.model.aggregator.instance_queries_proj` | 2 | 0.131M |
| `model.encoder.model.camera_head.embed_pose` | 2 | 0.020M |
| `model.encoder.model.aggregator.register_token` | 1 | 0.008M |
| `model.encoder.model.camera_head.token_norm` | 2 | 0.004M |
| `model.encoder.model.camera_head.trunk_norm` | 2 | 0.004M |
| `model.encoder.model.depth_head.norm` | 2 | 0.004M |
| `model.encoder.model.semantic_head.norm` | 2 | 0.004M |
| `model.encoder.model.aggregator.camera_token` | 1 | 0.002M |
| `model.encoder.model.camera_head.empty_pose_tokens` | 1 | 0.000M |
| **合计** | | **1148.356M** |

**(3) dtype**

- 可训集合：float32 1148.356M
- 全模型：float32 1654.252M

**(4) param_groups 归属与实际 lr**（`param_groups`）

| 组 | keywords | lr_multiplier | 实际 lr | 张量 | 参数量 | dtype |
|---|---|---:|---:|---:|---:|---|
| | `instance_`, `semantic_head`, `lora_` | 3.333 | 2.00e-04 | 1265 | 496.310M | float32 496.310M |
| | `<default>` | 1.0 | 6.00e-05 | 805 | 652.046M | float32 652.046M |

### instseg_iggt

- wrapper `IGGTWrapper` · base lr `3e-05` · 总参数 1216.223M · **可训 90.936M** (7.5%)
- 构造期就已冻结（`setup()` 之前）：**1** 个参数张量

**(1) freeze_keywords 命中**

| keyword | 命中张量 | 命中参数量 |
|---|---:|---:|
| `aggregator` | 1210 | 909.112M |
| `camera_head` | 69 | 216.175M |

**(2) 实际 requires_grad=True 的集合（按模块 rollup）**

| 模块 | 张量 | 参数量 |
|---|---:|---:|
| `model.encoder.point_head.resize_layers.3` | 2 | 9.438M |
| `model.encoder.depth_head.resize_layers.3` | 2 | 9.438M |
| `model.encoder.part_adaptor.resize_layers.0` | 26 | 4.723M |
| `model.encoder.point_head.scratch.refinenet1` | 10 | 2.426M |
| `model.encoder.point_head.scratch.refinenet2` | 10 | 2.426M |
| `model.encoder.point_head.scratch.refinenet3` | 10 | 2.426M |
| `model.encoder.depth_head.scratch.refinenet1` | 10 | 2.426M |
| `model.encoder.depth_head.scratch.refinenet2` | 10 | 2.426M |
| `model.encoder.depth_head.scratch.refinenet3` | 10 | 2.426M |
| `model.encoder.point_head.scratch.layer3_rn` | 1 | 2.359M |
| `model.encoder.point_head.scratch.layer4_rn` | 1 | 2.359M |
| `model.encoder.depth_head.scratch.layer3_rn` | 1 | 2.359M |
| `model.encoder.depth_head.scratch.layer4_rn` | 1 | 2.359M |
| `model.encoder.point_head.projects.2` | 2 | 2.098M |
| `model.encoder.point_head.projects.3` | 2 | 2.098M |
| `model.encoder.depth_head.projects.2` | 2 | 2.098M |
| `model.encoder.depth_head.projects.3` | 2 | 2.098M |
| `model.encoder.part_adaptor.resize_layers.3` | 13 | 1.903M |
| `model.encoder.part_adaptor.resize_layers.1` | 13 | 1.575M |
| `model.encoder.part_adaptor.resize_layers.2` | 11 | 1.313M |
| `model.encoder.point_head.scratch.refinenet4` | 6 | 1.246M |
| `model.encoder.depth_head.scratch.refinenet4` | 6 | 1.246M |
| `model.encoder.part_head.refinenet1.resConfUnit1` | 4 | 1.180M |
| `model.encoder.part_head.refinenet1.resConfUnit2` | 4 | 1.180M |
| `model.encoder.part_head.refinenet2.resConfUnit1` | 4 | 1.180M |
| `model.encoder.part_head.refinenet2.resConfUnit2` | 4 | 1.180M |
| `model.encoder.part_head.refinenet3.resConfUnit1` | 4 | 1.180M |
| `model.encoder.part_head.refinenet3.resConfUnit2` | 4 | 1.180M |
| `model.encoder.part_head.refinenet4.resConfUnit2` | 4 | 1.180M |
| `model.encoder.point_head.scratch.layer2_rn` | 1 | 1.180M |
| `model.encoder.depth_head.scratch.layer2_rn` | 1 | 1.180M |
| `model.encoder.point_head.projects.1` | 2 | 1.049M |
| `model.encoder.point_head.resize_layers.1` | 2 | 1.049M |
| `model.encoder.depth_head.projects.1` | 2 | 1.049M |
| `model.encoder.depth_head.resize_layers.1` | 2 | 1.049M |
| `model.encoder.point_head.resize_layers.0` | 2 | 1.049M |
| `model.encoder.depth_head.resize_layers.0` | 2 | 1.049M |
| `model.encoder.part_head.window_cross_attention.atten_block` | 17 | 0.791M |
| `model.encoder.part_head.window_cross_attention.conv_after_body` | 2 | 0.590M |
| `model.encoder.point_head.scratch.layer1_rn` | 1 | 0.590M |
| `model.encoder.depth_head.scratch.layer1_rn` | 1 | 0.590M |
| `model.encoder.part_head.layer1_rn.weight` | 1 | 0.590M |
| `model.encoder.part_head.layer2_rn.weight` | 1 | 0.590M |
| `model.encoder.part_head.layer3_rn.weight` | 1 | 0.590M |
| `model.encoder.part_head.layer4_rn.weight` | 1 | 0.590M |
| `model.encoder.point_head.projects.0` | 2 | 0.525M |
| `model.encoder.depth_head.projects.0` | 2 | 0.525M |
| `model.encoder.part_adaptor.projects.0` | 2 | 0.525M |
| `model.encoder.part_adaptor.projects.1` | 2 | 0.525M |
| `model.encoder.part_adaptor.projects.2` | 2 | 0.525M |
| `model.encoder.part_adaptor.projects.3` | 2 | 0.525M |
| `model.encoder.part_head.window_self_atten.atten_block` | 20 | 0.296M |
| `model.encoder.point_head.scratch.output_conv1` | 2 | 0.295M |
| `model.encoder.depth_head.scratch.output_conv1` | 2 | 0.295M |
| `model.encoder.part_head.output_conv1.weight` | 1 | 0.295M |
| `model.encoder.part_head.window_cross_attention.conv_last` | 2 | 0.148M |
| `model.encoder.part_head.window_self_atten.conv_after_body` | 2 | 0.148M |
| `model.encoder.part_head.window_cross_attention.conv_before_upsample` | 2 | 0.148M |
| `model.encoder.part_head.window_self_atten.conv_last` | 2 | 0.074M |
| `model.encoder.part_head.window_self_atten.conv_before_upsample` | 2 | 0.074M |
| `model.encoder.part_head.refinenet1.out_conv` | 2 | 0.066M |
| `model.encoder.part_head.refinenet2.out_conv` | 2 | 0.066M |
| `model.encoder.part_head.refinenet3.out_conv` | 2 | 0.066M |
| `model.encoder.part_head.refinenet4.out_conv` | 2 | 0.066M |
| `model.encoder.part_head.cross_attention_1.projq` | 2 | 0.066M |
| `model.encoder.part_head.cross_attention_1.projk` | 2 | 0.066M |
| `model.encoder.part_head.cross_attention_1.projv` | 2 | 0.066M |
| `model.encoder.part_head.cross_attention_1.proj` | 2 | 0.066M |
| `model.encoder.part_head.cross_attention_2.projq` | 2 | 0.066M |
| `model.encoder.part_head.cross_attention_2.projk` | 2 | 0.066M |
| `model.encoder.part_head.cross_attention_2.projv` | 2 | 0.066M |
| `model.encoder.part_head.cross_attention_2.proj` | 2 | 0.066M |
| `model.encoder.point_head.scratch.output_conv2` | 4 | 0.037M |
| `model.encoder.depth_head.scratch.output_conv2` | 4 | 0.037M |
| `model.encoder.part_head.output_conv2.0` | 2 | 0.037M |
| `model.encoder.point_head.norm.weight` | 1 | 0.002M |
| `model.encoder.point_head.norm.bias` | 1 | 0.002M |
| `model.encoder.depth_head.norm.weight` | 1 | 0.002M |
| `model.encoder.depth_head.norm.bias` | 1 | 0.002M |
| `model.encoder.part_adaptor.norm.weight` | 1 | 0.002M |
| `model.encoder.part_adaptor.norm.bias` | 1 | 0.002M |
| `model.encoder.part_head.window_cross_attention.patch_embed` | 2 | 0.001M |
| `model.encoder.part_head.window_cross_attention.norm` | 2 | 0.001M |
| `model.encoder.part_head.output_conv2.2` | 2 | 0.000M |
| `model.encoder.part_head.window_self_atten.patch_embed` | 2 | 0.000M |
| `model.encoder.part_head.window_self_atten.norm` | 2 | 0.000M |
| `model.encoder.part_head.output_conv1.bias` | 1 | 0.000M |
| **合计** | | **90.936M** |

**(3) dtype**

- 可训集合：float32 90.936M
- 全模型：bfloat16 909.112M, float32 307.111M

**(4) param_groups 归属与实际 lr**（`param_groups`）

| 组 | keywords | lr_multiplier | 实际 lr | 张量 | 参数量 | dtype |
|---|---|---:|---:|---:|---:|---|
| | `part_adaptor`, `part_head` | 1.0 | 3.00e-05 | 192 | 25.627M | float32 25.627M |
| | `point_head`, `depth_head` | 0.1 | 3.00e-06 | 124 | 65.309M | float32 65.309M |
| | `<default>` | 0.1 | 3.00e-06 | 0 | 0.000M | — |

### phys_iggt

- wrapper `IGGTWrapper` · base lr `3e-05` · 总参数 1230.238M · **可训 79.324M** (6.4%)
- 构造期就已冻结（`setup()` 之前）：**1** 个参数张量

**(1) freeze_keywords 命中**

| keyword | 命中张量 | 命中参数量 |
|---|---:|---:|
| `aggregator` | 1210 | 909.112M |
| `camera_head` | 69 | 216.175M |
| `part_adaptor` | 73 | 11.615M |
| `part_head` | 119 | 14.012M |

**(2) 实际 requires_grad=True 的集合（按模块 rollup）**

| 模块 | 张量 | 参数量 |
|---|---:|---:|
| `model.encoder.point_head.resize_layers.3` | 2 | 9.438M |
| `model.encoder.depth_head.resize_layers.3` | 2 | 9.438M |
| `model.encoder.point_head.scratch.refinenet1` | 10 | 2.426M |
| `model.encoder.point_head.scratch.refinenet2` | 10 | 2.426M |
| `model.encoder.point_head.scratch.refinenet3` | 10 | 2.426M |
| `model.encoder.depth_head.scratch.refinenet1` | 10 | 2.426M |
| `model.encoder.depth_head.scratch.refinenet2` | 10 | 2.426M |
| `model.encoder.depth_head.scratch.refinenet3` | 10 | 2.426M |
| `model.encoder.physics_scheme.physics_head.refinenet1` | 10 | 2.426M |
| `model.encoder.physics_scheme.physics_head.refinenet2` | 10 | 2.426M |
| `model.encoder.physics_scheme.physics_head.refinenet3` | 10 | 2.426M |
| `model.encoder.point_head.scratch.layer3_rn` | 1 | 2.359M |
| `model.encoder.point_head.scratch.layer4_rn` | 1 | 2.359M |
| `model.encoder.depth_head.scratch.layer3_rn` | 1 | 2.359M |
| `model.encoder.depth_head.scratch.layer4_rn` | 1 | 2.359M |
| `model.encoder.point_head.projects.2` | 2 | 2.098M |
| `model.encoder.point_head.projects.3` | 2 | 2.098M |
| `model.encoder.depth_head.projects.2` | 2 | 2.098M |
| `model.encoder.depth_head.projects.3` | 2 | 2.098M |
| `model.encoder.physics_scheme.physics_head.window_cross_attention` | 27 | 1.678M |
| `model.encoder.point_head.scratch.refinenet4` | 6 | 1.246M |
| `model.encoder.depth_head.scratch.refinenet4` | 6 | 1.246M |
| `model.encoder.physics_scheme.physics_head.refinenet4` | 6 | 1.246M |
| `model.encoder.point_head.scratch.layer2_rn` | 1 | 1.180M |
| `model.encoder.depth_head.scratch.layer2_rn` | 1 | 1.180M |
| `model.encoder.point_head.projects.1` | 2 | 1.049M |
| `model.encoder.point_head.resize_layers.1` | 2 | 1.049M |
| `model.encoder.depth_head.projects.1` | 2 | 1.049M |
| `model.encoder.depth_head.resize_layers.1` | 2 | 1.049M |
| `model.encoder.point_head.resize_layers.0` | 2 | 1.049M |
| `model.encoder.depth_head.resize_layers.0` | 2 | 1.049M |
| `model.encoder.physics_scheme.physics_head.window_self_atten` | 30 | 0.592M |
| `model.encoder.point_head.scratch.layer1_rn` | 1 | 0.590M |
| `model.encoder.depth_head.scratch.layer1_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.layer1_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.layer2_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.layer3_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.layer4_rn` | 1 | 0.590M |
| `model.encoder.point_head.projects.0` | 2 | 0.525M |
| `model.encoder.depth_head.projects.0` | 2 | 0.525M |
| `model.encoder.point_head.scratch.output_conv1` | 2 | 0.295M |
| `model.encoder.depth_head.scratch.output_conv1` | 2 | 0.295M |
| `model.encoder.physics_scheme.physics_head.output_conv1` | 2 | 0.295M |
| `model.encoder.physics_scheme.physics_head.cross_attention_1` | 8 | 0.263M |
| `model.encoder.physics_scheme.physics_head.cross_attention_2` | 8 | 0.263M |
| `model.encoder.physics_scheme.physics_head.output_conv2` | 4 | 0.038M |
| `model.encoder.point_head.scratch.output_conv2` | 4 | 0.037M |
| `model.encoder.depth_head.scratch.output_conv2` | 4 | 0.037M |
| `model.encoder.physics_scheme.physics_classifier.mlp` | 4 | 0.002M |
| `model.encoder.point_head.norm.weight` | 1 | 0.002M |
| `model.encoder.point_head.norm.bias` | 1 | 0.002M |
| `model.encoder.depth_head.norm.weight` | 1 | 0.002M |
| `model.encoder.depth_head.norm.bias` | 1 | 0.002M |
| **合计** | | **79.324M** |

**(3) dtype**

- 可训集合：float32 79.324M
- 全模型：bfloat16 909.112M, float32 321.125M

**(4) param_groups 归属与实际 lr**（`param_groups`）

| 组 | keywords | lr_multiplier | 实际 lr | 张量 | 参数量 | dtype |
|---|---|---:|---:|---:|---:|---|
| | `physics_scheme` | 1.0 | 3.00e-05 | 123 | 14.015M | float32 14.015M |
| | `point_head`, `depth_head` | 0.1 | 3.00e-06 | 124 | 65.309M | float32 65.309M |
| | `<default>` | 0.1 | 3.00e-06 | 0 | 0.000M | — |

### phys_prop_iggt

- wrapper `IGGTWrapper` · base lr `3e-05` · 总参数 1230.238M · **可训 14.015M** (1.1%)
- 构造期就已冻结（`setup()` 之前）：**1** 个参数张量

**(1) freeze_keywords 命中**

| keyword | 命中张量 | 命中参数量 |
|---|---:|---:|
| `aggregator` | 1210 | 909.112M |
| `camera_head` | 69 | 216.175M |
| `point_head` | 62 | 32.655M |
| `depth_head` | 62 | 32.655M |
| `part_adaptor` | 73 | 11.615M |
| `part_head` | 119 | 14.012M |

**(2) 实际 requires_grad=True 的集合（按模块 rollup）**

| 模块 | 张量 | 参数量 |
|---|---:|---:|
| `model.encoder.physics_scheme.physics_head.refinenet1` | 10 | 2.426M |
| `model.encoder.physics_scheme.physics_head.refinenet2` | 10 | 2.426M |
| `model.encoder.physics_scheme.physics_head.refinenet3` | 10 | 2.426M |
| `model.encoder.physics_scheme.physics_head.window_cross_attention` | 27 | 1.678M |
| `model.encoder.physics_scheme.physics_head.refinenet4` | 6 | 1.246M |
| `model.encoder.physics_scheme.physics_head.window_self_atten` | 30 | 0.592M |
| `model.encoder.physics_scheme.physics_head.layer1_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.layer2_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.layer3_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.layer4_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.output_conv1` | 2 | 0.295M |
| `model.encoder.physics_scheme.physics_head.cross_attention_1` | 8 | 0.263M |
| `model.encoder.physics_scheme.physics_head.cross_attention_2` | 8 | 0.263M |
| `model.encoder.physics_scheme.physics_head.output_conv2` | 4 | 0.038M |
| `model.encoder.physics_scheme.physics_property_readout.mlp` | 4 | 0.003M |
| **合计** | | **14.015M** |

**(3) dtype**

- 可训集合：float32 14.015M
- 全模型：bfloat16 909.112M, float32 321.126M

**(4) param_groups 归属与实际 lr**（`param_groups`）

| 组 | keywords | lr_multiplier | 实际 lr | 张量 | 参数量 | dtype |
|---|---|---:|---:|---:|---:|---|
| | `physics_scheme` | 1.0 | 3.00e-05 | 123 | 14.015M | float32 14.015M |
| | `<default>` | 0.1 | 3.00e-06 | 0 | 0.000M | — |

### physgm_iggt

- wrapper `IGGTWrapper` · base lr `3e-05` · 总参数 1216.623M · **可训 0.400M** (0.0%)
- 构造期就已冻结（`setup()` 之前）：**1** 个参数张量

**(1) freeze_keywords 命中**

| keyword | 命中张量 | 命中参数量 |
|---|---:|---:|
| `aggregator` | 1210 | 909.112M |
| `camera_head` | 69 | 216.175M |
| `point_head` | 62 | 32.655M |
| `depth_head` | 62 | 32.655M |
| `part_adaptor` | 73 | 11.615M |
| `part_head` | 119 | 14.012M |

**(2) 实际 requires_grad=True 的集合（按模块 rollup）**

| 模块 | 张量 | 参数量 |
|---|---:|---:|
| `model.encoder.physics_scheme.physgm_readout.decoders` | 15 | 0.400M |
| **合计** | | **0.400M** |

**(3) dtype**

- 可训集合：float32 0.400M
- 全模型：bfloat16 909.112M, float32 307.510M

**(4) param_groups 归属与实际 lr**（`param_groups`）

| 组 | keywords | lr_multiplier | 实际 lr | 张量 | 参数量 | dtype |
|---|---|---:|---:|---:|---:|---|
| | `physics_scheme` | 1.0 | 3.00e-05 | 15 | 0.400M | float32 0.400M |
| | `<default>` | 0.1 | 3.00e-06 | 0 | 0.000M | — |

### physgm_dpt_iggt

- wrapper `IGGTWrapper` · base lr `0.0002` · 总参数 1230.242M · **可训 14.019M** (1.1%)
- 构造期就已冻结（`setup()` 之前）：**1** 个参数张量

**(1) freeze_keywords 命中**

| keyword | 命中张量 | 命中参数量 |
|---|---:|---:|
| `aggregator` | 1210 | 909.112M |
| `camera_head` | 69 | 216.175M |
| `point_head` | 62 | 32.655M |
| `depth_head` | 62 | 32.655M |
| `part_adaptor` | 73 | 11.615M |
| `part_head` | 119 | 14.012M |

**(2) 实际 requires_grad=True 的集合（按模块 rollup）**

| 模块 | 张量 | 参数量 |
|---|---:|---:|
| `model.encoder.physics_scheme.physics_head.refinenet1` | 10 | 2.426M |
| `model.encoder.physics_scheme.physics_head.refinenet2` | 10 | 2.426M |
| `model.encoder.physics_scheme.physics_head.refinenet3` | 10 | 2.426M |
| `model.encoder.physics_scheme.physics_head.window_cross_attention` | 27 | 1.678M |
| `model.encoder.physics_scheme.physics_head.refinenet4` | 6 | 1.246M |
| `model.encoder.physics_scheme.physics_head.window_self_atten` | 30 | 0.592M |
| `model.encoder.physics_scheme.physics_head.layer1_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.layer2_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.layer3_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.layer4_rn` | 1 | 0.590M |
| `model.encoder.physics_scheme.physics_head.output_conv1` | 2 | 0.295M |
| `model.encoder.physics_scheme.physics_head.cross_attention_1` | 8 | 0.263M |
| `model.encoder.physics_scheme.physics_head.cross_attention_2` | 8 | 0.263M |
| `model.encoder.physics_scheme.physics_head.output_conv2` | 4 | 0.038M |
| `model.encoder.physics_scheme.physgm_dense_readout.decoders` | 15 | 0.007M |
| **合计** | | **14.019M** |

**(3) dtype**

- 可训集合：float32 14.019M
- 全模型：bfloat16 909.112M, float32 321.130M

**(4) param_groups 归属与实际 lr**（`param_groups`）

| 组 | keywords | lr_multiplier | 实际 lr | 张量 | 参数量 | dtype |
|---|---|---:|---:|---:|---:|---|
| | `physics_scheme` | 1.0 | 2.00e-04 | 134 | 14.019M | float32 14.019M |
| | `<default>` | 0.1 | 2.00e-05 | 0 | 0.000M | — |

