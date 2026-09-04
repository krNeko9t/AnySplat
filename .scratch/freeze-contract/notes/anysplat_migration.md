# 08 号票实测报告：`freeze_backbone`/`freeze_module` → `freeze_keywords` 逐参数等价

日期 2026-09-04 · 分支 `fix_freeze` · `paper_repo` env / CPU / `--random-backbone`
原始记录：[`raw/08-pre/`](raw/08-pre/)（HEAD）与 [`raw/08-post/`](raw/08-post/)（迁移后），各 6 份。

## 结论

**6 份 anysplat 配方全部逐参数等价。** 判据（08 号票「完成判据（硬）」）四项逐个比对，
零差异：

- 每个参数的 `requires_grad` 逐个相等（不是总数相等）
- 每个参数的 `dtype` 逐个相等
- 每个参数的 `numel` 与参数**名字集合**相等
- `param_groups` 归属（关键词、`lr_multiplier`、参数数、numel）与 `lr` 不变

| config | 参数总数 | 冻结数 | 迁移后 `freeze_keywords` | 关键词命中 | 判定 |
|---|---|---|---|---|---|
| `dl3dv` | 2746 | 1685 | `[patch_embed]` | 688 | 等价 |
| `co3d` | 2746 | 1685 | `[patch_embed]` | 688 | 等价 |
| `scannetpp` | 2746 | 1685 | `[patch_embed]` | 688 | 等价 |
| `multi-dataset` | 2746 | 1685 | `[patch_embed]` | 688 | 等价 |
| `instseg_anysplat` | 1659 | 1341 | `[aggregator, camera_head, depth_head]` | 1210 / 69 / 62 | 等价 |
| `instseg_small` | 1659 | 1 | **不设** | — | 等价 |

可训参数量：四份上游 886.4M / 2348.7M，`instseg_anysplat` 91.1M / 1249.0M，
`instseg_small` 1249.0M / 1249.0M（全可训）。

## 为什么等价——冻结集的构成

**四份上游**（`freeze_module: patch_embed` + `distill: true`）。旧分支走的是**无条件赋值**
（`param.requires_grad = (freeze_module not in name and "distill" not in name)`），
冻结集 1685 = `distill_*` 的 1341 ∪ 非 distill 的 `patch_embed` 344；两者交集为空、并集即全部，
**没有第三类**。迁移后这 1685 由两个独立来源产生，都不经过被删的分支：

- 1341 个 `distill_*`：`__init__` 里的 distill 块自己冻的（`anysplat.py` 的
  `if self.distill:`，冻 + 搬 CPU），本票原样保留。
- 688 个命中 `patch_embed` 的：`setup()` 冻的。688 = 344（`aggregator.patch_embed.*`）
  + 344（`distill_aggregator.patch_embed.*`，本已冻，只冻不解冻 ⇒ 无变化）。

无条件赋值那半边（把未命中的一律置 `True`）在这四份上零影响，实测确认：
迁移前后**没有任何参数**的 `requires_grad` 从 `False` 翻成 `True` 或反向。

**`instseg_anysplat`**（`freeze_backbone: true`，`pred_head_type: depth`）。冻结集 1341
恰好是三个关键词的并集（1210 + 69 + 62），且**可训集合里没有任何参数的名字包含这三个词**
（实测，不是推理）。`point_head` 虽因 `instance_feat_dim=8` + `depth` 路径而存在
（`_use_point_head_for_part`），旧 `freeze_backbone` 分支不冻它，三个关键词也不命中它 ⇒ 仍可训。

**`instseg_small`**：两个字段都没设（旧默认 `freeze_module: "None"`，零行为）⇒ 不设
`freeze_keywords`，仍是零行为。

## 08 号票预警的两个坑，实测结果

- **`patch_embed` 裸子串多命中**（01 号票 F2 记录它在 IGGT 上多命中 4 个
  `part_head.window_{self,cross}_atten`）：AnySplat 侧**未发生**。688 个命中全部落在
  `aggregator.patch_embed.*` 与 `distill_aggregator.patch_embed.*` 两棵子树内，
  两份带 instance head 的配方（`instseg_*`）也一样——它们本就不设这个关键词。
- **`camera_head`/`depth_head` 被 `distill_camera_head`/`distill_depth_head` 多命中**：
  `instseg_anysplat` 是 `distill: false`，无这些模块，**未发生**。若将来有配方同时开
  `distill: true` 与这组关键词，多命中的目标本就已被 distill 块冻死 ⇒ 只冻不解冻，仍无行为差。
- **零命中 `raise`**：六份都没触发，说明五份的翻译各自至少命中一个参数。

## 顺带记下的两条事实（给 09 / 10 用）

1. **AnySplat 路线的构造期冻结不变量只有一个参数**：`aggregator.patch_embed.mask_token`
   （六份配方一致，迁移前后都在）。它同时被 `patch_embed` 关键词命中，但因「只冻不解冻」而无差别。
   这是 02 号票立的「构造期冻结不变量」在 AnySplat 侧的全部实体。
2. **探针补了第二条 HF 下载路径**：`pretrained_weights: "hf:..."` 走
   `src/model/arch/weight_loading.py` 的 `init_anysplat_from_hf`，它是
   `AnySplat(encoder_cfg, decoder_cfg)` + `load_state_dict`——又一份约 5GB 下载，只改参数**值**。
   `scripts_probe.py` 的 `_skip_backbone_download()` 现在把它一并短路（与 VGGT 那条同理），
   否则 `instseg_*` 两份在 `--random-backbone` 下依然会联网、实测卡了 22 分钟。
   09 号票的生成端要复用这条探针路径，会撞上同一件事。
