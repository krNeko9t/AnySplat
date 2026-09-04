# 09 — 落地指纹 + lock 层

Type: task
Status: open
Blocked by: 06

## Question

03（指纹正文形态与字段）、04（存放位置、生成流程、强制）、06（失败行为与逃生门）定完后的实现票。

### 要做的

1. **唯一的 capture 函数**（04 D3 的硬约束）：输入一个已 `setup()` 完的 `LightningModule`，
   输出 03 定义的定宽文本（头部 `experiment` / `arch` / `freeze_keywords` 原文 / `base_lr` /
   `structure: sha256:...`，正文一行一参数 `T|- dtype numel name`，按 name 排序）。
   生成端与校验端都调它。
2. **生成端** `scripts/freeze_lock.py +experiment=<X>`：复用 01 号票探针路径
   （hydra compose → `load_typed_root_config` → 建模 → `setup()`），`--random-backbone`，纯 CPU。
   写 `config/experiment/<X>.freeze.lock`。
3. **校验端**：`BaseWrapper.setup()` 末尾（`base_wrapper.py:501-532`，strategy wrap 之前）
   重算一份、与 lock 逐行比。**每个 rank 各自算各自比。零个 if。**
4. **批量生成 22 份初始 lock** 并提交。

### 完成判据（硬）

- 22 份 experiment config 各有一份 lock，且**再跑一次生成脚本 diff 为空**（幂等）。
- 九份有 `freeze_keywords` 的 lock 与 01 号票的实测记录（`notes/raw/<experiment>.json`
  的 `params` 字段）**逐参数一致**——不是"看着对"。
- 人为改一份配方的 `freeze_keywords`（如从 `segvggt_physgm` 删掉 `camera_token`），
  确认启动硬错，且 lock 重生成后的 diff **恰好是两行**（`5f1eff7` 的形状，见 03 号票的案例表）。
- 删掉某份 lock，确认启动硬错（04 D2）。

### 已知风险

01 号票只证明了**九份**能在 CPU / 无数据集 / 无 HF 权重下建模。剩余 13 份未验。
若某份建不起来 ⇒ 按 04 D2 它本来就跑不起来，属那份配方自身的问题，不是本票放宽覆盖面的理由。

---

## 05 号票的输入（2026-09-04）

跨 stage 校验已整票划出 scope。**本票不需要为它预留任何接口**：03 定的指纹正文已经是
逐参数名，两份 lock 求交本就够用。将来真要建，它是一个**读两份 lock 的独立脚本**，
不回改指纹格式、不改本票的 capture 函数。

---

## 06 号票的收窄（2026-09-04）

- **校验端的门**：唯一条件是 `stage == "fit"`（06 D4）。`fast_dev_run` / sanity check 不豁免（06 D3）。
- **必须删掉 `base_wrapper.py:507-508` 的 `if not freeze_kw: return`**——否则 13 份不设
  `freeze_keywords` 的配方走不到校验点，04 D2 的「22 份全覆盖」当场失效。
- **硬错三条**：结构哈希与 lock 不符 / lock 文件缺失 / `freeze_keywords` 零命中（已有）。无 warn 档。
- **失败路径**：把实测到的指纹全文写进 run 目录（`cfg.train.output_path`，`src/main.py:66`），
  异常正文点名四样——experiment 名 / lock 路径 / 重生成命令原文 / 头几行差异。成功时不落任何文件。
- **生成端**：不设 `--yes`、不做交互确认，但无条件把新旧差异打到 stdout（06 D1）。
