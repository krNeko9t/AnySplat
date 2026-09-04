# 09 — 落地指纹 + lock 层

Type: task
Status: closed (2026-09-04)
Blocked by: 06 (closed)
Assignee: krNeko9t (session ac2eebb8)

## Question

03（指纹正文形态与字段）、04（存放位置、生成流程、强制）、06（失败行为与逃生门）定完后的实现票。

### 要做的

1. **唯一的 capture 函数**（04 D3 的硬约束）：输入一个已 `setup()` 完的 `LightningModule`，
   输出 03 定义的定宽文本（头部 `experiment` / `arch` / `freeze_keywords` 原文 / `base_lr` /
   `structure: sha256:...`，正文一行一参数 `T|- dtype numel name`，按 name 排序）。
   生成端与校验端都调它。
2. **生成端** `scripts/freeze_lock.py +experiment=<X>`：复用 01 号票探针路径
   （hydra compose → `load_typed_root_config` → 建模 → `setup()`），`--random-backbone`，纯 CPU。
   写 `config/experiment/locks/<X>.lock`（路径 2026-09-04 修订，见 04 D1）。
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

---

## 本 session 的两条输入（2026-09-04）

**1. lock 路径修订**：`config/experiment/locks/<X>.lock`（详见 04 D1 的修订框）。
本票第 2 条的写出路径与第 3 条的读入路径都按此。

**2. 批量生成必须是「每份一个子进程，串行」**（**新增硬判据**）。

起因是现场事故：用户试跑了一次 CPU 建模路径，**把整机干崩**（非 OOM kill——
`journalctl -k` 无记录，是 swap 抖死后失去响应）。机制：探针在 fp32 下构造 VGGT-1B
级模型 ≈ 5GB/份，**一个进程内连着建多份而不释放**，本机 62G 内存 + 7G swap 撑不到 22 份。

因此：`scripts/freeze_lock.py` 的**单份生成是原子操作**，批量由外层 driver 起 22 次子进程、
串行。峰值内存 = 单份；进程退出即彻底归还；某一份建不起来不拖垮其余 21 份——那正是本票
「已知风险」里 13 份未验配方的场景。**明确拒绝并行子进程**（峰值内存乘并发数）。


---

## 解决（2026-09-04）

**落地并提交（`b53f7dd`）。22 份 lock 全覆盖，四条完成判据逐条实测通过。**

### 交付物

| 文件 | 角色 |
|---|---|
| `src/freeze_contract.py` | 唯一的 capture 函数 + lock 路径 + 身份键 + `verify()` |
| `scripts/freeze_lock.py` | 生成端（纯 CPU、离线、`--all` 串行子进程） |
| `src/model/wrapper/base_wrapper.py` | `apply_freeze()` 拆出；`setup()` 末尾校验 |
| `config/experiment/locks/*.lock` | 22 份，44877 行，3.6M |

### 完成判据的实测读数

1. **22 份全生成，幂等**：22/22 成功——**包括本票「已知风险」里那 13 份未验配方，无一建不起来**；
   再跑一次全部 `unchanged`，diff 零行。
2. **与 01 号票逐参数一致**：九份全 `OK`（`instseg_iggt` 1595 参数 / 可训 316，`physgm_iggt`
   1610/15，`segvggt_physgm` 2621/15，`segvggt_scannet` 2606/2070 …）。顺带把 08 号票的六份
   AnySplat 记录也比了，同样全 `OK`（`co3d`/`dl3dv`/`multi-dataset`/`scannetpp` 各 2746/1061，
   `instseg_anysplat` 1659/318，`instseg_small` 1659/1658）——**15 份有历史实测的配方零偏差**。
3. **反做 `5f1eff7`**（从 `segvggt_physgm` 删掉 `camera_token` + `register_token`）：启动
   **硬错**，报错正文里的差异恰好是那两行 `-` → `T`；重生成后 lock 的**正文 diff 恰好两行**，
   且头部 `freeze_keywords` 那行就在正上方一起变——03 号票要的「改了哪个 keyword」与
   「因此哪些参数动了」上下并排，在真实 diff 里成立。
4. **删掉 lock**：启动硬错（`FileNotFoundError`），报错点名 experiment / lock 路径 / 重生成命令。

另外实测：`stage="test"` 在缺 lock 下**放行**（06 D4 的唯一一道门）；`instseg_small`
（`freeze_keywords` 为空）能走到校验点并通过——**04 D2 的 22 份全覆盖在删掉
`if not freeze_kw: return` 后确实成立**；成功路径**连 output 目录都没创建**。

### 三处偏离设计、且比设计更好的地方

**1. 校验比的是 lock 正文逐行，不是 lock 自己声明的哈希。**

原打算「实测哈希 vs lock 头部的 `structure:` 哈希」。写完第一版测试当场发现：手改 lock 正文
而不动哈希行，校验**完全放行**。那意味着人在 `git diff` 里签字的那份文本，和被执法的那份东西
可以不是同一个——正是 02/03 号票反复警告的「护栏在守自己的影子」，只是这次跑在 lock 文件内部。
改为 `lock_body()` 逐行比对，哈希降级为**每次从正文重算**的人类摘要，永不作为可信输入。

**2. 冻结实现拆出 `BaseWrapper.apply_freeze()`。**

生成端原本调 `wrapper.setup("fit")`，而 `setup()` 现在会校验 ⇒ 生成一份 lock 需要先有那份
lock。拆成 `apply_freeze()`（施加冻结，是 `requires_grad` 终态的唯一产生点）+ `setup()`
（`apply_freeze()` 然后校验）。**冻结实现仍只有一份，两个调用者。**

**3. 免权重的手段从「打三个补丁」收成「config 层清空 `pretrained_weights`」。**

现场撞到第三条读权重路径：`SegVGGTModel.from_checkpoint` / `IGGTModel.from_checkpoint`
读的是 `pretrained_weights` 指向的**集群本地 ckpt**（`segvggt_physgm` 指向
`output/exp_segvggt_finetune_agnostic/.../last.ckpt`，本机不存在）。08 号票发现的 `hf:` 那条
也走同一个字段。于是不逐条打补丁，直接在 compose 时追加 `model.encoder.pretrained_weights=`：
`get_model()` 三个分支同时落到裸构造器。**只剩 VGGT-1B backbone 那一个补丁**——它在 arch
构造器内部，够不到 config。

### 现场事实（可复核）

- **单份生成峰值 RSS 7.3G**（`segvggt_physgm`，fp32），耗时约 13 秒。这就是用户那次崩机的
  机制：一个进程内连建多份，62G 内存 + 7G swap 撑不到 22 份，swap 抖死后整机失去响应
  （`journalctl -k` 无 OOM 记录，因为不是 OOM kill）。**`--all` 因此是串行子进程，不是循环。**
- 全量参数规模：IGGT 1595–1729，SegVGGT 2606–2621，AnySplat 1659–2746。
- `config/experiment/` 条目数：**23**（22 yaml + `locks/`），而非原方案的 44。

### 顺带修掉的一处错注释

`base_wrapper.py:73` 原文「Single source of truth for freezing **when non-empty**: params
matching any keyword are frozen, **all others unfrozen**」——后半句与 02 号票确立的「只冻不解冻」
直接相反，前半句的 "when non-empty" 也是 `freeze_module` 时代的残留。已改。**这是 10 号票
「长注释归位」的一个样本**：错的不只是位置。

### 对下游的约束

- → [10 号票](issues/10-converge-docs-and-comments.md) **解除阻塞**，且拿到三样新输入：
  (a) 术语表要收「**apply_freeze 是 requires_grad 终态的唯一产生点**」；
  (b) 04 D4 说的「lock 生成与更新流程」现在有实际命令与实际读数可写；
  (c) `base_wrapper.py` 的长注释里**至少有一条是内容错误而非位置问题**，10 号票不能只搬不读。
- lock 路径修订已同步进 04 与 06 两张票。
