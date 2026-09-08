# AnySplat / SegVGGT 训练语境

本仓库是多个上游仓库的缝合体（AnySplat 框架 + VGGT/SegVGGT backbone + IGGT instance head +
自研 physics 头），同一件事在不同上游里有不同叫法。本文件只定义**词**：它是什么，不是它怎么实现。
机制现状见 [`docs/repo_knowledge.md`](docs/repo_knowledge.md)，分层契约见
[`docs/layered_scheme.md`](docs/layered_scheme.md)。

## Language

### 冻结

**冻结**：
一个参数同时满足三条判据的状态——不建计算图、不进优化器、权重不变。三条缺一条就不是冻结，
只是"看起来没在动"。
_Avoid_：不训练、锁住、freeze 掉（口语里常指其中任意一条）

**冻结入口**：
config 能拨动的冻结开关。全仓**只有一个**：`optimizer.freeze_keywords`。"入口唯一"是硬约束，
不是现状描述——第二个入口意味着"这个参数在不在训"有两个答案。
_Avoid_：冻结配置、freeze API

**构造期冻结不变量**：
模块建自己时就冻掉的参数（LoRA 冻它包裹的 attention 基座、`patch_embed.mask_token`）。
config 拨不动它，冻结入口也够不着它——模型建完时它早已成立。它**不是**第二个入口，
是模型自身结构的一部分：保留、记录，不迁移。
_Avoid_：硬编码冻结、隐式冻结

**指纹**（trainable-parameter fingerprint）：
一次 run 的**全量**参数结构快照——一行一参数（`requires_grad` / `dtype` / `numel` / `name`），
按 name 排序。全量而非可训子集：在子集里，"被冻了"和"已经不存在了"长得一模一样。
_Avoid_：参数清单、可训集合快照

**lock**：
提交进 git 的那份指纹，一份配方一份，放在 `config/experiment/locks/`。它是配方对
"本次 run 该训什么"的**声明**；启动时实测指纹与它逐行比对，不符即拒训。
被执法的是 lock 的正文，不是 lock 自己声明的哈希。
_Avoid_：基线、快照文件、freeze snapshot

**bf16 死参数不是冻结**：
参数被永久 `.to(torch.bfloat16)`（`src/model/arch/anysplat.py:113`、`src/model/arch/iggt.py:80`）后，
AdamW 的动量状态也是 bf16，量级 ~lr 的更新被舍入成 0——权重不变，但 `requires_grad` 全程为真、
计算图照建、优化器照进。它只占三条判据里的第三条，机制在**训练精度**那一层，不在冻结这一层。
指纹带 `dtype` 一列只为可见性（改了哈希会撞），不为它设任何断言。
_Avoid_：静默冻结（这个比喻正是把它误当冻结的原因）

**权重落位断言 ≠ 指纹**：
`_assert_query_physgm_coverage`（`src/model/arch/segvggt.py:297`）查的是"ckpt 里的
`query_physgm` 权重有没有全部加载成功"——参数的**值**是否落位，冻结三判据一条都不占。
它与指纹层各管各的，不合并。
_Avoid_：把它算作"另一种冻结校验"

### 物性读出

**物性读出**（physics readout）：
把 backbone 的特征变成 (E, ν, ρ) 的那一小段网络。本仓库里它有**三种粒度**，
说"物性头"而不说粒度，基本一定会指错东西：

- **场景级**：池化整帧 aggregator token → 一帧一个 (mu, var)。`phys_scheme="physgm_copy"`
  （`src/model/heads/physics/physgm_readout.py`）。**给不出"哪个物体多重"**。
- **实例级（掩码池化）**：dense feature map 按 instance mask 掩码平均 → 每实例一个 (mu, var)。
  `phys_scheme="physgm_dpt"`（`physgm_dense_readout.py`）。
  **池化严格在 per-property MLP 之前**——这一条是可替换池化域（2D 像素 / 3D 高斯）的全部依据。
- **实例级（query 自带）**：query 向量直接过 MLP，实例身份由 query 自己携带，不需要 mask。
  `QueryPhysGMReadout`（`src/model/arch/segvggt.py:183`）。
_Avoid_：物性头、phys head（都不带粒度）

**GT-mask 依赖**：
一个物性读出在**推理时**是否需要外部给的实例掩码。掩码池化那一档需要，query 那一档不需要。
它不是实现细节而是路线属性——需要它，就必须回答"推理时掩码从哪来"，
而这正是 IGGT 路线（实例由 HDBSCAN 事后聚出）与 SegVGGT 路线的分水岭。
_Avoid_：train-test 失配（这是它的**后果**之一，不是它本身）
