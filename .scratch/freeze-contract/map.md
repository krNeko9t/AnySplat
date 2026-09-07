# Map: 让"这个参数到底在不在训"变成可验收的事实

Label: wayfinder:map

## Destination

一层**可训参数指纹（trainable-parameter fingerprint）验收**，在训练启动时当场断言本次
run 的实际可训集合与配方声明一致；配套一份权威冻结契约文档。

**判据**：漏冻（如 `camera_token`）、误冻（如把本 stage 主体写进 freeze）、以及不走
`requires_grad` 的静默冻结（bf16 舍入），三类都在 step 0 炸掉，而不是训完才发现。

**不是**"设计一套新的冻结 API"——见 Out of scope 第一条。

## Notes

**这张图携带执行**（override wayfinder 默认的 plan-only）：终点是落地的校验层，不是它的
方案书。task 票直接动手；grilling 票只出决策。

**领域**：PyTorch Lightning + DDP 下的参数冻结与优化器分组。ICLR 2027 截止 **2026-09-25**。

**每个 session 应调用的 skill**：`grilling` + `domain-modeling`。

### 本图开工前已定的事（2026-09-03 grilling，不再重开）

1. **终点是验收，不是新语法**。核心痛点是"可训的 instance 分支长在 `Aggregator` 内部"
   （`src/model/segvggt/models/aggregator.py:182-234`），SegVGGT 路线因此**无法用一个
   `aggregator` handle 冻底座**，被迫逐子模块枚举——换任何声明式语法照样要枚举。
   会咬人的是"枚举漏了没人发现"，那是校验问题。
2. **指纹超出 `requires_grad`，但只作记录**（2026-09-04 由 [03 号票](issues/03-what-goes-in-the-fingerprint.md) 收窄，
   原文是「dtype 是第二真相源」）。`dtype` 进指纹、进哈希，**不设任何断言**——bf16 死参数
   （`cbe93f9`：68.5%≈623M 参数 50 步后从不更新，`requires_grad=True` 全程为真）**不是冻结**，
   它只占冻结三判据的第三条，机制在另一层（永久 cast ⇒ AdamW 状态也是 bf16）。管它叫「静默
   冻结」是比喻，比喻把它偷渡进了本图。**07 号票已于 2026-09-04 整票判出 scope**（见
   Out of scope），本图对它零承诺、零动作。dtype 那一列保住的是
   **可见性**：真发生时哈希撞一次，人被迫看一眼。
3. **`freeze_keywords` 现有两道护栏保留**：零命中 `raise`（`base_wrapper.py:526-527`）、
   只冻不解冻（`:515-521`）。第三道**已于 2026-09-04 由 09 号票补上**（`setup():533-546`）：
   声明的可训集合 vs 实际的可训集合。
4. **冻结入口必须唯一**（2026-09-04 由 [02 号票](issues/02-reconcile-six-schemes.md) 收窄，
   原文是「官方实现一律保留」）。**冻结入口 = config 能拨动的冻结开关**，只有 `freeze_keywords`
   一个。vendored 模型内部的**构造期冻结不变量**（LoRA 冻基座、`mask_token`）不是入口——config
   拨不动、且 `setup()` 跑时 LoRA 早已建完，原理上无法用 `freeze_keywords` 表达——**保留 + 记录**。
   「是官方代码」本身不再构成保留理由。

### 事实底座（可复核，别重新推导，2026-09-03 现场读码）

- ~~`freeze_research.md`~~ — 5 种冻结实现的分类；**已于 10 号票整份删除**，内容进
  `docs/repo_knowledge.md` §6.2 的定性表
- `docs/repo_knowledge.md:118 / :126 / :148` — 各 stage 冻结集的**意图**与实测参数量；
  机制现在在同文件 §6，术语在根 `CONTEXT.md`

关键事实：
- 唯一通用入口 `BaseWrapper.apply_freeze()`（`src/model/wrapper/base_wrapper.py:503-531`，
  09 号票从 `setup()` 拆出），匹配是裸子串 `kw in name`（`:515`）。
- `camera_token` = 1×2×1×1024 = **2048**，`register_token` = 1×2×4×1024 = **8192**，共
  **10240**（`aggregator.py:176-177`），`:486` 拼进每帧 token 序列喂给所有输出头。
- 全仓**只有一个**冻结入口 `freeze_keywords`；三个 arch 文件里 `grep requires_grad` 的全部命中
  都是构造期不变量（AnySplat 侧只有 `aggregator.patch_embed.mask_token` 一个参数 + distill 块）。
- `cbe93f9`（bf16 dtype 修复）**不在 `fix` 分支上**；`anysplat.py:113`、`iggt.py:80`
  仍是无条件 `.to(torch.bfloat16)`，且**按 07 号票的出图结论原样保留**。
  `arch/segvggt.py` 干净（只用 autocast）。`iggt.py:80` 那行经 `git log -L` 溯到
  `8bace2c` — 是本仓自己写的，**上游 IGGT 并无此 cast**。
- **可训的 bf16 aggregator 参数命中 5/22 份配方**（2026-09-04 按 09 的 22 份 lock 统计，
  取代 01 号票「九份里零次发生」的过期读数）：`instseg_small` 1209 个 / **909.1M**
  （占其 12.49 亿可训参数的 72.8%，`base_lr: 1e-5`），`co3d`/`dl3dv`/`multi-dataset`/
  `scannetpp` 各 866 个。**这是本图产物证伪本图事实底座的一次**，靠的是 03 保留的 dtype 列。
  它不改变任何冻结判据 ⇒ 归训练精度策略，见 Out of scope。
- 冻结相关 config 注释共 14 行（**过期读数：10 号票实测为 20 行**，判完后 16 行）；告警型长注释集中在
  `base_wrapper.py:503-531`（09 号票后的位置）+ `repo_knowledge.md` 四段。
- 历史上 freeze 相关修复 commit 共 4 次：`12aaec6`（改为增量式）、`5f1eff7`
  （补 camera_token/register_token）、`7e196e9`（stage-1 误冻 instance 主体）、
  `cbe93f9`（bf16 静默冻结）。

## Decisions so far

<!-- 一行一个已关闭的票 -->

- [10 — 文档三分与长注释归位](issues/10-converge-docs-and-comments.md)：
  **三分全部落地：根 `CONTEXT.md` 建成（7 词条纯术语）、机制并进 `docs/repo_knowledge.md` 新增 §6、
  `freeze_research.md` 整份 `git rm`、20 行 config 冻结注释逐条判完（净删 4 行）。**
  09 号票预警的「长注释里至少有一条内容错误」找到了：`repo_knowledge.md` 精度那条写
  「aggregator 跑 bf16 autocast」，实际是 `anysplat.py:113`/`iggt.py:80` 的**无条件永久 cast**——
  bf16 死参数的机制被一句「autocast」盖了过去。已改写并指回 `CONTEXT.md` 的隔离词条。
  注释判据（提醒 ⇒ 删 / 机制因果 ⇒ 留）落到实处：physgm 那 5 行
  「camera_token 容易漏、漏了会被悄悄拖动」删净——`5f1eff7` 那类漏冻现在在 lock diff 里就是两行硬错；
  手抄的实测数（`~0.2M`、`~487M`、`Trainable: ...`）一律删，lock 是唯一权威；
  裸子串匹配语义、DDP wrap 时序、论文出处一律留。`base_wrapper.py` 的长注释一行未动。
  **票面「14 行」是 02 号票时期的过期读数，实测 20 行**，已在票内更正。
  三份被改注释的配方重跑生成端，lock 全部 `unchanged` ⇒ 对指纹零影响。
  **`.scratch/` 之外全仓对 `freeze_research.md` 零引用**，无迁移说明、无 deprecated 标注。

- [09 — 落地指纹 + lock 层](issues/09-implement-lock-layer.md)：
  **校验层已落地并提交（`b53f7dd`），22 份 lock 全覆盖，四条完成判据逐条实测通过。**
  `src/freeze_contract.py`（唯一 capture 函数）+ `scripts/freeze_lock.py`（纯 CPU 生成端）+
  `BaseWrapper.setup()` 末尾校验。22/22 生成成功——**本票列为已知风险的 13 份未验配方无一失手**；
  再跑一次 diff 为空；**15 份有历史实测的配方逐参数零偏差**（01 的九份 + 08 的六份）；
  反做 `5f1eff7` 启动硬错且正文 diff 恰好两行；缺 lock 硬错；`stage="test"` 放行；成功不落文件。
  **三处偏离设计且更好**：(1) 校验比 lock **正文逐行**而非 lock 自己声明的哈希——否则手改正文
  不改哈希行完全放行，「人签字的文本」与「被执法的文本」可以不是同一个，那是 02/03 反复警告的
  「守自己的影子」跑进了 lock 文件内部；(2) 冻结实现拆出 `apply_freeze()`，否则生成一份 lock
  需要先有那份 lock；(3) 免权重从「打三个补丁」收成「config 层清空 `pretrained_weights`」——
  现场撞到第三条读权重路径（`from_checkpoint` 读集群本地 ckpt），三条路径共用同一个字段。
  **两条现场事实**：单份生成峰值 **RSS 7.3G**（这就是崩机机制：一个进程内连建多份，62G+7G swap
  撑不到 22 份，swap 抖死而非 OOM kill ⇒ `--all` 是串行子进程）；`base_wrapper.py:73` 的注释
  「all others unfrozen」与 02 号票的结论直接相反，已改。
  **对下游**：→ 10 解除阻塞，且它不能只搬不读——长注释里至少有一条是内容错误。

- [08 — 彻底清除 `freeze_backbone` / `freeze_module`](issues/08-erase-freeze-module.md)：
  **已删净，六份 anysplat 配方逐参数等价实测通过**（[报告](notes/anysplat_migration.md)）。
  两个字段 + `__init__` 里的整块判定（净 -50 行）删除，5 份配方迁到 `freeze_keywords`
  （四份上游 `[patch_embed]`、`instseg_anysplat` `[aggregator, camera_head, depth_head]`、
  `instseg_small` 不设），`freeze_research.md` 方案 2 整节删除并重编号，
  `repo_knowledge.md:222` 的例外句整句删除。02 号票「无条件赋值那半边今天零影响」的判断
  **被实测正面证实**：迁移前后没有任何参数的 `requires_grad` 发生翻转。
  预警的两个裸子串坑（`patch_embed` 多命中 `part_head`、`camera_head` 多命中 `distill_*`）
  **都未发生，且由实测而非推理证明**。
  **两条现场发现**：(1) AnySplat 路线的构造期冻结不变量只有 `aggregator.patch_embed.mask_token`
  一个参数；(2) 探针漏了第二条 HF 下载路径——`pretrained_weights: "hf:..."` 走
  `init_anysplat_from_hf`（= 构造 + `load_state_dict`，又一份约 5GB、只改参数**值**），
  不短路它 `instseg_*` 在 `--random-backbone` 下仍联网（实测卡 22 分钟）。
  **对下游的约束**：→ [09 号票](issues/09-implement-lock-layer.md) 生成端复用探针路径时必须
  同样短路它，否则 22 份 lock 的批量生成挂在网络上；→ 10 少一节要吸收，术语表里
  「构造期冻结不变量」的 AnySplat 实例可直接写 `mask_token`。

- [06 — 校验失败时做什么](issues/06-failure-behaviour.md)：
  **硬错三条（哈希不符 / lock 缺失 / 零命中）、只记录一条（`base_lr` 与 dtype 列）、无 warn 档；
  逃生门不设确认仪式；只在 `stage == "fit"` 校验，`fast_dev_run` 与 sanity check 零豁免。**
  开场划掉候选清单里的「跨 stage 约束被违反」——05 出图后这条判据不存在。逃生门选静默覆盖 + 无条件
  打印差异，因为 04 D4 已把签字定位在 **git diff**（可 review、可回溯），终端 `--yes` 是在信息量
  更少的地方再签一次，且拦不住「人本来就想改」，还会让批量重生成在无 TTY 下卡死。
  打印位置选 rank 0 logger、不进 TB text（错在 step 0 之前 raise，永远没有曲线），成功时不落文件
  （lock 已在 git 里）；但**失败时**把实测指纹全文写进 run 目录并在报错里点名——mismatch 常在远程
  节点上，若差异来自环境，本地重跑复现不出来，那份全文是唯一证据。报错正文必须点名
  experiment / lock 路径 / 重生成命令 / 头几行差异。
  只守 fit 是因为**冻结三判据全是训练期概念**：`mode == "test"` 下没有优化器，护栏守的是空气，
  而拦住「拿老 ckpt 复现一个数字」会逼人去找真的逃生门——D2 靠「跑不起来」立威，前提是每次都拦对。
  **现场发现**：`base_wrapper.py:507-508` 的 `if not freeze_kw: return` 必须删，否则 13 份不设
  `freeze_keywords` 的配方走不到校验点，04 D2 当场失效。
  **对下游的约束**：→ [09 号票](issues/09-implement-lock-layer.md) 得到校验端挂点、门条件、
  失败路径与异常正文四要素；→ 10 无新增术语。

- [04 — 契约存哪、怎么写、谁维护](issues/04-where-the-contract-lives.md)：
  **lock 落 `config/experiment/locks/<X>.lock`（`<X>` = hydra 的 experiment choice，即 yaml 文件名；
  路径 2026-09-04 由 09 号票 session 修订，原为同目录同名的 `<X>.freeze.lock`），
  22 份配方全覆盖、缺 lock = 启动硬错，生成端是独立 CPU 脚本且与校验端共用同一个 capture 函数。**
  身份键不用 `wandb.name`——实测 22 份里 4 份与文件名不等、**两组重名**，重名 = 两份配方共用一份 lock。
  选「全覆盖 + 缺 lock 硬错」而非「有就校验」，因为 02 号票删掉 `freeze_module` 的前提正是「lock 是强制的」，
  且它把第三种故障形态（**忘了写 `freeze_keywords`**）也纳入——与 03 选全量参数同一条理由：
  **不能让「没有」和「被冻了」同形**。生成端不走 `src/main.py`，因为那要起一遍 Trainer 才到 `setup()`，
  把纯 CPU 的事绑死在 GPU 节点上。
  **文档三分**：术语（冻结三判据 / 冻结入口 vs 构造期不变量 / 指纹 / lock / bf16 死参数不是冻结）
  进**新建的根 `CONTEXT.md`**，机制现状并进 `docs/repo_knowledge.md`，`freeze_research.md` 整份删。
  术语单列是因为 02 与 03 是**同一个错误跑在代码层和概念层**，术语表是防它的常驻器官。
  **对下游的约束**：→ 06 硬错档追加「lock 缺失」，逃生门收缩为「重生成并提交」（`skip_freeze_check`
  出局）；→ [09 号票](issues/09-implement-lock-layer.md) 实现；→ [10 号票](issues/10-converge-docs-and-comments.md) 文档与长注释。

- [03 — 可训参数指纹里放什么](issues/03-what-goes-in-the-fingerprint.md)：
  **一个捕获点（`setup()` 末尾）、覆盖全量参数、一行一参数按 name 排序的定宽文本、顶部一个哈希、
  零个 if。** 字段 = `requires_grad` + `dtype` + `numel` + `name`；头部 = experiment / arch /
  `freeze_keywords` 原文 / `base_lr` / 哈希（不记 git commit）。选全量而非可训子集，是因为
  **可训子集分不清「被冻了」和「不存在了」**。选定宽文本而非 JSON，是因为 lock 唯一的读者场景是
  `git diff`——`5f1eff7` 那类故障在 diff 里就该是**一行**。
  **砍掉两样**：(1) **`lr`/分组判等整个出图**（它要两点捕获 + 双哈希 + 四元组，是 round 2 全部
  结构性复杂度的来源，且 `lr` 不占冻结三判据的任何一条 ⇒ Out of scope；`base_lr` 只记进头部、
  不判等，留可见性扔执法）；(2) **dtype 的专门断言整条撤销**（06 号票那条判据自己就要一个
  「只算进了优化器的」分支 if，是在为另一层的机制擦屁股）。据此收窄 Notes 第 2 条（见上）。
  **对下游的约束**：→ 04 正文形态与字段已定，04 只决定存放位置/生成流程/是否强制；
  → 06 dtype 判据整条删除，失败行为只剩「结构哈希不符 = 硬错」一档；→ 07 与本图脱钩。

- [02 — 六种冻结实现：哪些留、哪些记录、哪些删](issues/02-reconcile-six-schemes.md)：
  **`freeze_backbone`/`freeze_module` 判定彻底清除**（→ [08 号票](issues/08-erase-freeze-module.md)）。
  它确是 AnySplat 官方上游代码（`8d6180e`，逐字未改，用它的 4 份 config 同出该 commit），但也确是
  `freeze_keywords` 的**严格功能子集**——唯一的非子集部分（else 分支无条件赋值解冻）只能作用于
  `mask_token`/distill 块，两者靠名字巧合躲过 ⇒ 删掉零行为变化。据此推翻并收窄 Notes 第 4 条为
  **「冻结入口必须唯一」**（见上）；LoRA/`mask_token` 作为构造期不变量照旧保留。
  连带纠正一条被两份文档写反的因果（**并非**「只在 `freeze_keywords` 为空时生效」；真实顺序
  `__init__` → `setup()`，后跑的 `freeze_keywords` 只冻不解冻 ⇒ **它赢**，净效果是并集），
  `freeze_research.md:37` 与 `docs/repo_knowledge.md:222` 已在票内修正。
  **对 04 的约束输入**：本票选「指纹层天然覆盖」而不加专门断言，前提是 04 把 lock 定成**强制**。

- [01 — 实测当前每份 config 的真实可训集合](issues/01-measure-current-trainable-sets.md)：
  九份配方的事实基线已落地（[读数](notes/trainable_sets.md) + [逐参数原始记录](notes/raw/)，
  后者即 03 号票要设计的指纹的样本数据）。七份与 `repo_knowledge.md` 一致；两处不一致：
  **`segvggt_agnostic_phys_joint` 的实际 lr 比注释高 5 倍**（可训集合完全正确，
  现有两道护栏抓不到纯数值型偏差 ⇒ 03 号票「指纹要不要带 lr」的实物论据），
  **`repo_knowledge.md:222` 关于 `freeze_module` 生效条件的一句是错的**（⇒ 02 号票）。
  另：**bf16 死参数在这九份里零次发生**（IGGT 五份都把 bf16 的 aggregator 冻了）——
  **此读数已于 2026-09-04 在 22 份上被推翻，见事实底座**；
  06 号票的 dtype 判据是前瞻护栏而非现存 bug；`phys_iggt` 物理阶段仍训几何头 ⇒ 05 号票；
  stage 链集合关系已算出且今天自洽 ⇒ 05 号票。

## Not yet specified

<!-- 04 号票（2026-09-04）清掉了原有的两条：
     「长注释怎么收敛」已可精确表述 ⇒ 毕业为 [10 号票](issues/10-converge-docs-and-comments.md)；
     「新 arch 怎么被强制纳入」已被 04 D2 直接回答 ⇒ 自动覆盖，新 arch 总以新 experiment config
     的形式到来，而缺 lock 是硬错，它跑不起来直到有人生成并看过一份 lock。 -->

（无。雾已散尽，全部 10 张票均已关闭——**本图到达终点**：
校验层在 `src/freeze_contract.py` + `BaseWrapper.setup()`，22 份 lock 在 `config/experiment/locks/`，
术语在根 `CONTEXT.md`，机制在 `docs/repo_knowledge.md` §6。）

## Out of scope

- **跨 stage 的冻结关系校验**（2026-09-04 由 [05 号票](issues/05-cross-stage-relation.md) 划出，
  该票整票关闭，不在路线上解决）：全仓**只有一条**真正的 stage 链边（`segvggt_physgm` ←
  `segvggt_finetune_agnostic` 的产物），五份 IGGT 配方是同一外部基座的平行分支而非续作。
  本图对链断裂的实际防线是 **lock 的 `git diff`**（`5f1eff7` 那类漏冻在 diff 里就是多出的两行）；
  跨 stage 约束比它多抓的只有「配方第一次就写错」一类。为 n=1 条、且实测完全自洽
  （`trained(s1) \ frozen(s2)` = 0）的链边建第二套机制，收益对不上。
  两个实现走法都另有硬伤：静态 `continues_from` 与实际加载的 ckpt 无绑定（`pretrained_weights`
  是可被 CLI 覆盖的路径），ckpt 盖戳则在 22 份里 21 份的无戳路径上静默。
  指纹正文已是逐参数名 ⇒ 未来真要建，它是一个**读两份 lock 的独立脚本**，不回改指纹格式。
  连带出图：**F4**（`phys_iggt` 与同路线邻居对「几何该不该动」相反）——它们不构成链，
  各自的 lock 锁各自的集合就够。

- **`lr` / `param_groups` 的判等**（2026-09-04 由 [03 号票](issues/03-what-goes-in-the-fingerprint.md) 划出）：
  冻结的三条判据是不建计算图、不进优化器、权重不变；`lr` 一条都不占。把它纳入判等要付两点捕获
  （分组的真相源在 wrap **之后**的 `configure_optimizers`）+ 双哈希 + lr 四元组的代价。
  F1（`segvggt_agnostic_phys_joint` 实际 lr 比注释高 5 倍）的真身是「注释里的 base lr 过期了」，
  属配方审查，不由冻结层持枪站岗。`base_lr` 仍记进 lock 头部（不参与哈希、不判等）保留可见性。

- **bf16 死参数 / 永久 `.to(bfloat16)`**（2026-09-04 由 [07 号票](issues/07-bf16-silent-freeze.md) 划出，
  该票整票关闭）：它只占冻结三判据的第三条，机制在另一层，属**训练精度策略**。
  `cbe93f9` 的 `aggregator_param_dtype` 开关**不搬过来**——它既不解决问题也不发现问题，
  只能告诉你「dtype 有没有被改过」，而那正是 03 保留的 dtype 列已经免费提供的；
  再加一个 config 旋钮是在同样的可见性上叠一层，还多一个能拨动的开关。真正的修法是
  拿掉那行无条件 cast，那是精度决策不是开关决策。命中面确实不小（5/22，`instseg_small`
  的 72.8% 可训参数在其中，见事实底座），但**读数变了不等于判据变了**——与 `lr` 出图、
  跨 stage 出图同一条理由。本图对它的全部承诺只有一条且已兑现：改了，lock 的哈希会撞。
  **去向**：寄存为 `.scratch/phys-on-query/issues/06-aggregator-precision.md`（该图明确标注
  它不在路上、不阻塞任何票），只因落在同一双眼睛前，不因它属于那张图。

- **把 `instance_*` 搬出 `Aggregator`**：物理上能让 backbone 变成可整体冻的干净模块，
  但要改 vendored 结构 + 全量 ckpt 键 remap + 与上游彻底分叉。距 ICLR 截止 22 天，纯风险。
- **"要不要放开 LoRA"（冻结底座 23.4 vs LoRA joint 31.9）**：这是训练配方决策，直接决定
  第一个 checkpoint 的质量，属于 `phys-on-query` 图的终点范围，不是冻结机制问题。
  见 `.scratch/phys-on-query/issues/01-unfreeze-lora.md`。
  **2026-09-07 该票已关，并反向咬回本图一口**：它要的配方"block 里除了 lora 全冻"，
  用裸子串 OR **写不出来**——实测最接近的写法会连带冻死 `instance_cross_blocks`(302.32M)
  + `instance_query_self_attn`(151.30M)，即 `7e196e9` 那类故障。这**证伪了本图 Notes 第 1 条
  与下面"设计新冻结 API"那条所依赖的前提的一半**（原文：「换任何声明式语法照样要枚举，
  会咬人的是枚举漏了没人发现，那是校验问题」）——真问题不是枚举，是**表达力**：
  裸子串做不出"除了"。
  **处置：本图不重开**（终点仍成立，验收层确实建成了），该窄缺口由 `phys-on-query` 图承接为
  `.scratch/phys-on-query/issues/09-freeze-matching-language.md`（fnmatch glob + 前缀 `!` 取反，
  一个循环、零新增分支、零命中 raise 与只冻不解冻两道护栏原样存活，LoRA 构造期不变量不动）。
  它落地时会**机械替换 22 份 config 的裸关键词并重生成 22 份 lock，diff 必须全空**——
  即本图 09 号票那台机器的第一次真实体检。**本图对该改动零承诺、零动作**，
  留此一行只为不让两张图互指。
- **"判断某段是否需要前向"这类手动筛查/显存优化**：冻结的目的只有三条——不建计算图、
  不进优化器、权重不变。其余不在此列。
- **教师网 CPU 卸载 / `torch.no_grad` 推理路径**（`freeze_research.md` 方案 3/5）：
  是推理与显存管理，不是训练期冻结。
