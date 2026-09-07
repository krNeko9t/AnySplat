# 09 — 冻结匹配语言加"除了"（glob + `!` 取反）

Type: task
Status: closed (2026-09-07)
Blocked by: —
Blocks: 01 号票的 (b) 臂 config
Assignee: @krNeko9t (session 2026-09-07)

> 由 [01 号票](01-unfreeze-lora.md) 逼出。**本票破了本图 Out of scope 的"冻结机制本身"那条**，
> 是一次有据的开口，不是失守——见下方"为什么归本图"。

## Question

`freeze_keywords` 是裸子串 OR 匹配（`src/model/wrapper/base_wrapper.py:515`，`kw in name`）。
**它表达不出"除了"**，因此 01 号票要的配方——"block 里除了 lora 全冻"——**写不出来**。

实测：最接近的写法（`.mlp. / .norm1. / .norm2. / .ls1. / .ls2. / .attn.`）会连带冻死
`aggregator.instance_cross_blocks`(302.32M) + `aggregator.instance_query_self_attn`(151.30M)，
即**把本 stage 要训的主体冻了**——正是 `7e196e9` 修过的那类故障。
逐块枚举（24×2 块 × 5 组件 ≈ 240 条关键词）能work，但会污染 lock 头部且次日即删。

### 改法（已定，2026-09-07 grilling）

**A：`kw in name` 换成 fnmatch glob + 前缀 `!` 取反，同一个循环里 last-match-wins。**

```yaml
freeze_keywords:
  - "*frame_blocks*"
  - "*global_blocks*"
  - "!*.lora.*"       # 唯一的新语义
```

- 仍是**一个函数、一个循环、零业务 if**（`apply_freeze()`，`base_wrapper.py:503-531`）。
- 两道现有护栏原样存活：**零命中 raise**（`:526-527`）、**只冻不解冻**（`:515-521`）。
  `!` 的语义是"**该关键词不选中它**"，不是"解冻它"——所以 LoRA 构造期冻死 attn 底座那条
  不变量（`layers/lora.py:123-125`）分毫未动。这条必须在实现时守住，否则就把 freeze-contract
  02 号票确认过的不变量推翻了。
- **不选** B（反转极性、config 声明可训集合）：语义更贴"我要训什么"，但与"只冻不解冻"正面冲突，
  且 22 份配方要逐份重写而非机械替换。19 天内不值。

### 为什么归本图（而不是 freeze-contract）

`freeze-contract` 图**已到达终点**（10 票全关）。它的 Notes 第 1 条写死"终点是验收，不是新语法"，
Out of scope 明文划出"设计一套新的冻结 API"，理由是：

> 换任何声明式语法照样要枚举 —— 会咬人的是"枚举漏了没人发现"，那是校验问题。

**这条前提被 01 号票的实测证伪了一半**：问题不是"照样要枚举"，是**裸子串在表达力上写不出来**，
一条合法配方根本不存在对应写法。这不是"语法更好看"，是"这条配方压根写不出来"。
2026-09-07 判定：**不重开 freeze-contract**（它的终点仍然成立——验收层确实建成了），
把这一个窄缺口收进本图，因为只有本图需要它。已在 freeze-contract 的 Out of scope 留反向指针。

## 完成判据

1. `apply_freeze()` 改为 glob + `!`，零新增业务分支，两道护栏实测仍生效。
2. **22 份 config 的裸关键词机械替换成 `*x*`**（`x` → `*x*`）。
3. **重生成 22 份 lock，`git diff` 必须全空**——这是本次迁移唯一的验收，
   也是 `freeze-contract` 09 号票那台机器该干的第一件活。
4. joint 的 (b) 臂 config（新文件，不改 (a) 臂那份）加 `!*.lora.*`，
   重生成其 lock，diff **恰好 192 行**从 `-` 翻成 `T`（block LoRA 192 张量 / 9.44M），
   **不多不少一行**。这是"改完就知道对不对"的当场证明。


---

## 解决 (2026-09-07)

**做完了，(b) 臂现在跑得起来。** `apply_freeze()` 的匹配语言换成 fnmatch glob + 前缀 `!` 取反、
后命中者胜，改法与票面的 A 案逐字一致：**一个函数、一个循环、零新增业务分支**
（`src/model/wrapper/base_wrapper.py:503-548`）。

```python
for kw in freeze_kw:
    negated = kw.startswith("!")
    if fnmatchcase(name, kw[1:] if negated else kw):
        hit_counts[kw] += 1
        selected = not negated  # last match wins
```

### 四条完成判据，逐条实测

1. **零新增分支 + 两道护栏仍生效**——`!` 的语义钉死为"该关键词不选中它"，
   `requires_grad = True` 这个赋值在函数里**根本不存在**。六项实测（真模型，`phys_query_arm_b_lora`）：
   - 零命中 `raise` 对**普通关键词**和 **`!` 关键词**都照样炸（`!` 拼错和普通拼错一样危险：
     它本该放行的东西会被冻掉）；
   - 只写 `["!*.lora.*"]` 时 `apply_freeze()` 一个参数都不选中，
     **192 张 LoRA 基座（`*.linear.weight/bias`）依旧是冻的**——构造期冻结不变量
     （`layers/lora.py:123-125`、freeze-contract 02 号票确认过的那条）分毫未动；
   - `[*frame_blocks*, !*.lora.*]` → LoRA 可训；把两条对调 → LoRA 冻死。后命中者胜可观测。
   同样六条固化成**单元测试** `tests/test_freeze_matching.py`（把 `apply_freeze` 当未绑定函数
   打在一个 stand-in 上，毫秒级，不建 5GB 底座），`6 passed`。

2. **配方机械迁移**：`x` → `"*x*"`，**17 份**（第 18 份 `phys_query_arm_b_lora.yaml`
   05 号票已按新语法写好）。全仓 `freeze_keywords` 里没有一个关键词带 glob 元字符，
   替换是纯机械的。
   ⚠️ **票面的"22 份"是旧数**：`config/experiment/` 现有 **26 份 yaml**，其中 **18 份**自己声明
   `freeze_keywords`，其余 8 份靠 defaults 继承（它们的 lock 也跟着变了，说明迁移覆盖到了）。

3. **重生成 26 份 lock，正文全空 diff。** 先在**改动前**跑了一遍 `--all` 打基线
   （25/26 unchanged，唯一失败的就是 `phys_query_arm_b_lora`——它带着 `!` 而当时还没有 `!`），
   改完再跑一遍：**26/26 生成，每份 lock 恰好差 1 行，且那一行必是 lock 头部回显的
   `freeze_keywords:`**（`[patch_embed]` → `[*patch_embed*]`）。
   `structure:` 哈希与逐张量正文**一行未动**。
   ⚠️ 所以**判据 3 的字面"git diff 必须全空"不可能成立，也不该成立**——lock 头部按设计
   逐字回显配方的关键词，配方文本变了它就得变。判据的**实质**（可训集合零变化）成立，
   而且比字面版更强：不是"没人看"的空 diff，是 22 份 lock 各一行、肉眼一眼扫完的 diff。

4. **(b) 臂的 lock：恰好 192 行从 `-` 翻成 `T`，不多不少一行。**
   与 (a) 臂 lock 逐行比（2628 行，等长）：**195 行不同 = 192 行 `-`→`T` + 3 行头部**
   （配方名、`freeze_keywords` 回显、`structure` 哈希）。192 行**全部含 `.lora.`**，
   合计 **9,437,184 参数 = 9.44M**，与 01 号票从官方权重逐张量数出来的数字对上。

**顺带**：`freeze_contract.verify()`（启动时那道硬错）在 CPU 上对
`phys_query_arm_b_lora` / `phys_query_arm_a_frozen` / `segvggt_agnostic_phys_joint`
三份配方实跑一遍，**全 PASS**——(b) 臂不会在 `setup()` 被自己的护栏拦下。

### 一并改掉的文档与注释

- `docs/repo_knowledge.md` §6.1 的"三条语义"第 1 条从"裸子串匹配"改写为
  "glob + `!` 取反、后命中者胜"，并点明**裸写 `patch_embed` 现在匹配不到任何东西**
  （会撞第 3 条零命中硬错——这是本次迁移唯一会咬人的地方，而它咬得很响）；
  第 2、3 条补上 `!` 的位置。§8 第 5 条同步。
- 同一份文档里 22→26 的陈旧计数就地核实改正：26 份 yaml、7 份 `wandb.name` 与文件名不等、
  三组重名（不是原文的"4 份 / 两组"）。`scripts/freeze_lock.py` 的 docstring 同理。
- `config/experiment/phys_query_arm_b_lora.yaml` 顶部的
  "⚠️ NOT RUNNABLE YET -- BLOCKED ON TICKET 09" 换成实测结果。

### 没做、且有意没做的

- **`param_groups[].keywords` 仍是裸子串**。它管的是分组学习率，不是冻结；本票的开口
  （见票面"为什么归本图"）只有"表达『除了』"这一个，不含"把 glob 铺到别处"。
- **没碰冻结机制的其余任何部分**——极性没反转、声明式可训集合（票面的 B 案）没做、
  校验层没动。本图 Out of scope 的"冻结机制本身"那条**仍然只开了这一个口**。

### 改动清单

- `src/model/wrapper/base_wrapper.py`（`apply_freeze` + `fnmatchcase` import）
- `tests/test_freeze_matching.py`（新增，6 个测试）
- `config/experiment/*.yaml` 17 份 + `phys_query_arm_b_lora.yaml` 注释
- `config/experiment/locks/*.lock` 22 份各 1 行 + 新增 `phys_query_arm_b_lora.lock`
- `docs/repo_knowledge.md`、`scripts/freeze_lock.py`（docstring）
