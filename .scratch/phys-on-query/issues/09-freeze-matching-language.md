# 09 — 冻结匹配语言加"除了"（glob + `!` 取反）

Type: task
Status: open
Blocked by: —
Blocks: 01 号票的 (b) 臂 config

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
