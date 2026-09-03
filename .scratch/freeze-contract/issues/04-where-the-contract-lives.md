# 04 — 契约存哪、怎么写、谁维护

Type: grilling
Status: open
Blocked by: 03

## Question

指纹要和**声明**比对才有意义。声明放哪：

(a) **写进 experiment yaml**：如 `optimizer.expect_trainable: {instance_: 454M, ...}`。
    改配方就要改声明，改动可见于同一份 diff。缺点：yaml 变长，且手写数字会过期。
(b) **旁挂 lock 文件**：`config/experiment/xxx.freeze.lock`，由脚本生成、提交进 git。
    改配方 ⇒ lock 不匹配 ⇒ 训练拒绝启动 ⇒ 人跑一次 `--update-lock` 显式确认。
    像 `package-lock.json`。缺点：多一类文件、多一个仪式。
(c) **不存声明，只做跨 stage 比对**：见 05 号票。适用于 stage 链，管不住单份配方写错。

还要决：**首次生成**怎么来？01 号票的实测产物直接落成初始 lock，还是人手写一遍？

**注意**：这条与地图 Out of scope 里"不设计新冻结 API"不冲突——`freeze_keywords`
的写法一个字不改，加的是它旁边的验收物。

## 完成判据

声明的存放位置、格式、生成与更新流程定下来，且明确"改配方"时人要做的动作是什么。
