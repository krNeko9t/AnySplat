---
id: T13
title: 修复 re10k manifest 路径缺场景目录
type: wayfinder:task
status: open
assignee: -
blocked-by: []
---

## Question

**R2 挖出的代码事实：`processed_re10k` 现在根本加载不了。** 地图「已锁定的配方」里
数据那一行写的是 `processed_scannetpp_v2` + `processed_re10k`，但事实上短期内只有前者。

证据：

- `scripts/make_manifest_inscene_scannetpp_v2.py:132-135` 写路径时带 `scene_rel` 前缀
  （相对 `data_root`）：`str((scene_rel / img_path.relative_to(scene_dir)))`。
- `scripts/make_manifest_inscene_re10k.py:85-86` 写的是 `str(rgb_path.relative_to(scene_dir))`
  ——**相对场景目录，没有场景前缀**。
- `src/dataset/dataset_manifest.py:96` `self.root = cfg.root` 是整库唯一的根；
  `:134-136` 的 `_as_path` 把相对路径直接拼到 `root` 上。

→ re10k 的帧路径会解析成 `root/<frame>.jpg`，落不到实际文件上。
`git log` 的 `b486ef9` 提交信息原文即「…单独 ScanNet++ v2 数据集训练验证通过。**re10k未定**」，
与这个结论吻合。

## 要做的

1. 修 `make_manifest_inscene_re10k.py` 的路径写法，与 scannetpp 脚本对齐（加场景前缀）。
   **注意**：决定是改脚本重新生成 manifest，还是让 `DatasetManifest` 容忍两种写法——
   **倾向改脚本**，因为 dataset 层是既有实现、且两种路径语义并存会长期埋雷。
2. **顺带排查 `make_manifest_inscene_re10k.py:46-48,59` 的 `zip(sorted(...))` 位置配对**。
   R2 指出：如果 rgb 与 instance mask 是按各自排序后位置配对的，而两边文件名规则不一致，
   会**错帧配对**。错帧的表现恰恰就是「跨视角 id 全对不上」——**别把自己的 bug 归因给上游
   标注质量**。确认配对是按文件名 stem 匹配而非位置匹配（scannetpp 脚本 `:83-88,103` 是
   按 stem 配对的，照它改）。
3. 重新生成 re10k manifest，确认能加载。

## 谁来跑

**HITL**：需要挂载 `/mnt/shared-storage-gpfs2/...` 的数据集，本机没有。
代码改动可以在本机完成并做 CPU 层面的路径拼接单测；**重新生成 manifest + 实际加载验证
需要人在集群上执行**。

## 验证

- CPU：构造一个假的目录树，跑脚本，断言写出的 `rgb_path` / `instance_mask_path` 拼上 `root`
  之后能解析到正确文件；断言 rgb 与 mask 是按 stem 配对的（造一个「两边文件名排序不一致」
  的用例，位置配对会挂、stem 配对能过）。
- 集群：加载若干个 re10k 场景，确认帧数、shape、id 分布正常。

## 影响

在这张票关掉之前，**地图的数据配比事实上是「只有 scannetpp_v2」**。T7 写配置、T9 smoke run、
T10 全量训练如果在此之前动手，都要按单子集处理，并在各自的 `## 解决` 里写明。
