# 分层方案契约（Layered Scheme）

> 本仓库是探索版：要频繁换 backbone × head × loss × 数据格式。
> **层与层隔离是第一设计原则**——改一个小模块不应牵动其他层。
>
> Physics 路径是首个按本规范落地的范例；**新增能力与重构旧路径时，默认按本规范写。**
> 现状说明见 [`repo_knowledge.md`](repo_knowledge.md)；编号/语义约定见 [`conventions.md`](conventions.md)。

## 1. 四层契约

只允许上层依赖下层；跨层只通过**命名类型**传数据，禁止散落字符串 key / 硬编码映射。

```
Dataset          Parser  →  *Target          （外部格式 → 监督契约）
Encoder          Head(+Classifier) → *Prediction  （可学习；填 EncoderOutput 槽）
Loss             resolver + 公式            （无可学习参数；只读 Pred/Target）
Experiment       parser + head cfg + loss yaml  （Hydra 组合切换方案）
```

| 层 | 职责 | 有无可学习参数 | 禁止做什么 |
|----|------|----------------|------------|
| Dataset / Parser | 读外部标注，产出 `*Target` | 无 | 在 dataset 里写死类别映射却不经 parser；把 raw JSON 直接塞进 batch |
| Encoder / Head | 前馈特征与 logits，填 `*Prediction` | **有** | 在 loss 里放 classifier / 投影层；改已有 `EncoderOutput` 字段语义 |
| Loss | pred↔GT 对齐（resolver）+ 公式 | **无** | 拥有 `nn.Module` 权重；反向被 wrapper 借权重做推理 |
| Experiment | 用 Hydra 组合三块配置 | — | 在代码里用 if 开关方案 |

### 数据流（以 Physics 为例）

```
manifest + physics_labels.json
        │
        ▼
 PhysicsParser  ──►  PhysicsTarget          batch["physics_target"]
        │
        ▼
 PhysicsHead + PhysicsClassifier
        │
        ▼
 PhysicsPrediction  ──►  EncoderOutput.physics_prediction
        │
        ▼
 resolve_instance_ce(pred, target)  ──►  focal / CE
```

Instance 等旧路径仍可能用 `depth_dict` 裸 key；**新路径必须用结构化 `*Target` / `*Prediction`。** 重构旧路径时按本规范迁移，不要再加兼容层。

## 2. 改需求时指哪里

人类指路径、AI 只改对应层。**不要**为了「方便」把映射、权重、公式揉进同一文件。

| 改什么 | 改哪里 |
|--------|--------|
| 新标注格式 | `src/dataset/<domain>/parsers.py` + 注册表；experiment 设 `*_parser` |
| 监督字段 / 编号语义 | `src/dataset/<domain>/types.py`（`*Target`）+ `docs/conventions.md` |
| dense / 特征 head | `src/model/encoder/.../<name>_head.py` |
| 分类器 / 可学习读出头 | 同目录 `*_classifier.py`（或 head 内子模块）；**挂在 encoder** |
| 预测槽字段 | `*Prediction` dataclass → `EncoderOutput` 可选字段（默认 `None`） |
| loss 公式 / pred↔GT 对齐 | `src/loss/loss_*.py`（`resolve_*` + 公式） |
| 方案组合 | `config/experiment/*.yaml` = parser + encoder head cfg + `loss: [...]` |

### Physics 落地路径（范例）

| 改什么 | 改哪里 |
|--------|--------|
| 新标注格式 | [`src/dataset/physics/parsers.py`](../src/dataset/physics/parsers.py) → `PHYSICS_PARSERS`；`dataset.manifest.physics_parser` |
| 监督语义 | [`src/dataset/physics/types.py`](../src/dataset/physics/types.py)（`PhysicsTarget` / `PhysicsPropertyTarget`） |
| dense 特征 | [`src/model/encoder/iggt_heads/physics_head.py`](../src/model/encoder/iggt_heads/physics_head.py) |
| 方案装配 | [`src/model/encoder/physics_scheme.py`](../src/model/encoder/physics_scheme.py)（`phys_scheme`: `class` / `property`） |
| 分类器 | [`src/model/encoder/iggt_heads/physics_classifier.py`](../src/model/encoder/iggt_heads/physics_classifier.py) |
| 属性读出 | [`src/model/encoder/iggt_heads/physics_property_readout.py`](../src/model/encoder/iggt_heads/physics_property_readout.py) |
| 预测槽 | [`physics_prediction.py`](../src/model/encoder/physics_prediction.py) / [`physics_property_prediction.py`](../src/model/encoder/physics_property_prediction.py) → `EncoderOutput` |
| loss | [`src/loss/loss_phys.py`](../src/loss/loss_phys.py)（class）/ [`src/loss/loss_phys_prop.py`](../src/loss/loss_phys_prop.py)（property） |
| 实验 | [`config/experiment/phys_iggt.yaml`](../config/experiment/phys_iggt.yaml) / [`phys_prop_iggt.yaml`](../config/experiment/phys_prop_iggt.yaml) |

## 3. 硬性规则

1. **可学习参数不进 loss。** Classifier / 投影层挂在 encoder（或明确的 model 子模块）。Loss 只做 resolver + 公式。Validation / 推理读 `*Prediction`，禁止 `_get_*_classifier()` 从 `self.losses` 反查。
2. **类别名与 id 映射只活在 parser（+ conventions）。** 禁止在 dataset / wrapper / loss 再抄一份 `PHYS_CLASS_NAMES` / `LABEL_MAP`。
3. **维数配置单点。** 如 `phys_feat_dim` / `num_classes` 只在 encoder（或唯一权威 cfg）写一次；loss 不重复持有「架构维」字段。
4. **`EncoderOutput` 只加可选槽。** 新字段默认 `None`，不改已有字段含义。
5. **方案切换靠 Hydra，不靠代码分支。** 一个 experiment ≈ 一个 parser + 一组 head 配置 + 一个 loss 列表。
6. **不要预留兼容层。** 探索版允许丢掉旧接线重写；不要同时维护 `phys_label_map` 与 `PhysicsTarget` 两套。
7. **Collate 要能装契约对象。** 变长 LUT / dataclass 用专用 collate（如 `src/dataset/collate.py`），不要假设 `default_collate` 能 stack 一切。

## 4. 新增一条能力的检查清单

- [ ] `*Target` dataclass + Parser 协议 + 注册表
- [ ] `*Prediction` dataclass + `EncoderOutput` 可选字段
- [ ] Head（特征）与 Classifier/读出（若需要）挂在 encoder
- [ ] Loss：`resolve_*` + 公式，`parameters()` 为空
- [ ] Wrapper：只搬运 `*Prediction` / `*Target`，不做二次 pooling / 借权重
- [ ] `config/`：parser 字段、encoder head cfg、`config/loss/*.yaml`、experiment 组合
- [ ] 在本文件或 `repo_knowledge.md` 的「改需求时指哪里」表里加一行
- [ ] 编号语义写入 `conventions.md`（若引入新 ignore / 索引约定）

## 5. 反模式（曾出现，禁止再写）

| 反模式 | 正确做法 |
|--------|----------|
| Classifier 定义在 `LossPhys.__init__` | `PhysicsClassifier` 在 encoder |
| Wrapper `_get_phys_classifier()` 遍历 losses | 读 `encoder_output.physics_prediction` |
| `phys_feat_dim` 在 encoder yaml 与 loss yaml 各写一遍 | 只在 encoder cfg |
| `PHYS_LABEL_MAP` 写在 `DatasetManifest` 里 | Parser 内 + `PhysicsTarget` |
| 监督全塞 `depth_dict["phys_label_map"]` 字符串 | `batch["physics_target"]` / `depth_dict["physics_prediction"]` 结构化对象 |
| 为旧 key 留半年兼容别名 | 直接删旧接线，一次改完调用方 |
