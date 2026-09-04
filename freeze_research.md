我先探索代码库中与冻结参数相关的实现。
本仓库在训练时"冻结部分网络"共有 **6 种实现方案**，其中前 3 种是真正参与训练的冻结，另外几种是辅助手段。下面按重要性说明。

---

## 方案 1（官方唯一通用入口）：`optimizer.freeze_keywords` + `BaseWrapper.setup()`

**实现要点**（`src/model/wrapper/base_wrapper.py:501-528`）：

```python
501:    def setup(self, stage: str) -> None:
506:        freeze_kw = list(self.optimizer_cfg.freeze_keywords or [])
511:        for name, param in self.named_parameters():
512:            matched = [kw for kw in freeze_kw if kw in name]
519:            if matched:
520:                param.requires_grad = False
524:            raise ValueError(f"freeze_keywords matched no parameters: {missed}")
```

三个设计细节值得注意：

1. **时机必须在 `setup()`，不能在 `configure_optimizers()`**。注释写得很清楚（`base_wrapper.py:502-505`）：`setup()` 在 strategy 包装模型**之前**执行，而 DDP reducer 只注册包装那一刻 `requires_grad=True` 的参数——包装后再翻转，要么报错，要么**静默跳过梯度同步**。
2. **"只冻不解冻"（additive only）**。命中就置 `False`，未命中的保持建模时的状态不动。早期版本是无条件赋值 `requires_grad = not matched`，会把构造期已冻结的参数（LoRA 基座）悄悄解冻，导致基座 + adapter 一起训、LoRA 完全失效（`base_wrapper.py:515-518`）。
3. **关键词零命中直接抛错**，防止拼错单词导致"以为冻了其实没冻"。

配置声明在 `OptimizerCfg`（`base_wrapper.py:64-77`）。实例：`config/experiment/segvggt_physgm.yaml:74-83` 用 9 个 keyword 把模型冻到只剩 ~0.2M 可训参数。

---

## 方案 2：arch 构造期冻结 —— `freeze_backbone` / `freeze_module`（AnySplat 老路线）

`src/model/arch/anysplat.py:157-193`：

- `freeze_backbone: true` → 冻 `aggregator + camera_head + (depth_head | point_head)`（`:157-167`）
- `freeze_backbone: false` 时走 `freeze_module`（`:168-193`）：`"None"` / `"all"` / 组合名（`patch_embed+frame`、`patch_embed+global`、`global+frame`）/ 任意单模块名（如 `patch_embed`）。

⚠️ **单模块名分支是无条件赋值**（`:191-193`），语义与方案 1 相反：它会把未命中的参数一律置 `requires_grad=True`，解冻别人冻的。

**执行顺序**：本方案在 `EncoderAnySplat.__init__` 里跑，方案 1 在 `BaseWrapper.setup()` 里跑，`__init__` 在前。所以方案 1 后跑、只冻不解冻，**它赢**——两者并用时净效果是二者冻结集的并集。本方案的解冻只能作用于**比 `__init__` 更早**冻的东西，即子模块构造期冻结（方案 3 LoRA 基座、方案 5 `mask_token`）。今天两者都因名字含 `patch_embed`/`distill` 而侥幸未被解冻。

默认值见 `config/experiment/dl3dv.yaml:23-27`（`freeze_backbone: false` + `freeze_module: patch_embed`）。

**处置：整节待删。** 本方案是方案 1 的严格功能子集（唯一的非子集部分就是上面那个解冻 bug，且今天零影响），已判定彻底清除、6 份 config 迁到 `freeze_keywords`。见 `.scratch/freeze-contract/issues/02-reconcile-six-schemes.md`。

---

## 方案 3：LoRA —— 冻基座 + 只训低秩旁路

`src/model/segvggt/layers/lora.py:93-147`：

```python
114:        self.linear = linear
123:        # Freeze the original linear layer
124:        for param in self.linear.parameters():
125:            param.requires_grad = False
137:        return self.linear(x) + self.lora(x)
```

`lora_B` 零初始化（`:66`），保证初始输出与原模型一致。

注入时机在 `Aggregator._apply_lora`（`src/model/segvggt/models/aggregator.py:306-444`），只作用于 `frame_blocks` / `global_blocks` 的 attention（qkv / proj）和可选 MLP（fc1/fc2），参数由 `config/model/encoder/segvggt.yaml:31-37` 控制（rank 32 / alpha 32）。

⚠️ **注意**：LoRA 只冻住被它包裹的 attention 基座（约 202M），同一 block 里的 MLP / LayerNorm（约 403M）仍全量可训——这是 vendored 官方代码的行为（`docs/repo_knowledge.md:120`）。想连 LoRA 一起冻死，要额外靠方案 1 的 `frame_blocks` / `global_blocks` 关键词（`config/experiment/segvggt_finetune_agnostic.yaml:64-67` 正是这么做）。

---

## 方案 4：冻结教师/参考网络 + `torch.no_grad()` + CPU 卸载

这一档是"连加权重的常驻显存都省掉"，分三处：

| 位置 | 做法 |
|---|---|
| `src/model/arch/anysplat.py:144-155` + `365-429` | deepcopy 出 `distill_aggregator/camera_head/depth_head`，`requires_grad=False` **且** `param.data = param.data.cpu()`；前向时先 `.to(device)`，在 `torch.no_grad()` + bf16 autocast 内跑，跑完搬回 CPU 并 `del` 中间量 |
| `src/model/wrapper/segvggt_wrapper.py:180-204` | `@torch.no_grad()` 装饰的 `_geo_target_from_teacher`，懒加载 `VGGT.from_pretrained(...).eval()` + `requires_grad_(False)`；加载失败则 warn 并让 geo loss 归 0 而不是崩 |
| `src/loss/loss_depth.py:52-57` | `LossDepth` 内的 DepthAnything 全参 `requires_grad=False` |

代价是每次 forward 有 H2D 拷贝开销，换来的是这套教师权重平时完全不占显存。

---

## 方案 5：构造期冻单个 token / 缓冲

`src/model/segvggt/models/aggregator.py:302-304` 与 vendored 的 `src/model/vggt/models/aggregator.py:183-185`：

```python
302:            # Disable gradient updates for mask token
303:            if hasattr(self.patch_embed, "mask_token"):
304:                self.patch_embed.mask_token.requires_grad_(False)
```

---

## 方案 6（仅推理期，非训练冻结）：`model.eval()` + 全参 `requires_grad=False`

`src/model/wrapper/anysplat_wrapper.py:279-282`、`src/eval_pose.py:204-206`、`scripts/instseg_infer.py:396-398`、`scripts/trace_instance_to_gaussians.py:170-172`。注意 `_test_step_align` 里是 `torch.set_grad_enabled(True)` 包住的测试时位姿优化，属于例外。