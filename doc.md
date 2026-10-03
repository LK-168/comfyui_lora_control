# lk_lora_stack 设计与验证文档

> 目标：做一个「不把 LoRA 融合进权重、而是推理时现算低秩分支再相加」的加载器，
> 并在此基础上支持 **per-sample alpha**（cond / uncond 用不同强度），
> 同时**不修改 ComfyUI 内核代码**。

---

## 0. TL;DR

- 内核 `comfy/weight_adapter` **本来就有** bypass（运行时）LoRA 机制，只是没暴露给采样。
- 它缺的只有三样：per-sample alpha、`cond_or_uncond` 的接线、DoRA/reshape 的 fallback。
- 我们全部在插件内实现，**内核零改动**。
- 数值与接线验证已全部通过（见 §4）。

---

## 1. 环境

| 项 | 值 |
|---|---|
| ComfyUI 根目录 | `D:\comfyui\Comfyui` |
| 插件目录 | `D:\comfyui\Comfyui\custom_nodes\lk_lora_stack` |
| Python | `D:\CondaEnv\comfyui_py312\python.exe`（torch 2.11.0+cu128） |
| 启动脚本 | `D:\comfyui\run_nvidia_gpu.bat`（`conda activate comfyui_py312`） |

> 跑任何测试都必须用 `comfyui_py312`，系统默认的 `python` 没有 torch。

---

## 2. 内核调研结论（事实，均带出处）

### 2.1 采样调用链

```
nodes.KSampler.sample()                     nodes.py:1623
  → common_ksampler()                       nodes.py:1572
    → comfy.sample.sample()                 comfy/sample.py:79
      → comfy.samplers.KSampler(...).sample()
        → CFGGuider.predict_noise()         comfy/samplers.py:1217
          → sampling_function()             comfy/samplers.py:609
            → calc_cond_batch([cond, uncond])
              → _calc_cond_batch()          comfy/samplers.py:222
```

### 2.2 cond / uncond 是**合批**的

`_calc_cond_batch` 把 cond 和 uncond 拼成一个 batch，一次 `apply_model` 算完
（`comfy/samplers.py:303-327`）：

```python
batch_chunks = len(cond_or_uncond)
input_x = torch.cat(input_x)                     # ← 合成一个 batch
output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
transformer_options["cond_or_uncond"] = cond_or_uncond[:]   # ← 每段属于 cond(0)/uncond(1)
```

要点：合批**不减少 FLOPs**（batch 2N 的行数不变），收益来自减少 kernel launch、
提升占用率、以及**权重只流式读取一遍**。所以计算受限时约 1.0–1.3×，
访存受限（小 token 数）时才可能接近 2×。

### 2.3 几个会影响 α 索引的边界

| 情况 | 后果 | 出处 |
|---|---|---|
| `cfg == 1.0` | `uncond_ = None`，uncond 根本不进 batch | `samplers.py:610` |
| 显存不足 | `to_batch` 分块，`apply_model` 被多次调用，每次 batch 组成不同 | `samplers.py:265-283` |
| 不同 hook | 按 `HookGroup` 分桶，**各跑各的前向** | `samplers.py:249-258` |
| 区域条件 | 一个 cond index 可对应多个 patch entry | `samplers.py:33` |

⇒ alpha 向量必须在**每次 `apply_model` 调用内**按当时的 `cond_or_uncond` 重建。

### 2.4 LoRA 在 ComfyUI 里是**权重级**的

`comfy/model_patcher.py:842`：

```python
current_patches.append((strength_patch, patches[k], strength_model, offset, function))
self.patches[key] = current_patches                      # 按 weight key 存
```

`comfy/lora.py:451 calculate_weight()` 把补丁折叠进权重张量：

$$W' = W + \alpha BA,\qquad y = W'x$$

**折叠是一次性的，运行时零额外开销。** 这就是「对称 LoRA 已经最优」的原因。

### 2.5 内核已有 bypass（运行时）系统 ⭐

这套东西原本只给训练节点用（`comfy_extras/nodes_train.py` 的 `bypass_mode`）。

| 位置 | 内容 |
|---|---|
| `weight_adapter/base.py:121` | `bypass_forward(org_forward, x)` → `g(f(x) + h(x))` |
| `weight_adapter/base.py:78` | `h(x, base_out)` = **加法分量**，即 $BAx$ |
| `weight_adapter/base.py:28` | `multiplier: float = 1.0` |
| `weight_adapter/lora.py:288` | `LoRAAdapter.h()`：`F.linear(F.linear(x, down), up) * scale` |
| `weight_adapter/bypass.py:107` | `BypassForwardHook`，inject 时替换 `module.forward` |
| `weight_adapter/bypass.py:107` | `get_module_type_info()` 推断 conv 参数 |
| `weight_adapter/bypass.py:284` | `BypassInjectionManager.add_adapter(key, adapter, strength)` |
| `weight_adapter/bypass.py:428` | `create_bypass_injections_from_patches(model, patches, strength)` |
| `model_patcher.py:1371` | **公开 API**：`model_patcher.set_injections(key, injections)` |
| `model_patcher.py:1419/1430` | `inject_model()` / `eject_model()`，由 `patch_model()`(1127) 触发 |

`lora.py:288` 的核心就是：

```python
hidden = op(x, down, **kw_dict)
out = op(hidden, up)
return out * scale            # scale = alpha/rank * multiplier
```

**它就是 $y = f(x) + h(x)$ 的现成实现。**

### 2.6 缺少的三样东西

1. `multiplier` 是**标量**，不能 per-sample。
2. 没有把 `cond_or_uncond` 接到 α 上。
3. `h()` **忽略** `dora_scale`(DoRA) 和 `reshape`（实测算错，见 §4.1）。

### 2.7 可用的注入点

| 注入点 | 说明 |
|---|---|
| `WrappersMP.APPLY_MODEL` | `model_base.py:208` 的 `apply_model` 外层包装，**不覆盖**别人的 unet wrapper |
| `set_model_unet_function_wrapper` | `samplers.py:333` 会显式传 `cond_or_uncond`，但会覆盖其他人 |
| `model_options["transformer_options"]` | `calc_cond_batch` 会把它 merge 进每次调用（`merge_nested_dicts` 对 list 是 **extend**） |

**坑**：`BaseModel.apply_model` 是**按位置**把 `transformer_options` 传给 wrapper 的
（第 6 个参数），不是 kwarg。wrapper 里要读 `args[5]`：

```python
.execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
```

---

## 3. 方案（路线 A：插件内实现，内核零改动）

复用内核的 `adapter.h()`，只替换「按样本缩放」这一层：

$$y_i = f(x_i) + s\cdot\alpha_i\cdot h(x_i)$$

```
lk_lora_stack/
├─ __init__.py
├─ runtime/
│  ├─ __init__.py            # 对外导出
│  ├─ alpha_context.py       # contextvar + APPLY_MODEL wrapper（cond_or_uncond -> alpha）
│  └─ bypass.py              # PerSampleBypassHook / RuntimeLoraManager / is_runtime_safe
├─ nodes/
│  ├─ __init__.py
│  └─ lora_loader.py         # LKLoraLoader（新增 mode / alpha_cond / alpha_uncond）
├─ tests/
│  ├─ verify_bypass_lora.py  # 数值等价性
│  └─ verify_plumbing.py     # 端到端接线
├─ README.md
└─ doc.md                    # 本文档
```

### 3.1 节点参数

| 参数 | 说明 |
|---|---|
| `model` / `clip` / `lora_name` | 同原生 |
| `strength_model` / `strength_clip` | 整体强度 |
| `mode` | `fold (native)`（默认）/ `runtime (bypass, per-branch alpha)` |
| `alpha_cond` | **仅 runtime 生效**，cond 分支额外系数，实际 = `strength_model × alpha_cond` |
| `alpha_uncond` | 同上，uncond 分支。设 0 = 「positive 用 LoRA、negative 不用」 |

### 3.2 关键设计决策

1. **`alpha_cond == alpha_uncond` 时自动退回折叠**（结果完全一致，省掉运行时开销）。
2. **不支持的 LoRA 类型自动 fallback 到折叠**，保证结果正确，只在日志里警告。
3. **CLIP 侧一律折叠**：文本编码在扩散循环之外，per-sample alpha 无意义。
4. **用 `APPLY_MODEL` wrapper 而不是 `set_model_unet_function_wrapper`**，避免覆盖 NAG/PAG 等节点。
5. **adapter 权重按输入张量的 device/dtype 惰性迁移**，不用 `module.weight.device`：
   lowvram 下权重可能还在 CPU，而真正的矩阵乘在 GPU 上。

---

## 4. 验证结果

### 4.1 数值等价性验证 `tests/verify_bypass_lora.py`

测试 LoRA：`models/loras/Anima/style/tianliang_duohe_fangdongye_anima_v1.safetensors`
测试 key：`lora_unet_blocks_0_cross_attn_k_proj`（out=2048, in=1024, rank=16, alpha=16）
覆盖率：普通 280 / DoRA 0 / reshape 0 / tucker 0

| 用例 | 结果 | 误差 |
|---|---|---|
| Linear：bypass == folding | ✅ PASS | max_abs 8.011e-5，rel 5.976e-7 |
| per-sample α == 两张折叠模型的拼接 | ✅ PASS | rel 5.973e-7 |
| **α=0 的行严格等于 base（零 LoRA 泄漏）** | ✅ PASS | max_abs **0.000e+0** |
| 分数 α 线性插值 | ✅ PASS | 4.196e-5 |
| Conv1×1（真实 LoRA） | ✅ PASS | 5.722e-6 |
| synth conv 3×3 stride1 | ✅ PASS | 2.861e-6 |
| synth conv 3×3 stride2 | ✅ PASS | 2.384e-6 |
| synth conv 5×5 stride1 | ✅ PASS | 5.722e-6 |
| synth conv 1×1 | ✅ PASS | 4.768e-7 |
| synth conv 3×3 alpha=8.0 | ✅ PASS | 3.338e-6 |
| synth linear alpha=None | ✅ PASS | 3.576e-7 |
| synth linear alpha=8.0 | ✅ PASS | 4.768e-7 |
| **DoRA (`dora_scale`)** | ❌ **算错 → 必须 fallback** | rel **5.290e-1** |
| **reshape 类** | ❌ **算错 → 必须 fallback** | rel **1.602e+0** |

> 误差量级 ~1e-7 属于 fp32 累加误差，可忽略。

**结论**：`h()` 走的是 $y = f(x) + \alpha BAx$，与折叠在数值上等价；
DoRA 与 reshape 是内核 bypass 自身的缺口，必须检测并退回折叠。

### 4.2 端到端接线验证 `tests/verify_plumbing.py`

用真实 `ModelPatcher` + 真实 `PatcherInjection` / `APPLY_MODEL` wrapper 机制，
在小 `nn.Module` 上跑完整链路（不加载大模型）。

| 用例 | 结果 |
|---|---|
| `is_runtime_safe(普通 LoRA) == True` | ✅ |
| `is_runtime_safe(DoRA) == False` | ✅ |
| `is_runtime_safe(reshape) == False` | ✅ |
| `create_injections` 没有跳过任何层 | ✅ |
| `APPLY_MODEL` wrapper 已被收集 | ✅ wrappers=1 |
| 未 inject 时前向等于原始模型 | ✅ |
| `inject_model()` 后 `is_injected == True` | ✅ |
| **cond 行(α=1) == 折叠模型** | ✅ max_abs 1.788e-7 |
| **uncond 行(α=0) == 原始模型（无泄漏）** | ✅ max_abs 2.980e-8 |
| cond / uncond 确实产生差异（α 生效） | ✅ diff 9.682e-1 |
| α=0.5 == 半强度折叠 | ✅ max_abs 7.451e-8 |
| 无 `cond_or_uncond` 时退化为整体缩放 | ✅ max_abs 2.384e-7 |
| `eject_model()` 后前向完全恢复 | ✅ max_abs 0.0 |
| `eject_model()` 后 `is_injected == False` | ✅ |

**结论**：injection / wrapper / α 通道 / 生命周期全部正确。

### 4.3 怎么跑

```powershell
D:\CondaEnv\comfyui_py312\python.exe d:\comfyui\Comfyui\custom_nodes\lk_lora_stack\tests\verify_bypass_lora.py
D:\CondaEnv\comfyui_py312\python.exe d:\comfyui\Comfyui\custom_nodes\lk_lora_stack\tests\verify_plumbing.py
```

---

## 5. 踩过的坑（重要）

1. **`transformer_options` 是位置参数**。`BaseModel.apply_model` 调
   `.execute(x, t, c_concat, c_crossattn, control, transformer_options)`，
   wrapper 里必须读 `args[5]`，读 `kwargs` 拿不到。
   （最初写成 `kwargs.get("transformer_options")`，靠端到端测试才暴露。）

2. **不能按 `module.weight.device` 放 adapter 权重**。lowvram 下权重留在 CPU、
   计算却在 GPU。最终方案是在 `forward` 里按 `x.device`/`x.dtype` 惰性迁移并缓存。

3. **α 不是输出的线性插值**。α 逐层作用在低秩分支上，多层串联后存在
   $\Delta_0\Delta_1$ 交叉项，所以 $\alpha=0.5$ 的结果**不等于**
   `base + 0.5*(full - base)`，正确参考是「半强度折叠后的权重」。
   （测试里先用错公式，误报为 FAIL，实为期望写错。）

4. **测试隔离**：`ModelPatcher.model` 共享同一个 `nn.Module`，
   在同一对象上建两个 patcher 会把两组 hook **串起来**。
   多场景测试必须各自 deepcopy / 重建模块。

5. **`h()` 忽略 `dora_scale` 和 `reshape`**。这不是我们的 bug，是内核 bypass
   的既有缺口，必须检测并 fallback。

6. **OFT / BOFT / GLoRA 不能走这条路**：
   - OFT/BOFT 的 `multiplier` 做的是 `r = r·m + (1-m)·I` 插值，不是加性缩放；
   - GLoRA 需要覆写 `bypass_forward`，直接调 `h()` 不正确。

---

## 6. 性能预期

设维度 $d$、秩 $r$、单分支 token 数 $N$（batch = $2N$）。

| 方案 | apply_model 次数 | 权重流式读取 | 每层 FLOPs |
|---|---|---|---|
| A. 对称 LoRA（原生折叠） | **1** | $d^2$ | $2Nd^2$ |
| B. 非对称 LoRA（hooks 机制） | **2** | $2d^2$ | $2Nd^2$ |
| C. 本插件 runtime + per-sample α | **1** | $d^2(1+\frac{2r}{d})$ | $2Nd^2(1+\frac{2r}{d})$ |

额外开销 $\frac{2r}{d}$：$d=1024,\ r=16$ 时单 LoRA **+3.1%**，10 个 LoRA **+31%**。

| 场景 | 预期 |
|---|---|
| 计算受限（大分辨率 / 视频） | 约 **−3%/LoRA**，基本打平 |
| 访存受限（小 token 数、小 batch） | 可能接近 **2×** |
| **batch 内每张图 α 不同** | **碾压性获胜**（方案 B 要退化成 N 次遍历） |

⚠️ 尚未做真实模型的端到端计时，这是下一步要补的。

---

## 7. 路线对比：为什么不改内核

| | 路线 A（本方案） | 路线 B（改内核） |
|---|---|---|
| 代码量 | **560–890 行，全在插件内** | 仅 bypass 路径 250–450 行（散在 8 个内核文件）；含 folding 全路径要 900–2000 行 |
| 主色面 | `runtime/` + 节点 | `weight_adapter/*`、`model_patcher`、`samplers`、`ops.py`（29 个 `forward` 变体 + 量化/fp8/compile） |
| 上游同步 | 无影响 | 每次更新都冲突，需上游 PR 才能持续 |
| 风险 | 边界情况靠 fallback 兜底 | 波及量化 / lowvram / torch.compile |

**结论：路线 A 用 1/2 ~ 1/4 的代码量达成同样目标。**

---

## 8. 待办 / 后续方向

- [ ] 真实模型端到端计时：runtime(α=1,0) vs 折叠 vs hooks 非对称，出实测数字
- [ ] 前端 JS：选 `runtime` 时才显示 `alpha_cond` / `alpha_uncond`
- [ ] 多 LoRA 叠加（stack）时合并 hook，避免层层包裹
- [ ] 考虑支持 LoHa / LoKr（需先补数值验证）
- [ ] 若 DoRA 使用率高，可在我们侧实现 `dora_scale` 的 per-sample 版本
