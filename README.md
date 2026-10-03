# lk_lora_stack

自定义 LoRA 加载器。核心能力：**不把 LoRA 融合进权重**，而是在推理时现算低秩分支再相加，
从而支持给 cond / uncond 分别指定强度（per-sample alpha），且仍只做一次 batch 前向。

> 详细设计、内核调研与验证记录见 [`doc.md`](./doc.md)。

## 节点

| 节点 | ID | 说明 |
|---|---|---|
| LK Lora Loader | `LKLoraLoader` | 加载单个 LoRA，支持 fold / runtime 两种模式 |

- 分类：`lk_lora_stack`
- 输出：`model`、`clip`
- 行为：`strength_model == strength_clip == 0` 时直接透传；LoRA 文件缺失时抛出可读的报错。

### 参数

| 参数 | 说明 |
|---|---|
| `lora_name` | LoRA 文件（combo 自带关键字过滤） |
| `strength_model` / `strength_clip` | 整体强度 |
| `mode` | `fold (native)`（默认，最快）/ `runtime (bypass, per-branch alpha)` |
| `alpha_cond` | **仅 runtime 生效**，cond 分支额外系数，实际强度 = `strength_model × alpha_cond` |
| `alpha_uncond` | 同上，uncond 分支。**设 0 = positive 用 LoRA、negative 不用** |

### 注意事项

- `alpha_cond == alpha_uncond` 时会自动退回折叠模式（结果完全一致，且零额外开销）。
- 只有满足 $bypass(f)(x) = f(x) + h(x)$ 的 LoRA 才能走 runtime 路径。
  **DoRA(`dora_scale`)、reshape 类、tucker(`lora_mid`)、LoHa/LoKr/OFT/BOFT/GLoRA
  会自动回退成折叠**，并在日志里说明。
- CLIP 侧一律折叠：文本编码在扩散循环之外，per-sample alpha 没有意义。
- 收益取决于场景：访存受限时接近 2×，计算受限时约 −3%/LoRA。

## 目录结构

```
lk_lora_stack/
├─ __init__.py
├─ runtime/
│  ├─ alpha_context.py   # cond_or_uncond -> per-sample alpha（contextvar + APPLY_MODEL wrapper）
│  └─ bypass.py          # PerSampleBypassHook / RuntimeLoraManager / is_runtime_safe
├─ nodes/
│  ├─ __init__.py        # 节点聚合（新增节点在这里登记）
│  └─ lora_loader.py     # LK Lora Loader
├─ tests/
│  ├─ verify_bypass_lora.py   # 数值等价性：bypass vs folding
│  └─ verify_plumbing.py      # 端到端接线：injection / wrapper / 生命周期
└─ doc.md
```

## 跑验证

必须用 ComfyUI 自己的环境（系统默认 `python` 没有 torch）：

```powershell
D:\CondaEnv\comfyui_py312\python.exe custom_nodes\lk_lora_stack\tests\verify_bypass_lora.py
D:\CondaEnv\comfyui_py312\python.exe custom_nodes\lk_lora_stack\tests\verify_plumbing.py
```

## 后续方向

- 前端 JS：选 `runtime` 时才显示 `alpha_cond` / `alpha_uncond`
- 真实模型端到端计时：runtime(α=1,0) vs 折叠 vs hooks 非对称
- 多槽位 LoRA Stack（参考 rgthree-comfy 的 `FlexibleOptionalInputType` + 动态 widget 路线）
- 支持 LoHa / LoKr（需先补数值验证）

## 安装

把 `lk_lora_stack` 放到 `ComfyUI/custom_nodes/` 下，重启 ComfyUI 即可。
