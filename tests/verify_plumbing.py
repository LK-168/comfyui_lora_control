"""端到端接线验证：真实 ``ModelPatcher`` + 真实 injection / wrapper 机制。

不加载大模型，用一个极小的 ``nn.Module`` 冒充 UNet 的两层，走完整链路::

    RuntimeLoraManager.create_injections
      -> ModelPatcher.set_injections
      -> ModelPatcher.inject_model
      -> register_alpha_wrapper（APPLY_MODEL wrapper）
      -> 按 BaseModel.apply_model 的方式触发前向

验证点：
  1. injection 真的替换了 module.forward
  2. alpha wrapper 能读到 ``transformer_options["cond_or_uncond"]``（位置参数形态）
  3. alpha=[1,0] 时，第 0 行 == 「折叠后的模型」，第 1 行 == 「原始模型」
  4. eject 之后前向完全恢复

跑法::

    D:\\CondaEnv\\comfyui_py312\\python.exe verify_plumbing.py
"""

import os
import sys
import traceback
from collections import OrderedDict

import torch
import torch.nn as nn

COMFY_ROOT = r"D:\comfyui\Comfyui"
CUSTOM_NODES = os.path.join(COMFY_ROOT, "custom_nodes")
for _p in (COMFY_ROOT, CUSTOM_NODES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import comfy.model_patcher  # noqa: E402
import comfy.patcher_extension as pe  # noqa: E402
import comfy.weight_adapter  # noqa: E402

import lk_lora_stack.runtime as lkrt  # noqa: E402

ATOL = 1e-5
FAILURES = []


def report(name, ok, detail=""):
    if not ok:
        FAILURES.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def max_err(a, b):
    return (a - b).abs().max().item()


# --------------------------------------------------------------------------------------
class TinyNet(nn.Module):
    """两个串联的 Linear，命名模仿 UNet 的 block 结构。"""

    def __init__(self, dim=16):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(OrderedDict([("proj", nn.Linear(dim, dim, bias=False))])) for _ in range(2)]
        )
        self.device = torch.device("cpu")  # ModelPatcher 会读取这个属性

    def forward(self, x):
        for block in self.blocks:
            x = block.proj(x)
        return x


def make_adapter(in_features, out_features, rank=4, alpha=None, dora=None, reshape=None):
    up = torch.randn(out_features, rank) * 0.3
    down = torch.randn(rank, in_features) * 0.3
    return comfy.weight_adapter.LoRAAdapter(
        set(), (up, down, alpha, None, dora, reshape)
    )


def delta_of(adapter, W):
    """用内核折叠路径算出 ΔW = W' - W。"""
    patches = [(1.0, adapter, 1.0, None, None)]
    import comfy.lora

    return comfy.lora.calculate_weight(patches, W.clone(), "dummy.weight") - W


# --------------------------------------------------------------------------------------
def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    dim = 16
    net = TinyNet(dim)
    W0 = net.blocks[0].proj.weight.detach().clone()
    W1 = net.blocks[1].proj.weight.detach().clone()

    # ---------------------------------------------------------------- is_runtime_safe
    plain = make_adapter(dim, dim)
    report("is_runtime_safe(普通 LoRA) == True", lkrt.is_runtime_safe(plain) is True)
    report(
        "is_runtime_safe(DoRA) == False",
        lkrt.is_runtime_safe(make_adapter(dim, dim, dora=torch.rand(dim) + 0.5)) is False,
    )
    report(
        "is_runtime_safe(reshape) == False",
        lkrt.is_runtime_safe(make_adapter(dim, dim, reshape=[dim, dim])) is False,
    )

    # ---------------------------------------------------------------- 建立 ModelPatcher
    mp = comfy.model_patcher.ModelPatcher(
        net, torch.device("cpu"), torch.device("cpu"), size=1
    )

    a0 = make_adapter(dim, dim)
    a1 = make_adapter(dim, dim)
    mgr = lkrt.RuntimeLoraManager(strength=1.0)
    mgr.add("blocks.0.proj.weight", a0)
    mgr.add("blocks.1.proj.weight", a1)

    injections, skipped = mgr.create_injections(mp.model)
    report("create_injections 没有跳过任何层", not skipped, f"skipped={skipped}")
    report("注入 hook 数量正确", len(injections) == 1)

    mp.set_injections(lkrt.INJECTION_KEY, injections)
    lkrt.register_alpha_wrapper(mp.model_options, alpha_cond=1.0, alpha_uncond=0.0)

    # ---------------------------------------------------------------- 构造前向执行器
    fake_apply = lambda x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kw: net(x)

    def run(cond_or_uncond, x, t):
        # 模拟 calc_cond_batch：把 model_options 里的 transformer_options 合并进来
        to = pe.copy_nested_dicts(mp.model_options["transformer_options"])
        to["cond_or_uncond"] = cond_or_uncond
        wrappers = pe.get_all_wrappers(pe.WrappersMP.APPLY_MODEL, to)
        executor = pe.WrapperExecutor.new_class_executor(fake_apply, net, wrappers)
        return executor.execute(x, t, None, None, None, to), len(wrappers)

    B = 2
    x = torch.randn(B, dim)
    t = torch.tensor([1.0, 1.0])

    base = net(x)
    out_before, nw = run([0, 1], x, t)
    report("APPLY_MODEL wrapper 已注册并被收集到", nw == 1, f"wrappers={nw}")
    report("未 inject 时前向等于原始模型", max_err(out_before, base) < ATOL)

    # ---------------------------------------------------------------- 注入
    mp.inject_model()
    report("inject_model 后 is_injected == True", mp.is_injected is True)

    out, _ = run([0, 1], x, t)

    # 参考：第 0 行 = 折叠后的模型；第 1 行 = 原始模型
    W0f = W0 + delta_of(a0, W0)
    W1f = W1 + delta_of(a1, W1)
    row0 = x[0:1] @ W0f.t() @ W1f.t()
    row1 = x[1:2] @ W0.t() @ W1.t()

    report(
        "cond 行(alpha=1) == 折叠模型",
        max_err(out[0:1], row0) < ATOL,
        f"max_abs={max_err(out[0:1], row0):.3e}",
    )
    report(
        "uncond 行(alpha=0) == 原始模型（无 LoRA 泄漏）",
        max_err(out[1:2], row1) < ATOL,
        f"max_abs={max_err(out[1:2], row1):.3e}",
    )
    report(
        "cond / uncond 确实产生了差异（说明 alpha 生效）",
        max_err(out[0:1], out[1:2]) > ATOL,
        f"diff={max_err(out[0:1], out[1:2]):.3e}",
    )

    # ---------------------------------------------------------------- 分数 alpha
    # 用一份独立的权重副本，避免两组 hook 叠在同一个 module 上互相干扰。
    net2 = TinyNet(dim)
    with torch.no_grad():
        net2.blocks[0].proj.weight.copy_(W0)
        net2.blocks[1].proj.weight.copy_(W1)

    mp2 = comfy.model_patcher.ModelPatcher(
        net2, torch.device("cpu"), torch.device("cpu"), size=1
    )
    mgr2 = lkrt.RuntimeLoraManager(strength=1.0)
    mgr2.add("blocks.0.proj.weight", a0)
    mgr2.add("blocks.1.proj.weight", a1)
    inj2, _ = mgr2.create_injections(mp2.model)
    mp2.set_injections(lkrt.INJECTION_KEY, inj2)
    lkrt.register_alpha_wrapper(mp2.model_options, alpha_cond=0.5, alpha_uncond=0.0)
    mp2.inject_model()

    to2 = pe.copy_nested_dicts(mp2.model_options["transformer_options"])
    to2["cond_or_uncond"] = [0, 1]
    wrappers2 = pe.get_all_wrappers(pe.WrappersMP.APPLY_MODEL, to2)

    def apply2(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kw):
        return net2(x)

    ex2 = pe.WrapperExecutor.new_class_executor(apply2, net2, wrappers2)
    out_half = ex2.execute(x, t, None, None, None, to2)

    # 注意：alpha 是逐层作用在低秩分支上的，两层串联后交叉项使结果并不是
    # 「输出层面的简单线性插值」，正确参考是「半强度折叠后的权重」。
    d0 = delta_of(a0, W0)
    d1 = delta_of(a1, W1)
    half_folded = x[0:1] @ (W0 + 0.5 * d0).t() @ (W1 + 0.5 * d1).t()
    report(
        "alpha=0.5 == 半强度折叠",
        max_err(out_half[0:1], half_folded) < ATOL,
        f"max_abs={max_err(out_half[0:1], half_folded):.3e}",
    )
    report(
        "alpha=0 行 == 原始模型（mp2）",
        max_err(out_half[1:2], x[1:2] @ W0.t() @ W1.t()) < ATOL,
        f"max_abs={max_err(out_half[1:2], x[1:2] @ W0.t() @ W1.t()):.3e}",
    )
    mp2.eject_model()

    # ---------------------------------------------------------------- 没有 cond_or_uncond
    out_non, _ = run(None, x, t)
    full = x @ W0f.t() @ W1f.t()
    report(
        "没有 cond_or_uncond 时退化为整体缩放（两行都带 LoRA）",
        max_err(out_non, full) < ATOL,
        f"max_abs={max_err(out_non, full):.3e}",
    )

    # ---------------------------------------------------------------- eject
    mp.eject_model()
    out_after, _ = run([0, 1], x, t)
    report("eject 后前向完全恢复", max_err(out_after, base) < ATOL, f"max_abs={max_err(out_after, base):.3e}")
    report("eject 后 is_injected == False", mp.is_injected is False)

    print("-" * 78)
    if FAILURES:
        print(f"结果: {len(FAILURES)} 项失败 -> {FAILURES}")
        sys.exit(1)
    print("结果: 全部通过")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
