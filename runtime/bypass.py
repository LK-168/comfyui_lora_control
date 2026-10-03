"""运行时（bypass）LoRA 的核心机制。

内核里的等价关系（见 ``comfy/weight_adapter/base.py``）::

    bypass(f)(x) = g(f(x) + h(x))

其中 ``h(x)`` 就是 LoRA 的加法分量。本模块把 ``g`` 当恒等（只支持这类适配器），
并把整体缩放换成 **per-sample** 版本::

    y_i = f(x_i) + strength * alpha_i * h(x_i)

于是 ``alpha_cond = 1, alpha_uncond = 0`` 就表示「positive 用 LoRA，negative 不用」，
而且 cond / uncond 仍然共享同一次 batch 前向。
"""

import logging

import torch

import comfy.weight_adapter
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter import LoRAAdapter
from comfy.weight_adapter.bypass import get_module_type_info

from .alpha_context import current_alpha

LOG = logging.getLogger("lk_lora_stack")

#: ``model.injections`` 里用的 key。
INJECTION_KEY = "lk_lora_stack.runtime_lora"

#: 已知 ``bypass(f)(x) = f(x) + h(x)`` 成立的适配器类型。
#:
#: - ``OFT`` / ``BOFT`` 走的是输出变换 ``g()``，不能用「加性 + 缩放」表达；
#: - ``GLoRA`` 需要覆写 ``bypass_forward``，直接调 ``h()`` 不正确；
#: - ``LoHa`` / ``LoKr`` 理论上可行，但尚未做过数值验证，暂不启用。
RUNTIME_SAFE_ADAPTERS = (LoRAAdapter,)


def is_runtime_safe(adapter):
    """判断某个适配器能否安全地走 per-sample bypass 路径。"""
    if not isinstance(adapter, RUNTIME_SAFE_ADAPTERS):
        return False

    if isinstance(adapter, LoRAAdapter):
        # adapter.weights == (up, down, alpha, mid, dora_scale, reshape)
        _, _, _, mid, dora_scale, reshape = adapter.weights
        # tucker(LoCon mid)：h() 的 conv 分支会算错。
        if mid is not None:
            return False
        # DoRA：h() 完全忽略 dora_scale，实测相对误差 ~53%（见验证脚本）。
        if dora_scale is not None:
            return False
        # reshape 类：h() 同样忽略，实测相对误差 ~160%。
        if reshape is not None:
            return False

    return True


def _resolve_module(root, key):
    """按点分 key 取子模块，支持 ``layers.0.attn`` 这类数字下标。"""
    module = root
    for part in key.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


class PerSampleBypassHook:
    """把某个 module 的 ``forward`` 换成 ``f(x) + strength * alpha * h(x)``。"""

    def __init__(self, module, adapter, strength=1.0):
        self.module = module
        self.adapter = adapter
        self.strength = float(strength)
        self.original_forward = None
        self._weight_key = None

        info = get_module_type_info(module)
        # h() 内部会乘 multiplier；缩放由我们自己做，所以固定成 1。
        adapter.multiplier = 1.0
        adapter.is_conv = info["is_conv"]
        adapter.conv_dim = info["conv_dim"]
        adapter.kernel_size = info["kernel_size"]
        adapter.in_channels = info["in_channels"]
        adapter.out_channels = info["out_channels"]
        adapter.kw_dict = (
            {
                "stride": info["stride"],
                "padding": info["padding"],
                "dilation": info["dilation"],
                "groups": info["groups"],
            }
            if info["is_conv"]
            else {}
        )

    # ------------------------------------------------------------------ 生命周期
    def inject(self):
        if self.original_forward is not None:
            return
        self.original_forward = self.module.forward
        self.module.forward = self._forward

    def eject(self):
        if self.original_forward is None:
            return
        self.module.forward = self.original_forward
        self.original_forward = None

    def _ensure_weights(self, device, dtype):
        """把低秩分支的权重对齐到实际计算设备 / dtype（设备没变则零开销）。

        不能用 ``module.weight.device``：lowvram 下权重可能还留在 CPU，
        而真正的矩阵乘发生在 GPU 上。
        """
        key = (device, dtype)
        if self._weight_key == key:
            return
        self.adapter.weights = tuple(
            t.to(device=device, dtype=dtype) if torch.is_tensor(t) and t.is_floating_point() else t
            for t in self.adapter.weights
        )
        self._weight_key = key

    # ------------------------------------------------------------------ 前向
    def _forward(self, x, *args, **kwargs):
        base_out = self.original_forward(x, *args, **kwargs)

        self._ensure_weights(x.device, x.dtype)
        alpha = current_alpha()
        h_out = self.adapter.h(x, base_out)

        if alpha is None:
            # 没有 per-sample 信息（或所有样本 alpha==1）：整体缩放。
            return base_out + h_out * self.strength

        h_out = h_out * self.strength
        # batch 维度对不上时保守退回整体缩放，宁可结果弱一点也不出错。
        if alpha.shape[0] != h_out.shape[0]:
            return base_out + h_out

        view = (-1,) + (1,) * (h_out.dim() - 1)
        return base_out + h_out * alpha.view(view)


class RuntimeLoraManager:
    """收集 adapter，并生成一个可以把所有 hook 注入/弹出的 ``PatcherInjection``。"""

    def __init__(self, strength=1.0):
        self.strength = float(strength)
        self.entries = {}

    def add(self, key, adapter):
        """``key`` 是模型的 state_dict key（形如 ``xxx.weight``）。"""
        self.entries[key] = adapter

    def __len__(self):
        return len(self.entries)

    def create_injections(self, model):
        """返回 ``(injections, skipped)``；``skipped`` 是没能挂上 hook 的条目。"""
        hooks = []
        skipped = []

        for key, adapter in self.entries.items():
            module_key = key[: -len(".weight")] if key.endswith(".weight") else key
            try:
                module = _resolve_module(model, module_key)
            except Exception as exc:  # noqa: BLE001
                skipped.append((key, f"找不到模块: {exc}"))
                continue

            if not hasattr(module, "weight"):
                skipped.append((key, "目标模块没有 weight"))
                continue

            hooks.append(PerSampleBypassHook(module, adapter, self.strength))

        def inject_all(_model_patcher):
            for hook in hooks:
                hook.inject()
            LOG.info("[lk_lora_stack] 已注入 %d 个运行时 LoRA hook", len(hooks))

        def eject_all(_model_patcher):
            for hook in hooks:
                hook.eject()

        return [PatcherInjection(inject=inject_all, eject=eject_all)], skipped
