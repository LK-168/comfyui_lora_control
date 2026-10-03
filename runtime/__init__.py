"""运行时（bypass）LoRA 支撑模块。

提供「不把 LoRA 权重融合进模型、而是在每层 forward 里算低秩分支再加回去」的能力，
并在此基础上支持 per-sample alpha（cond / uncond 用不同强度）。

复用了 ComfyUI 内核 ``comfy.weight_adapter`` 已有的 ``h()``（加法分量），
只把「按样本缩放」这一层换成我们自己的实现。

数值等价性验证见 ``tests/verify_bypass_lora.py``。
"""

from .alpha_context import ALPHA_WRAPPER_KEY, current_alpha, register_alpha_wrapper
from .bypass import (
    INJECTION_KEY,
    PerSampleBypassHook,
    RuntimeLoraManager,
    is_runtime_safe,
)

__all__ = [
    "ALPHA_WRAPPER_KEY",
    "INJECTION_KEY",
    "PerSampleBypassHook",
    "RuntimeLoraManager",
    "current_alpha",
    "is_runtime_safe",
    "register_alpha_wrapper",
]
