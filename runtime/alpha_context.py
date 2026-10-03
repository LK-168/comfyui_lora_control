"""per-sample alpha 的传递通道。

采样时 ComfyUI 会把 cond / uncond 合批成一次前向，并通过
``transformer_options["cond_or_uncond"]`` 标注每个 batch 位置是属于
cond(0) 还是 uncond(1)（见 ``comfy/samplers.py`` 的 ``_calc_cond_batch``）。

我们在 ``apply_model`` 外面包一层 wrapper：每次前向开始时，把 alpha 向量
按 ``cond_or_uncond`` 展开并写进 contextvar；bypass hook 在每层 forward 里读它。
"""

import contextvars

import torch

import comfy.patcher_extension

#: 注册在 model_options 里的 wrapper key，方便去重。
ALPHA_WRAPPER_KEY = "lk_lora_stack.runtime_alpha"

_CURRENT_ALPHA = contextvars.ContextVar("lk_lora_alpha", default=None)


def current_alpha():
    """返回当前前向的 per-sample alpha 张量 ``[B]``；没有则返回 ``None``。

    ``None`` 表示「按整体强度应用 LoRA」（等价于所有样本 alpha == 1）。
    """
    return _CURRENT_ALPHA.get()


def _build_alpha(cond_or_uncond, alpha_cond, alpha_uncond, device):
    if not cond_or_uncond:
        return None
    values = [alpha_uncond if v == 1 else alpha_cond for v in cond_or_uncond]
    # 全 1 时无需逐样本缩放，走快路径。
    if all(v == 1.0 for v in values):
        return None
    return torch.tensor(values, dtype=torch.float32, device=device)


def register_alpha_wrapper(model_options, alpha_cond, alpha_uncond):
    """在 ``model_options`` 上挂一个 ``apply_model`` wrapper，用于注入 alpha。

    用 wrapper 而不是 ``set_model_unet_function_wrapper``，是为了不覆盖其他节点
    （NAG、PAG 之类）已经设置的 unet wrapper。
    """

    def _alpha_wrapper(executor, *args, **kwargs):
        # 注意：``BaseModel.apply_model`` 是按位置把 transformer_options 传进来的
        # （executor.execute(x, t, c_concat, c_crossattn, control, transformer_options)），
        # 所以这里要同时兼容 args[5] 和 kwarg 两种形态。
        transformer_options = args[5] if len(args) > 5 and isinstance(args[5], dict) else None
        if transformer_options is None:
            transformer_options = kwargs.get("transformer_options") or {}
        cond_or_uncond = transformer_options.get("cond_or_uncond", None)

        x = args[0] if args else None
        alpha = None
        if torch.is_tensor(x):
            alpha = _build_alpha(cond_or_uncond, alpha_cond, alpha_uncond, x.device)

        token = _CURRENT_ALPHA.set(alpha)
        try:
            return executor(*args, **kwargs)
        finally:
            _CURRENT_ALPHA.reset(token)

    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL,
        ALPHA_WRAPPER_KEY,
        _alpha_wrapper,
        model_options,
        is_model_options=True,
    )
