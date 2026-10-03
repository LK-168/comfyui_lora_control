"""LK LoRA 加载器。

两种模式：

``fold (native)``
    走内核原生路径，把 LoRA 融合进权重（$W' = W + \\alpha BA$）。
    运行时零额外开销，是默认且最快的选择。

``runtime (bypass)``
    不融合，在每层 forward 里现算低秩分支再加回去（$y = f(x) + s\\cdot\\alpha\\,h(x)$）。
    代价是每步都要多算一次低秩分支，好处是可以给 cond / uncond
    分别指定 alpha，实现「positive 用 LoRA、negative 不用」这类玩法，
    同时仍然只做一次 batch 前向。

    注意：只有满足 $bypass(f)(x) = f(x) + h(x)$ 的 LoRA 才能走这条路。
    DoRA(``dora_scale``)、reshape 类、tucker(``lora_mid``)、LoHa/LoKr/OFT/BOFT
    会自动回退成折叠，并在日志里说明。

参考：内核 ``comfy/weight_adapter`` 的 bypass 模式（原本只给训练节点用）。
"""

import logging

import folder_paths

import comfy.lora
import comfy.utils
from nodes import LoraLoader

from ..runtime import (
    INJECTION_KEY,
    RuntimeLoraManager,
    is_runtime_safe,
    register_alpha_wrapper,
)

LOG = logging.getLogger("lk_lora_stack")

MODE_FOLD = "fold (native)"
MODE_RUNTIME = "runtime (bypass, per-branch alpha)"


class LKLoraLoader:
    """加载单个 LoRA，可分别指定 model 与 clip 的强度。"""

    NAME = "LK Lora Loader"
    CATEGORY = "lk_lora_stack"

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name, missing-function-docstring
        strength = {"min": -100.0, "max": 100.0, "step": 0.01}
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, **strength, "tooltip": "LoRA 对 model 的整体强度。"},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, **strength, "tooltip": "LoRA 对 clip 的整体强度。"},
                ),
                "mode": (
                    [MODE_FOLD, MODE_RUNTIME],
                    {
                        "default": MODE_FOLD,
                        "tooltip": (
                            "fold: 融合进权重（最快）。\n"
                            "runtime: 不融合，推理时现算低秩分支再相加，"
                            "此时 alpha_cond / alpha_uncond 才生效。"
                        ),
                    },
                ),
                "alpha_cond": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        **strength,
                        "tooltip": (
                            "runtime 模式下 positive(cond) 分支的额外系数，"
                            "实际强度 = strength_model × alpha_cond。\n"
                            "fold 模式下该值被忽略。"
                        ),
                    },
                ),
                "alpha_uncond": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        **strength,
                        "tooltip": (
                            "runtime 模式下 negative(uncond) 分支的额外系数。\n"
                            "设成 0 就是「positive 用 LoRA、negative 完全不用」，"
                            "而采样仍然只做一次 batch 前向。"
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"

    def load_lora(
        self,
        model,
        clip,
        lora_name,
        strength_model,
        strength_clip,
        mode=MODE_FOLD,
        alpha_cond=1.0,
        alpha_uncond=1.0,
    ):
        """应用单个 LoRA；两个强度都为 0 时直接透传。"""
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        # 提前给出更友好的报错，避免用户在节点上选了已被删除的 LoRA 后一头雾水。
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise FileNotFoundError(f"[lk_lora_stack] 找不到 LoRA 文件: {lora_name!r}")

        runtime = mode == MODE_RUNTIME and strength_model != 0
        if runtime and alpha_cond == alpha_uncond:
            # 两个分支系数相同时结果与折叠完全一致，直接走原生路径，省掉运行时开销。
            LOG.info(
                "[lk_lora_stack] alpha_cond == alpha_uncond == %s，自动退回折叠模式", alpha_cond
            )
            runtime = False
            strength_model = strength_model * alpha_cond

        if not runtime:
            return LoraLoader().load_lora(
                model, clip, lora_name, strength_model, strength_clip
            )

        return self._load_runtime(
            model,
            clip,
            lora_name,
            lora_path,
            strength_model,
            strength_clip,
            alpha_cond,
            alpha_uncond,
        )

    # ------------------------------------------------------------------ runtime 模式
    def _load_runtime(
        self,
        model,
        clip,
        lora_name,
        lora_path,
        strength_model,
        strength_clip,
        alpha_cond,
        alpha_uncond,
    ):
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

        key_map = {}
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        patches = comfy.lora.load_lora(lora_sd, key_map, log_missing=False)

        model_sd_keys = set(model.model.state_dict().keys())
        runtime_patches, model_fold, clip_fold = {}, {}, {}
        for key, value in patches.items():
            if key not in model_sd_keys:
                clip_fold[key] = value
            elif is_runtime_safe(value):
                runtime_patches[key] = value
            else:
                model_fold[key] = value

        new_model = model.clone()
        new_clip = clip.clone()

        # CLIP 侧在扩散循环之外，per-sample alpha 没有意义，直接折叠。
        if clip_fold and strength_clip != 0:
            new_clip.patcher.add_patches(clip_fold, strength_clip, strength_clip)

        # 不支持的 LoRA 类型退回折叠，保证结果正确。
        if model_fold and strength_model != 0:
            new_model.add_patches(model_fold, strength_model, strength_model)
            LOG.warning(
                "[lk_lora_stack] %s: %d/%d 个权重不兼容运行时模式，已回退为折叠",
                lora_name,
                len(model_fold),
                len(patches),
            )

        if not runtime_patches:
            LOG.warning("[lk_lora_stack] %s: 没有可运行时处理的权重，等同折叠模式", lora_name)
            return (new_model, new_clip)

        manager = RuntimeLoraManager(strength=strength_model)
        for key, value in runtime_patches.items():
            manager.add(key, value)
        injections, skipped = manager.create_injections(new_model.model)
        if skipped:
            LOG.warning(
                "[lk_lora_stack] %s: %d 个权重没能挂上 hook（已跳过）",
                lora_name,
                len(skipped),
            )

        new_model.set_injections(INJECTION_KEY, injections)
        register_alpha_wrapper(new_model.model_options, alpha_cond, alpha_uncond)
        LOG.info(
            "[lk_lora_stack] %s: 运行时 LoRA 已启用（%d 层，alpha_cond=%s，alpha_uncond=%s）",
            lora_name,
            len(runtime_patches),
            alpha_cond,
            alpha_uncond,
        )
        return (new_model, new_clip)


NODE_CLASS_MAPPINGS = {
    "LKLoraLoader": LKLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LKLoraLoader": LKLoraLoader.NAME,
}
