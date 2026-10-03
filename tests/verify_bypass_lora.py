"""数值验证：内核 bypass（运行时相加）LoRA 与 折叠(folding) LoRA 是否等价。

目的：在动手写节点前，确认 ``comfy.weight_adapter`` 的 bypass 路径
（``LoRAAdapter.h()``）在数值上等于把 LoRA 折叠进权重后的结果。
并且验证「per-sample alpha」可以简单地用 ``base + h(x) * alpha`` 实现。

跑法（必须用 ComfyUI 自己的环境）::

    D:\\CondaEnv\\comfyui_py312\\python.exe verify_bypass_lora.py

参考：rgthree 无关，纯内核验证。
"""

import glob
import os
import sys
import traceback

import torch
import torch.nn.functional as F

COMFY_ROOT = r"D:\comfyui\Comfyui"
LORA_DIR = os.path.join(COMFY_ROOT, "models", "loras")

if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

import comfy.lora  # noqa: E402
import comfy.weight_adapter  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

ATOL = 1e-4
FAILURES = []


def report(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILURES.append(name)
    print(f"[{tag}] {name}" + (f"  -> {detail}" if detail else ""))


def dtype_stats(a, b):
    d = (a - b).abs()
    denom = b.abs().amax().clamp_min(1e-8)
    return d.amax().item(), (d.amax() / denom).item()


# --------------------------------------------------------------------------------------
# 1. 找一个尺寸最小的、包含 lora_up/down 的 LoRA 文件
# --------------------------------------------------------------------------------------
def pick_lora():
    cands = sorted(
        glob.glob(os.path.join(LORA_DIR, "**", "*.safetensors"), recursive=True),
        key=os.path.getsize,
    )
    for path in cands:
        try:
            sd = load_file(path)
        except Exception:
            continue
        ups = [k for k in sd if k.endswith(".lora_up.weight")]
        if ups:
            return path, sd, ups
    raise SystemExit("没有找到可用的 LoRA 文件")


def find_clean_key(sd, ups):
    """优先找一个不带 dora_scale / reshape / mid 的普通 LoRA key。"""
    for u in ups:
        x = u[: -len(".lora_up.weight")]
        if (x + ".lora_down.weight") not in sd:
            continue
        if x + ".dora_scale" in sd:
            continue
        if x + ".reshape_weight" in sd:
            continue
        if x + ".lora_mid.weight" in sd:
            continue
        return x
    return None


def build_adapter(sd, x):
    alpha = sd[x + ".alpha"].item() if (x + ".alpha") in sd else None
    dora = sd[x + ".dora_scale"] if (x + ".dora_scale") in sd else None
    adapter = comfy.weight_adapter.LoRAAdapter.load(x, sd, alpha, dora, set())
    return adapter, alpha


def folded(adapter, W):
    """用内核官方路径把 LoRA 折叠进权重，返回折叠后的权重。"""
    patches = [(1.0, adapter, 1.0, None, None)]
    return comfy.lora.calculate_weight(patches, W.clone(), "dummy.weight")


# --------------------------------------------------------------------------------------
# 2. Linear 路径
# --------------------------------------------------------------------------------------
def test_linear(adapter, label):
    up, down = adapter.weights[0], adapter.weights[1]
    out_f, in_f = up.shape[0], down.shape[1]
    torch.manual_seed(0)
    W = torch.randn(out_f, in_f, dtype=torch.float32)

    W_folded = folded(adapter, W)

    B = 4
    x = torch.randn(B, in_f, dtype=torch.float32)
    base = torch.nn.functional.linear(x, W)
    ref = torch.nn.functional.linear(x, W_folded)  # 折叠后的整批结果

    # 运行时（bypass）路径
    adapter.multiplier = 1.0
    adapter.is_conv = False
    adapter.conv_dim = 0
    adapter.kw_dict = {}
    h_out = adapter.h(x, base)
    runtime = base + h_out

    err, rel = dtype_stats(runtime, ref)
    report(
        f"linear/{label}: bypass == folding",
        err < ATOL,
        f"max_abs={err:.3e} rel={rel:.3e} shape=[{out_f},{in_f}] rank={down.shape[0]}",
    )

    # ---- per-sample alpha：alpha = [1, 0, 1, 0] -------------------------------------------------
    alpha_vec = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32).view(B, 1)
    mixed = base + h_out * alpha_vec
    expect = torch.where(alpha_vec.bool(), ref, base)
    err, rel = dtype_stats(mixed, expect)
    report(
        f"linear/{label}: per-sample alpha == 两张折叠模型的拼接",
        err < ATOL,
        f"max_abs={err:.3e} rel={rel:.3e}",
    )

    # ---- alpha_uncond = 0 时，uncond 行必须与纯 base 完全一致 ------------------------------------
    uncond_rows = mixed[1::2]
    base_rows = base[1::2]
    err, _ = dtype_stats(uncond_rows, base_rows)
    report(
        f"linear/{label}: alpha=0 的行严格等于 base（无 LoRA 泄漏）",
        err < 1e-6,
        f"max_abs={err:.3e}",
    )

    # ---- 分数 alpha（比如 0.5）也应该线性可插值 ------------------------------------------------
    half = 0.5
    mixed2 = base + h_out * half
    err, rel = dtype_stats(mixed2, base + (ref - base) * half)
    report(f"linear/{label}: 分数 alpha 线性插值", err < ATOL, f"max_abs={err:.3e}")


# --------------------------------------------------------------------------------------
# 3. Conv2d 路径
# --------------------------------------------------------------------------------------
def test_conv(adapter, label, kernel_size):
    up, down = adapter.weights[0], adapter.weights[1]
    if down.dim() != 2 or up.dim() != 2:
        report(f"conv/{label}: 跳过（权重不是 2D 原始 LoRA 格式）", True)
        return

    k = kernel_size
    in_ch = down.shape[1] // (k * k)
    if in_ch * k * k != down.shape[1]:
        report(f"conv/{label}: 跳过（{down.shape[1]} 无法按 {k}x{k} 整除）", True)
        return

    out_ch = up.shape[0]
    padding = k // 2
    torch.manual_seed(0)
    W = torch.randn(out_ch, in_ch, k, k, dtype=torch.float32) * 0.1
    W_folded = folded(adapter, W)

    B, H, Wd = 2, 8, 8
    x = torch.randn(B, in_ch, H, Wd, dtype=torch.float32)
    kw = dict(stride=(1, 1), padding=(padding, padding), dilation=(1, 1), groups=1)
    base = torch.nn.functional.conv2d(x, W, **kw)
    ref = torch.nn.functional.conv2d(x, W_folded, **kw)

    adapter.multiplier = 1.0
    adapter.is_conv = True
    adapter.conv_dim = 2
    adapter.kernel_size = (k, k)
    adapter.in_channels = in_ch
    adapter.out_channels = out_ch
    adapter.kw_dict = kw
    h_out = adapter.h(x, base)
    runtime = base + h_out

    err, rel = dtype_stats(runtime, ref)
    report(
        f"conv{k}x{k}/{label}: bypass == folding",
        err < ATOL,
        f"max_abs={err:.3e} rel={rel:.3e} base={tuple(base.shape)} h={tuple(h_out.shape)}",
    )


# --------------------------------------------------------------------------------------
# 4. 覆盖统计：这个 LoRA 里有多少 key 是 bypass 能正确处理的
# --------------------------------------------------------------------------------------
def coverage(sd):
    ups = [k for k in sd if k.endswith(".lora_up.weight")]
    n_dora = n_reshape = n_mid = n_plain = 0
    for u in ups:
        x = u[: -len(".lora_up.weight")]
        if (x + ".lora_down.weight") not in sd:
            continue
        if x + ".dora_scale" in sd:
            n_dora += 1
        elif x + ".reshape_weight" in sd:
            n_reshape += 1
        elif x + ".lora_mid.weight" in sd:
            n_mid += 1
        else:
            n_plain += 1
    total = n_plain + n_dora + n_reshape + n_mid
    print(
        f"     覆盖率: 普通={n_plain} DoRA={n_dora} reshape={n_reshape} tucker(mid)={n_mid} / 共 {total}"
    )
    return total


# --------------------------------------------------------------------------------------
# 5. 合成边界测试（不依赖具体 LoRA 文件，直接构造 adapter）
# --------------------------------------------------------------------------------------
def make_adapter(up, down, alpha=None, mid=None, dora=None, reshape=None):
    return comfy.weight_adapter.LoRAAdapter(set(), (up, down, alpha, mid, dora, reshape))


def synth_conv(k, stride=1, in_ch=8, out_ch=12, r=4, alpha=None):
    torch.manual_seed(1)
    up = torch.randn(out_ch, r) * 0.2
    down = torch.randn(r, in_ch * k * k) * 0.2
    W = torch.randn(out_ch, in_ch, k, k) * 0.2
    ad = make_adapter(up, down, alpha)
    W_folded = folded(ad, W)

    padding = (k - 1) // 2
    kw = dict(stride=(stride, stride), padding=(padding, padding), dilation=(1, 1), groups=1)
    x = torch.randn(2, in_ch, 8, 8)
    base = F.conv2d(x, W, **kw)
    ref = F.conv2d(x, W_folded, **kw)

    ad.multiplier = 1.0
    ad.is_conv = True
    ad.conv_dim = 2
    ad.kernel_size = (k, k)
    ad.in_channels = in_ch
    ad.out_channels = out_ch
    ad.kw_dict = kw
    try:
        h_out = ad.h(x, base)
    except Exception as e:  # noqa: BLE001
        report(f"synth conv{k}x{k} stride={stride}: h() 不抛异常", False, repr(e))
        return
    runtime = base + h_out
    err, rel = dtype_stats(runtime, ref)
    report(
        f"synth conv{k}x{k} stride={stride} alpha={alpha}: bypass == folding",
        err < ATOL and h_out.shape == base.shape,
        f"max_abs={err:.3e} rel={rel:.3e} base={tuple(base.shape)} h={tuple(h_out.shape)}",
    )


def synth_linear(alpha=None):
    torch.manual_seed(2)
    in_f, out_f, r = 32, 48, 4
    up = torch.randn(out_f, r) * 0.2
    down = torch.randn(r, in_f) * 0.2
    W = torch.randn(out_f, in_f) * 0.2
    ad = make_adapter(up, down, alpha)
    W_folded = folded(ad, W)
    x = torch.randn(2, in_f)
    ad.multiplier = 1.0
    ad.is_conv = False
    ad.conv_dim = 0
    ad.kw_dict = {}
    base = F.linear(x, W)
    ref = F.linear(x, W_folded)
    runtime = base + ad.h(x, base)
    err, rel = dtype_stats(runtime, ref)
    report(f"synth linear alpha={alpha}: bypass == folding", err < ATOL, f"max_abs={err:.3e} rel={rel:.3e}")


def synth_known_gaps():
    """已知缺口：这些情况 bypass 会算错，需要 fallback 到折叠。这里确认缺口真实存在。"""
    print("----- 已知缺口确认（bypass 处理不了，必须 fallback） -----")

    # DoRA (weight_decompose)
    torch.manual_seed(3)
    in_f, out_f, r = 32, 48, 4
    up, down = torch.randn(out_f, r) * 0.2, torch.randn(r, in_f) * 0.2
    dora = torch.randn(out_f).abs() + 0.5
    W = torch.randn(out_f, in_f) * 0.2
    ad = make_adapter(up, down, 8.0, None, dora)
    W_folded = folded(ad, W)
    ad.multiplier = 1.0
    ad.is_conv = False
    ad.conv_dim = 0
    ad.kw_dict = {}
    x = torch.randn(2, in_f)
    base = F.linear(x, W)
    ref = F.linear(x, W_folded)
    runtime = base + ad.h(x, base)
    err, rel = dtype_stats(runtime, ref)
    print(f"      DoRA: max_abs={err:.3e} rel={rel:.3e}  -> {'缺口确认(需 fallback)' if err > ATOL else '竟然一致?'}")

    # reshape
    torch.manual_seed(4)
    up, down = torch.randn(out_f, r) * 0.2, torch.randn(r, in_f) * 0.2
    W = torch.randn(out_f, in_f) * 0.2
    ad = make_adapter(up, down, None, None, None, [out_f, in_f])
    W_folded = folded(ad, W)
    ad.multiplier = 1.0
    ad.is_conv = False
    ad.conv_dim = 0
    ad.kw_dict = {}
    runtime = base + ad.h(x, base)
    ref = F.linear(x, W_folded)
    err, rel = dtype_stats(runtime, ref)
    print(f"      reshape: max_abs={err:.3e} rel={rel:.3e}  -> {'缺口确认(需 fallback)' if err > ATOL else '一致'}")


def test_synthetic():
    print("----- 合成测试 -----")
    synth_linear(None)
    synth_linear(8.0)
    synth_conv(3, 1)
    synth_conv(3, 2)
    synth_conv(5, 1)
    synth_conv(1, 1)
    synth_conv(3, 1, alpha=8.0)
    synth_known_gaps()
    print("-" * 78)


def main():
    torch.set_grad_enabled(False)

    path, sd, ups = pick_lora()
    print(f"测试 LoRA: {path}")
    x = find_clean_key(sd, ups)
    if x is None:
        raise SystemExit("这个 LoRA 里没有干净的 up/down 配对")
    print(f"测试 key : {x}")
    coverage(sd)

    adapter, alpha = build_adapter(sd, x)
    if adapter is None:
        raise SystemExit("adapter 构造失败")
    print(f"adapter alpha={alpha} up={tuple(adapter.weights[0].shape)} down={tuple(adapter.weights[1].shape)}")
    print(f"dora_scale 存在: {x + '.dora_scale' in sd}   (bypass 的 h() 会忽略它)")
    print("-" * 78)

    label = os.path.basename(path)
    test_linear(adapter, label)
    test_conv(adapter, label, 1)
    test_conv(adapter, label, 3)
    test_conv(adapter, label, 5)

    print("-" * 78)
    test_synthetic()

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
