"""lk_lora_stack —— 自定义 LoRA 加载器节点集。

目录结构::

    lk_lora_stack/
    ├─ __init__.py          # 包入口
    ├─ nodes/               # 后端节点
    │  ├─ __init__.py       # 节点聚合
    │  └─ lora_loader.py    # LK Lora Loader（单个）
    └─ js/                  # 前端扩展（暂时留空，等做动态槽位时再启用）
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# 前端还没有 JS 资源，先不要声明 WEB_DIRECTORY，否则加载时会告警。
# 等 js/ 目录有内容后再取消下面两行注释：
# WEB_DIRECTORY = "./js"
# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
