"""
这个文件用于统一管理 OpenVLA 在 Hugging Face AutoClass 体系下的本地加载策略。

输入：
- `pretrained_model_name_or_path`：一个 Hugging Face repo id，或者一个本地 checkpoint 目录路径。
- 可选的额外加载参数：例如 `torch_dtype`、`low_cpu_mem_usage` 等。

输出：
- 返回一组可直接传给 `AutoProcessor.from_pretrained(...)` 或
  `AutoModelForVision2Seq.from_pretrained(...)` 的关键字参数。
- 同时提供一个统一的 AutoClass 注册函数，保证“本地 checkpoint 路径”可以优先走仓库代码。
"""

from pathlib import Path
from typing import Any, Dict, Union

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from .configuration_prismatic import OpenVLAConfig
from .modeling_prismatic import OpenVLAForActionPrediction
from .processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def register_openvla_auto_classes() -> None:
    """把 OpenVLA 相关类统一注册到 Hugging Face 的 AutoClass 工厂里。"""
    # 这里使用 `exist_ok=True`，是为了允许训练入口、评测入口多次调用这个函数而不报错。
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)


def is_local_pretrained_path(pretrained_model_name_or_path: Union[str, Path]) -> bool:
    """判断输入路径当前是否真实存在于本地文件系统中。"""
    # 这里故意不用“是否包含斜杠”之类的字符串启发式规则，
    # 而是直接检查文件系统，避免把合法的 HF repo id 误判成“本地路径”。
    return Path(str(pretrained_model_name_or_path)).expanduser().exists()


def build_pretrained_load_kwargs(
    pretrained_model_name_or_path: Union[str, Path],
    *,
    trust_remote_code_for_remote: bool = True,
    **extra_kwargs: Any,
) -> Dict[str, Any]:
    """根据“本地路径”还是“远端 repo”生成一组安全的 HF 加载参数。"""
    # 先复制一份参数，避免直接修改调用方传进来的对象。
    load_kwargs: Dict[str, Any] = dict(extra_kwargs)

    # 如果路径真实存在于本地，我们就强制优先使用仓库代码：
    # - `trust_remote_code=False`：不要去执行 checkpoint 目录自带的 Python 文件。
    # - `local_files_only=True`：只从本地读文件，避免在本地路径场景下再去访问远端。
    if is_local_pretrained_path(pretrained_model_name_or_path):
        load_kwargs.setdefault("trust_remote_code", False)
        load_kwargs.setdefault("local_files_only", True)
        return load_kwargs

    # 如果不是本地路径，就保持原有远端加载行为。
    load_kwargs.setdefault("trust_remote_code", trust_remote_code_for_remote)
    return load_kwargs
