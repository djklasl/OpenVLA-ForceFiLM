# 这个文件用于构建和加载 PRIMUS 预训练 IMU 编码器，输入是六维力/力矩历史窗口，输出是 512 维力觉特征。
"""PRIMUS 预训练 IMU 编码器的轻量加载模块。"""

import os
import warnings
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

# torchmetrics 的导入链在部分环境里会触发 TensorFlow 初始化日志。
# PRIMUS 这里只需要 LightningModule 类型，不需要 TensorFlow 日志；提前设置环境变量可以减少无关输出。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# pytorch_lightning 导入链会触发 lightning_utilities 里关于 pkg_resources 的弃用提示。
# 这个提示不影响 PRIMUS 权重加载，而且每个训练 rank 都会打印一遍；这里只精确屏蔽这一条低价值提示。
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="lightning_utilities.core.imports",
)
import pytorch_lightning as pl


class Block(nn.Module):
    """PRIMUS 中使用的一段 Conv1d + Pooling 结构。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_type: str = "max", embedding_size: int = 32) -> None:
        super().__init__()
        if pool_type == "max":
            pool_fn = nn.MaxPool1d(kernel_size=3)
        elif pool_type == "adaptive":
            pool_fn = nn.AdaptiveAvgPool1d(output_size=embedding_size)
        else:
            raise ValueError(f"pool_type {pool_type} not supported")

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=2,
                bias=False,
            ),
            pool_fn,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """把输入时间序列送入当前卷积池化块。"""
        return self.net(batch)


class MW2StackRNNPoolingMultihead(pl.LightningModule):
    """PRIMUS 发布权重对应的多头 IMU 编码器。"""

    def __init__(self, input_dim: int = 32, size_embeddings: int = 512) -> None:
        super().__init__()
        # 和 PRIMUS 原始实现保持一致；这个字段不参与 forward，也不影响 checkpoint key。
        self.name = MW2StackRNNPoolingMultihead
        self.backbone = nn.Sequential(
            nn.GroupNorm(2, 6),
            Block(6, input_dim, 10),
            Block(input_dim, input_dim, 5),
            Block(input_dim, input_dim, 5, pool_type="adaptive", embedding_size=32),
            nn.GroupNorm(4, input_dim),
            nn.GRU(batch_first=True, input_size=input_dim, hidden_size=size_embeddings),
        )
        self.ssl_head = nn.Linear(size_embeddings, size_embeddings)
        self.mmcl_head = nn.Linear(size_embeddings, size_embeddings)

    def forward(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """返回 PRIMUS 多头输出，包含 ssl、mmcl 和 backbone emb。"""
        emb = self.backbone(batch)[1][0]
        return {
            "ssl": self.ssl_head(emb),
            "mmcl": self.mmcl_head(emb),
            "emb": emb,
        }


def _unwrap_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """从 Lightning checkpoint 或普通 state_dict 中取出真正的参数字典。"""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def extract_imu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """只抽取 PRIMUS checkpoint 里的 IMU encoder 权重。"""
    prefixes = ("imu_encoder.", "module.imu_encoder.")
    for prefix in prefixes:
        imu_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if imu_state:
            return imu_state

    encoder_prefixes = ("backbone.", "ssl_head.", "mmcl_head.")
    if any(key.startswith(encoder_prefixes) for key in state_dict):
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith(encoder_prefixes)
        }

    raise KeyError("No IMU encoder weights found in checkpoint state_dict.")


def module_has_meta_tensor(module: nn.Module) -> bool:
    """判断模块参数或 buffer 里是否还有 meta tensor。"""
    # HuggingFace 的 low_cpu_mem_usage 加载会先把新增模块建在 meta device 上。
    # meta tensor 只有形状没有真实数据，普通 copy/load 会变成 no-op，后续 .to(device) 会崩溃。
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.is_meta:
            return True
    return False


def load_state_dict_with_meta_handling(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> torch.nn.modules.module._IncompatibleKeys:
    """把 state_dict 加载进模块；遇到 meta tensor 时使用 assign=True 真正落地参数。"""
    # assign=True 会直接用 state_dict 中的真实 tensor 替换 meta 参数，避免 PyTorch 打印 no-op warning。
    if module_has_meta_tensor(module):
        return module.load_state_dict(state_dict, strict=strict, assign=True)
    return module.load_state_dict(state_dict, strict=strict)


def load_primus_imu_encoder(
    checkpoint_path: str,
    device: Optional[Union[str, torch.device]] = None,
    embedding_dim: int = 512,
    strict: bool = True,
    eval_mode: bool = True,
) -> MW2StackRNNPoolingMultihead:
    """加载冻结前的 PRIMUS 多头 IMU 编码器。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_imu_state_dict(_unwrap_state_dict(checkpoint))
    encoder = MW2StackRNNPoolingMultihead(size_embeddings=embedding_dim)
    load_state_dict_with_meta_handling(encoder, state_dict, strict=strict)
    if device is not None:
        encoder.to(device)
    if eval_mode:
        encoder.eval()
    return encoder
