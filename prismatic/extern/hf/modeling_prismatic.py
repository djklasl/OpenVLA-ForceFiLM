"""HuggingFace-style Prismatic / OpenVLA model definitions."""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig
from .primus_imu_encoder import (
    MW2StackRNNPoolingMultihead,
    load_primus_imu_encoder,
    load_state_dict_with_meta_handling,
    module_has_meta_tensor,
)

logger = logging.getLogger(__name__)


IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF 会改写名含 `gamma` 的参数名，这里把 timm LayerScale.gamma 改成 scale_factor。
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


def _get_block_feature_dim_from_norm(norm_module: nn.Module) -> int:
    """从 ViT block 的 `norm1` 中解析出最后一维的特征维度。"""
    normalized_shape = getattr(norm_module, "normalized_shape", None)
    if normalized_shape is None:
        raise ValueError("Unable to infer feature dimension from block.norm1; missing `normalized_shape`.")

    # 兼容 int / tuple / list 三种 LayerNorm 维度表示。
    if isinstance(normalized_shape, int):
        return normalized_shape
    if isinstance(normalized_shape, (tuple, list)) and len(normalized_shape) > 0:
        return int(normalized_shape[-1])

    raise ValueError(f"Unexpected `normalized_shape` on block.norm1: {normalized_shape}")


class FiLMWrappedViTBlock(nn.Module):
    """对单个 timm ViT block 做最小包装，只改写 `norm1 -> attn` 之间的逻辑。"""

    def __init__(self, original_block: nn.Module) -> None:
        super().__init__()

        # 只支持当前 OpenVLA 使用的标准 timm ViT block，结构不符时尽早报错。
        required_attributes = ["norm1", "attn", "ls1", "drop_path1", "norm2", "mlp", "ls2", "drop_path2"]
        for attribute_name in required_attributes:
            if not hasattr(original_block, attribute_name):
                raise ValueError(f"Unsupported ViT block for FiLM wrapping; missing attribute `{attribute_name}`")

        # 保留原子模块字段名，避免 checkpoint 键名多出 `original_block.*` 前缀。
        self.norm1 = original_block.norm1
        self.attn = original_block.attn
        self.ls1 = original_block.ls1
        self.drop_path1 = original_block.drop_path1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        self.ls2 = original_block.ls2
        self.drop_path2 = original_block.drop_path2

        # 每个 block 按自己的 feature_dim 消费独立 gamma/beta。
        self.feature_dim = _get_block_feature_dim_from_norm(self.norm1)
        # 仅缓存当前 forward 的条件参数，不参与 state_dict。
        self.current_gamma: Optional[torch.Tensor] = None
        self.current_beta: Optional[torch.Tensor] = None

    def set_film_params(self, gamma: torch.Tensor, beta: torch.Tensor) -> None:
        """给当前 block 设置一次前向中要使用的 `gamma` / `beta`。"""
        self.current_gamma = gamma
        self.current_beta = beta

    def clear_film_params(self) -> None:
        """清理当前 block 暂存的 `gamma` / `beta`，避免 batch 间串值。"""
        self.current_gamma = None
        self.current_beta = None

    def _apply_film(self, hidden_states: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """消费当前 block 的外部 gamma/beta。"""
        # 先在消费端校验维度，方便尽早定位 FiLM 参数切片错误。
        if gamma.shape[-1] != self.feature_dim or beta.shape[-1] != self.feature_dim:
            raise ValueError(
                f"FiLM parameter dimension mismatch: expected {self.feature_dim}, "
                f"got gamma={gamma.shape}, beta={beta.shape}"
            )

        # 这里的 gamma 是 delta-gamma：x * (1 + gamma) + beta。
        gamma = gamma.to(dtype=hidden_states.dtype, device=hidden_states.device)
        beta = beta.to(dtype=hidden_states.dtype, device=hidden_states.device)
        return hidden_states * (1 + gamma) + beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 只在 `norm1 -> attn` 之间插入 FiLM，其余 block 路径保持原样。
        hidden_states = self.norm1(x)
        if self.current_gamma is not None and self.current_beta is not None:
            hidden_states = self._apply_film(hidden_states, self.current_gamma, self.current_beta)
        x = x + self.drop_path1(self.ls1(self.attn(hidden_states)))

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ForceFiLMGenerator(nn.Module):
    """用可微调的 PRIMUS IMU 编码器把力历史窗口编码成所有 FiLM 参数。"""

    def __init__(
        self,
        encoder_checkpoint_path: str,
        embedding_type: str,
        history_length: int,
        total_film_dim: int,
    ) -> None:
        super().__init__()

        if embedding_type not in {"ssl", "mmcl", "emb"}:
            raise ValueError(f"Unsupported PRIMUS embedding_type: {embedding_type}")
        if history_length <= 0:
            raise ValueError("ForceFiLMGenerator history_length must be positive.")
        if total_film_dim <= 0:
            raise ValueError("ForceFiLMGenerator total_film_dim must be positive.")

        self.encoder_checkpoint_path = encoder_checkpoint_path
        self.embedding_type = embedding_type
        self.history_length = history_length
        self.total_film_dim = total_film_dim
        # best_model.ckpt 里的 PRIMUS GRU hidden size 和 mmcl/ssl head 都是 512 维。
        # 这里直接固定为 512，避免代码看起来像还在兼容未知 embedding 维度。
        self.encoder_embedding_dim = 512
        # rank-256 瓶颈用于把最后一层大输出映射的参数量减半：
        # 旧结构约是 512 * total_film_dim，新结构最后一层约是 256 * total_film_dim。
        self.projection_rank = 256
        # 先注册同构的 PRIMUS 模块；真正的预训练权重会在 post_init() 后加载，避免被 HF 初始化覆盖。
        # 注意：这里不再冻结 PRIMUS。训练脚本打开 train_force_film=True 后，
        # PEFT 会先冻结 base model，随后再由 unfreeze_force_film_generator(...) 显式解冻整条 Force-FiLM 分支。
        self.imu_encoder = MW2StackRNNPoolingMultihead(size_embeddings=self.encoder_embedding_dim)
        # 输出映射负责把 PRIMUS 力觉特征变成所有视觉层需要的 gamma/beta。
        # 结构说明：
        # 1. LayerNorm 先把 PRIMUS embedding 的尺度拉稳，减少后面大输出层被异常值放大的概率；
        # 2. 512 -> 512 保留 PRIMUS 表征容量，再用 512 -> 256 做瓶颈压缩；
        # 3. 最后一层从 256 维输出完整 gamma/beta 向量，减少参数量并降低 NaN 风险。
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.encoder_embedding_dim),
            nn.Linear(self.encoder_embedding_dim, self.encoder_embedding_dim),
            nn.GELU(),
            nn.Linear(self.encoder_embedding_dim, self.projection_rank),
            nn.GELU(),
            nn.LayerNorm(self.projection_rank),
            nn.Linear(self.projection_rank, self.total_film_dim),
        )
        self._initialize_output_projection_body()
        self._small_initialize_output_projection_final_layer()

    def _get_output_projection_final_layer(self) -> nn.Linear:
        """取出 output_projection 的最后一层大 Linear，便于单独做小初始化或清零。"""
        final_layer = self.output_projection[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError(f"Expected output_projection final layer to be nn.Linear, got {type(final_layer)}")
        return final_layer

    def _initialize_output_projection_body(self) -> None:
        """初始化最后一层之前的 MLP 主体，兼容 HF meta tensor 落地后的空内存。"""
        # low_cpu_mem_usage=True 时，新增模块可能先被放在 meta device 上。
        # 一旦后面调用 to_empty() 分配真实存储，参数内容不会自动有合理初值，
        # 所以这里显式初始化 LayerNorm 和中间 Linear，避免未初始化内存进入训练。
        for layer in list(self.output_projection.children())[:-1]:
            if isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _small_initialize_output_projection_final_layer(self) -> None:
        """小初始化最后一层，避免刚开始训练时直接产生过大的 gamma/beta。"""
        final_layer = self._get_output_projection_final_layer()
        nn.init.normal_(final_layer.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(final_layer.bias)

    def load_pretrained_imu_encoder(self) -> None:
        """在 HF 初始化完成后加载 PRIMUS 权重，并保留后续训练时的可微调能力。"""
        loaded_encoder = load_primus_imu_encoder(
            checkpoint_path=self.encoder_checkpoint_path,
            embedding_dim=self.encoder_embedding_dim,
            eval_mode=True,
        )
        load_state_dict_with_meta_handling(self.imu_encoder, loaded_encoder.state_dict(), strict=True)
        # 这里只负责把预训练权重放进当前模块，不再把 requires_grad 关掉。
        # 这样训练端解冻 ForceFiLMGenerator 后，PRIMUS backbone 和 output_projection 会一起进入 optimizer。
        self.imu_encoder.requires_grad_(True)

    def zero_initialize_output_layer(self) -> None:
        """把输出层清零，保证初始输出严格退化到 identity FiLM。"""
        # low_cpu_mem_usage=True 时，新增的输出映射层可能仍是 meta tensor。
        # 先分配真实 CPU 存储，再清零，后续 vla.to(device_id) 才能正常搬到 GPU。
        if module_has_meta_tensor(self.output_projection):
            self.output_projection.to_empty(device=torch.device("cpu"))
        self._initialize_output_projection_body()
        final_layer = self._get_output_projection_final_layer()
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, suction_wrench: torch.Tensor) -> torch.Tensor:
        if suction_wrench.ndim != 3:
            raise ValueError(
                f"`suction_wrench` must have shape [B, 6, {self.history_length}], got {tuple(suction_wrench.shape)}"
            )
        if suction_wrench.shape[1] != 6 or suction_wrench.shape[2] != self.history_length:
            raise ValueError(
                f"`suction_wrench` shape mismatch: expected [B, 6, {self.history_length}], "
                f"got {tuple(suction_wrench.shape)}"
            )

        encoder_param = next(self.imu_encoder.parameters())
        cleaned_force = torch.nan_to_num(suction_wrench.to(device=encoder_param.device, dtype=encoder_param.dtype))
        # 这里不能使用 torch.no_grad()，否则 PRIMUS backbone 即使 requires_grad=True 也不会得到梯度。
        imu_outputs = self.imu_encoder(cleaned_force)
        force_features = imu_outputs[self.embedding_type]
        projection_param = next(self.output_projection.parameters())
        force_features = force_features.to(device=projection_param.device, dtype=projection_param.dtype)
        return self.output_projection(force_features)


def _wrap_vit_blocks_with_identity_film(featurizer: nn.Module, selected_indices: List[int]) -> None:
    """把指定下标的 ViT block 原地替换成带 FiLM 包装层的版本。"""
    if not hasattr(featurizer, "blocks"):
        raise ValueError("Vision backbone does not expose `blocks`; cannot insert FiLM wrappers safely.")

    blocks = featurizer.blocks

    # 原地替换 selected blocks；包装器内部保留原子模块名以兼容旧权重。
    for block_index in selected_indices:
        if not isinstance(blocks[block_index], FiLMWrappedViTBlock):
            blocks[block_index] = FiLMWrappedViTBlock(blocks[block_index])


class ForceFiLMController:
    """集中管理 Force-FiLM 的 block 包装、参数生成、参数注入和临时状态清理。"""

    def __init__(
        self,
        featurizers: List[Tuple[str, nn.Module]],
        force_encoder_checkpoint_path: str,
        force_encoder_embedding_type: str,
        force_history_length: int,
    ) -> None:
        # Force-FiLM 固定调制每一路视觉 backbone 的所有 ViT blocks。
        self.force_encoder_checkpoint_path = force_encoder_checkpoint_path
        self.force_encoder_embedding_type = force_encoder_embedding_type
        self.force_history_length = force_history_length
        self.force_film_targets: List[Dict[str, Any]] = []

        for backbone_name, featurizer in featurizers:
            self._wrap_all_blocks(featurizer, backbone_name=backbone_name)
            self.force_film_targets.extend(self._collect_targets(featurizer, backbone_name=backbone_name))

        if len(self.force_film_targets) == 0:
            raise ValueError("Force-FiLM enabled, but the vision backbone does not expose any target blocks.")

        # 每个 target block 需要一份 gamma 和 beta。
        self.total_force_film_dim = sum(2 * int(target["feature_dim"]) for target in self.force_film_targets)
        self.force_film_generator = ForceFiLMGenerator(
            encoder_checkpoint_path=self.force_encoder_checkpoint_path,
            embedding_type=self.force_encoder_embedding_type,
            history_length=self.force_history_length,
            total_film_dim=self.total_force_film_dim,
        )

    def _wrap_all_blocks(self, featurizer: nn.Module, backbone_name: str) -> None:
        """把当前视觉 backbone 的所有 ViT blocks 都替换成可接收 FiLM 参数的包装层。"""
        if not hasattr(featurizer, "blocks"):
            raise ValueError(f"Vision backbone `{backbone_name}` does not expose `blocks`; cannot insert Force-FiLM.")

        # 固定 all 策略，不再支持部分层选择。
        all_block_indices = list(range(len(featurizer.blocks)))
        _wrap_vit_blocks_with_identity_film(featurizer, all_block_indices)

    def _collect_targets(self, featurizer: nn.Module, backbone_name: str) -> List[Dict[str, Any]]:
        """收集当前视觉 backbone 里所有真正会消费 Force-FiLM 参数的 wrapped blocks。"""
        targets: List[Dict[str, Any]] = []
        for block_index, block in enumerate(featurizer.blocks):
            if not isinstance(block, FiLMWrappedViTBlock):
                raise ValueError(
                    f"Expected wrapped ViT block at {backbone_name}.blocks[{block_index}], got {type(block)}"
                )
            targets.append(
                {
                    "backbone_name": backbone_name,
                    "block_index": block_index,
                    "feature_dim": block.feature_dim,
                    "block": block,
                }
            )
        return targets

    def load_pretrained_imu_encoder(self) -> None:
        """加载 ForceFiLMGenerator 内部可微调的 PRIMUS IMU 编码器。"""
        self.force_film_generator.load_pretrained_imu_encoder()

    def zero_initialize_output_layer(self) -> None:
        """把 ForceFiLMGenerator 的最后一层清零，保证初始行为退化为 identity FiLM。"""
        self.force_film_generator.zero_initialize_output_layer()

    def clear(self) -> None:
        """清理所有 target blocks 上当前 forward 暂存的 gamma/beta，避免不同 batch 串值。"""
        for target in self.force_film_targets:
            target["block"].clear_film_params()

    def apply(self, suction_wrench: torch.Tensor, batch_size: int) -> None:
        """根据 suction_wrench 一次性生成所有 target blocks 的 gamma/beta，并注入到对应 block。"""
        if suction_wrench.ndim != 3:
            raise ValueError(
                f"`suction_wrench` must have shape [B, 6, {self.force_history_length}], got {suction_wrench.shape}"
            )
        if suction_wrench.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch between pixel_values ({batch_size}) and suction_wrench ({suction_wrench.shape[0]})."
            )
        if suction_wrench.shape[1] != 6 or suction_wrench.shape[2] != self.force_history_length:
            raise ValueError(
                f"`suction_wrench` shape mismatch: expected [B, 6, {self.force_history_length}], got {suction_wrench.shape}"
            )

        # 对齐生成器 device/dtype，避免 Linear dtype 或 device 不一致。
        film_params = self.force_film_generator(suction_wrench)
        if film_params.shape[-1] != self.total_force_film_dim:
            raise ValueError(
                f"Force-FiLM parameter dimension mismatch: expected {self.total_force_film_dim}, got {film_params.shape[-1]}"
            )

        # 按 target block 顺序切成每层自己的 gamma/beta。
        start_index = 0
        for target in self.force_film_targets:
            feature_dim = int(target["feature_dim"])
            end_index = start_index + (2 * feature_dim)
            target_params = film_params[:, start_index:end_index]
            gamma, beta = torch.split(target_params, feature_dim, dim=-1)
            target["block"].set_film_params(gamma.unsqueeze(1), beta.unsqueeze(1))
            start_index = end_index


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
        use_force_film: bool = False,
        force_encoder_checkpoint_path: str = "/data1/ps/pretrained-imu-encoders/checkpoint/best_model.ckpt",
        force_encoder_embedding_type: str = "mmcl",
        force_history_length: int = 250,
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.use_force_film = use_force_film
        self.force_encoder_checkpoint_path = force_encoder_checkpoint_path
        self.force_encoder_embedding_type = force_encoder_embedding_type
        self.force_history_length = force_history_length
        # Force-FiLM 细节集中到 controller，backbone 只保留薄调用。
        self.force_film_controller: Optional[ForceFiLMController] = None
        # 继续把 ForceFiLMGenerator 注册在旧位置，保持完整模型 state_dict 和旧保存/加载工具的路径稳定。
        self.force_film_generator: Optional[ForceFiLMGenerator] = None

        # 创建 alpha featurizer，并把 forward 固定到倒数第二层 patch 输出。
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # 融合视觉 backbone 时再创建 beta featurizer。
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # 修正 LayerScale 参数名，避免 HF 加载时误处理 gamma。
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

        # 打开 Force-FiLM 时固定包装全部 ViT blocks。
        if self.use_force_film:
            force_featurizers = [("featurizer", self.featurizer)]
            if self.use_fused_vision_backbone:
                force_featurizers.append(("fused_featurizer", self.fused_featurizer))
            self.force_film_controller = ForceFiLMController(
                featurizers=force_featurizers,
                force_encoder_checkpoint_path=self.force_encoder_checkpoint_path,
                force_encoder_embedding_type=self.force_encoder_embedding_type,
                force_history_length=self.force_history_length,
            )
            self.force_film_generator = self.force_film_controller.force_film_generator

    def load_pretrained_force_encoder(self) -> None:
        """加载可微调的 PRIMUS IMU 编码器。"""
        if self.force_film_controller is not None:
            self.force_film_controller.load_pretrained_imu_encoder()

    def zero_initialize_force_film_output_layer(self) -> None:
        """把 ForceFiLMGenerator 的最后一层清零，保证初始行为退化为 identity FiLM。"""
        if self.force_film_controller is not None:
            self.force_film_controller.zero_initialize_output_layer()

    def forward(
        self, pixel_values: torch.Tensor, suction_wrench: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if self.force_film_controller is not None:
            self.force_film_controller.clear()
        try:
            if self.force_film_controller is not None and suction_wrench is not None:
                self.force_film_controller.apply(suction_wrench, batch_size=pixel_values.shape[0])

            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # 融合视觉输入按通道拆成两张图，再拼接两路 patch features。
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            return torch.cat([patches, patches_fused], dim=2)
        finally:
            if self.force_film_controller is not None:
                self.force_film_controller.clear()


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # 融合视觉 backbone 使用更宽的三层 projector。
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # HF 版只服务推理和微调，不用于从零训练 VLM。
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone,
            config.image_sizes,
            config.timm_model_ids,
            config.timm_override_act_layers,
            use_force_film=getattr(config, "use_force_film", False),
            force_encoder_checkpoint_path=getattr(
                config,
                "force_encoder_checkpoint_path",
                "/data1/ps/pretrained-imu-encoders/checkpoint/best_model.ckpt",
            ),
            force_encoder_embedding_type=getattr(config, "force_encoder_embedding_type", "mmcl"),
            force_history_length=getattr(config, "force_history_length", 250),
        )

        # Projector 将视觉特征映射到 LLM hidden size。
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        self.post_init()
        if getattr(config, "use_force_film", False) and getattr(config, "force_encoder_checkpoint_path", None):
            # 训练端会传入真实的 PRIMUS best_model.ckpt。
            # `post_init()` 会初始化新增模块，所以 PRIMUS 权重必须在它之后加载，避免被 HF 初始化覆盖。
            self.vision_backbone.load_pretrained_force_encoder()
            # PRIMUS 权重就位后再清零输出层，保证联合训练起点退化为 identity FiLM。
            self.vision_backbone.zero_initialize_force_film_output_layer()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        suction_wrench: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 训练时禁用 cache。
        use_cache = use_cache and not self.training

        projected_patch_embeddings = None

        # cached generation：仅处理新 token。
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # language-only forward。
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # multimodal forward：视觉 patch 插到 BOS 后。
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            patch_features = self.vision_backbone(pixel_values, suction_wrench=suction_wrench)

            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            input_embeddings = self.get_input_embeddings()(input_ids)

            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        suction_wrench: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # cache 模式只保留最后一个未处理 token。
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # inputs_embeds 只在首个 generation step 使用。
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "suction_wrench": suction_wrench,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # 补齐训练时在 OUT/ASSISTANT 后使用的特殊空格 token。
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)

        # 将动作 token 反离散化为归一化连续动作。
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
