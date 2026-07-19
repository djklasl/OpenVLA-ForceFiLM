"""
configuration_prismatic.py

HuggingFace-style configuration definition for Prismatic VLMs, inheriting from `transformers.PretrainedConfig`.
Default configuration specifies `siglip-224px+7b`.
"""

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

# === Utilities for Mapping Prismatic names to HF names ===
# fmt: off
VISION_BACKBONE_TO_RESOLUTION: Dict[str, List[int]] = {
    "clip-vit-l": [224], "siglip-vit-so400m": [224], "dinov2-vit-l": [224], "in1k-vit-l": [224],

    "clip-vit-l-336px": [336],
    "siglip-vit-so400m-384px": [384],

    "dinoclip-vit-l-336px": [336, 336],
    "dinosiglip-vit-so-224px": [224, 224],
    "dinosiglip-vit-so-384px": [384, 384],
}
VISION_BACKBONE_TO_TIMM_ID: Dict[str, List[str]] = {
    "clip-vit-l": ["vit_large_patch14_clip_224.openai"],
    "clip-vit-l-336px": ["vit_large_patch14_clip_336.openai"],

    "dinov2-vit-l": ["vit_large_patch14_reg4_dinov2.lvd142m"],
    "in1k-vit-l": ["vit_large_patch16_224.augreg_in21k_ft_in1k"],

    "siglip-vit-so400m": ["vit_so400m_patch14_siglip_224"],
    "siglip-vit-so400m-384px": ["vit_so400m_patch14_siglip_384"],

    "dinoclip-vit-l-336px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_large_patch14_clip_336.openai"],
    "dinosiglip-vit-so-224px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
    "dinosiglip-vit-so-384px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_384"],
}
TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
    "clip-vit-l": ["quick_gelu"], "clip-vit-l-336px": ["quick_gelu"],
    "dinov2-vit-l": [None], "in1k-vit-l": [None],
    "siglip-vit-so400m": [None], "siglip-vit-so400m-384px": [None],
    "dinoclip-vit-l-336px": [None, "quick_gelu"],
    "dinosiglip-vit-so-224px": [None, None], "dinosiglip-vit-so-384px": [None, None]
}

LLM_BACKBONE_TO_HF_PATH = {
    "llama2-7b-pure": "meta-llama/Llama-2-7b-hf", "llama2-13b-pure": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf", "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",

    "vicuna-v15-7b": "lmsys/vicuna-7b-v1.5", "vicuna-v15-13b": "lmsys/vicuna-13b-v1.5",

    "mistral-v0.1-7b-pure": "mistralai/Mistral-7B-v0.1",
    "mistral-v0.1-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",

    "phi-2-3b": "microsoft/phi-2",
}
LLM_BACKBONE_TO_HF_METACLASS = {
    "llama2-7b-pure": "llama", "llama2-13b-pure": "llama", "llama2-7b-chat": "llama", "llama2-13b-chat": "llama",
    "vicuna-v15-7b": "llama", "vicuna-v15-13b": "llama",

    "mistral-v0.1-7b-pure": "mistral", "mistral-v0.1-7b-instruct": "mistral",

    "phi-2-3b": "phi",
}

VALID_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTION.keys())
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATH)
# fmt: on


class PrismaticConfig(PretrainedConfig):
    model_type: str = "prismatic"
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "vicuna-v15-7b",
        arch_specifier: str = "no-align+gelu-mlp",
        use_fused_vision_backbone: Optional[bool] = None,
        image_resize_strategy: str = "letterbox",
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        if vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(f"Vision backbone `{vision_backbone_id}` not in {VALID_VISION_BACKBONES = }")

        if llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(f"LLM backbone `{llm_backbone_id}` not in {VALID_LLM_BACKBONES = }")

        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # [Contract] All vision backbone parameters are lists =>> supports fused backbones with different preprocessing
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(self.vision_backbone_id.startswith(v) for v in ["dinoclip", "dinosiglip"])
        )

        self.timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[self.vision_backbone_id]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[self.vision_backbone_id]
        self.image_sizes = VISION_BACKBONE_TO_RESOLUTION[self.vision_backbone_id]
        self.image_resize_strategy = image_resize_strategy

        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH[self.llm_backbone_id]
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of

        # [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that specific naming!
        self.text_config = (
            CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]](**text_config)
            if text_config is not None
            else CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]()
        )

        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class OpenVLAConfig(PrismaticConfig):
    model_type: str = "openvla"

    def __init__(
        self,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = None,
        n_action_bins: int = 256,
        use_force_film: bool = False,
        force_encoder_checkpoint_path: str = "/data1/ps/pretrained-imu-encoders/checkpoint/best_model.ckpt",
        force_encoder_embedding_type: str = "mmcl",
        force_history_length: int = 250,
        **kwargs: str,
    ) -> None:
        # 兼容旧 checkpoint 的 config.json：
        # 旧版本可能保存过 force_film_block_selection，但现在 Force-FiLM 固定调制所有 ViT blocks。
        # 这里主动丢弃旧字段，避免它继续进入 config 成为可配置项。
        kwargs.pop("force_film_block_selection", None)
        # 旧版本 Force-FiLM 使用单帧六维力输入和多层 MLP；现在改为 PRIMUS IMU 编码器 + rank-256 ForceFiLM 投影。
        kwargs.pop("force_feature_dim", None)
        kwargs.pop("force_mlp_hidden_dim", None)
        kwargs.pop("force_mlp_num_layers", None)

        # `use_force_film`：
        # - False：完全不启用基于 `suction_wrench` 的动态 FiLM。
        # - True：构建 ForceFiLMGenerator，并在视觉 backbone 的全部 ViT blocks 上动态注入 gamma / beta。
        self.use_force_film = use_force_film

        # `force_encoder_checkpoint_path`：PRIMUS 预训练 IMU 编码器 checkpoint 路径。
        self.force_encoder_checkpoint_path = force_encoder_checkpoint_path
        # `force_encoder_embedding_type`：使用 PRIMUS 多头输出中的哪一路，当前固定推荐 mmcl。
        self.force_encoder_embedding_type = force_encoder_embedding_type
        # `force_history_length`：每帧 suction_wrench 的历史窗口长度，当前数据集固定为 250。
        self.force_history_length = force_history_length

        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins

        super().__init__(**kwargs)
