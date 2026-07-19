"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.loading_utils import build_pretrained_load_kwargs, register_openvla_auto_classes
from prismatic.extern.hf.primus_imu_encoder import load_state_dict_with_meta_handling

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def locate_force_film_generator(model):
    """在评测模型里定位唯一的 ForceFiLMGenerator。"""
    candidates = []
    for module_name, module in model.named_modules():
        # 同时按模块名和类名找，是为了兼容后续轻微重命名。
        if module_name and (
            module_name.endswith("force_film_generator") or module.__class__.__name__ == "ForceFiLMGenerator"
        ):
            candidates.append((module_name, module))

    dedup_candidates = []
    seen_module_ids = set()
    for module_name, module in candidates:
        if id(module) not in seen_module_ids:
            seen_module_ids.add(id(module))
            dedup_candidates.append((module_name, module))

    if len(dedup_candidates) != 1:
        candidate_names = ", ".join(name for name, _ in dedup_candidates) or "<none>"
        raise ValueError(f"Expected exactly one ForceFiLMGenerator, but found: {candidate_names}")

    return dedup_candidates[0]


def load_force_film_generator_weights(vla, force_film_checkpoint):
    """评测端只从 force_film_generator.pt 完整恢复 ForceFiLMGenerator。"""
    checkpoint_path = Path(force_film_checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "force_film_generator.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Force-FiLM checkpoint not found: {checkpoint_path}")

    force_module_name, force_module = locate_force_film_generator(vla)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # 评测端要求 force_film_generator.pt 必须是完整模块权重：
    # - imu_encoder.*：冻结的 PRIMUS IMU encoder
    # - output_projection.*：训练得到的 gamma/beta 生成层
    # 如果缺任意一部分，直接报错，避免静默退回 best_model.ckpt 造成权重来源不清楚。
    has_imu_encoder = any(key.startswith("imu_encoder.") for key in state_dict)
    has_output_projection = any(key.startswith("output_projection.") for key in state_dict)
    if not has_imu_encoder or not has_output_projection:
        raise ValueError(
            "`force_film_generator.pt` must contain a complete ForceFiLMGenerator state_dict "
            "with both `imu_encoder.*` and `output_projection.*` keys."
        )

    # low_cpu_mem_usage=True 时，ForceFiLMGenerator 可能还在 meta device 上。
    # 这里用 assign=True 的兼容加载方式，保证完整权重真实落到模块里，然后再整体搬到 GPU。
    load_state_dict_with_meta_handling(force_module, state_dict, strict=True)

    # PRIMUS 的 backbone 里有 GRU；当前 CUDA GRU 路径不支持 bfloat16。
    # 评测时只把冻结的 IMU encoder 固定为 float32，主 VLA 和 output_projection 仍保持原来的评测 dtype。
    force_module.imu_encoder.float()
    force_module.imu_encoder.eval()
    force_module.imu_encoder.requires_grad_(False)
    print(f"[*] Loaded ForceFiLMGenerator weights into `{force_module_name}` from: {checkpoint_path}")


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # 统一注册仓库里的 OpenVLA AutoClass 映射。
    # 这样当 checkpoint 是本地目录时，我们就可以优先使用“当前仓库代码”，
    # 而不是执行 checkpoint 目录里自带的 `modeling_prismatic.py`。
    register_openvla_auto_classes()

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        **build_pretrained_load_kwargs(
            cfg.pretrained_checkpoint,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            # force-FiLM 评测时需要在模型构造阶段启用动态 FiLM 结构，
            # 之后再额外加载训练好的 ForceFiLMGenerator 小权重。
            use_force_film=getattr(cfg, "use_force_film", False),
            # 评测端不再通过 best_model.ckpt 初始化 IMU encoder；
            # 完整 ForceFiLMGenerator 会在下方直接从 force_film_generator.pt 恢复。
            force_encoder_checkpoint_path=getattr(cfg, "force_encoder_checkpoint_path", None),
            force_encoder_embedding_type=getattr(cfg, "force_encoder_embedding_type", "mmcl"),
            force_history_length=getattr(cfg, "force_history_length", 250),
        ),
    )

    # 评测端只从 force_film_generator.pt 完整恢复 ForceFiLMGenerator。
    # 不再加载 best_model.ckpt，也不再先把 output_projection 清零。
    if getattr(cfg, "use_force_film", False):
        force_film_checkpoint = getattr(cfg, "force_film_checkpoint", None)
        if force_film_checkpoint is None:
            raise ValueError("`force_film_checkpoint` must be provided when `use_force_film=True`.")
        load_force_film_generator_weights(vla, force_film_checkpoint)

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    # 这里单独再注册一次，是为了兼容“先拿 processor、后拿 model”的调用顺序。
    register_openvla_auto_classes()

    # processor 的本地加载策略与模型保持一致：
    # 本地路径优先仓库代码，远端路径保留原来的 remote-code 行为。
    processor = AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint,
        **build_pretrained_load_kwargs(cfg.pretrained_checkpoint),
    )
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(
    vla,
    processor,
    base_vla_name,
    obs,
    task_label,
    unnorm_key,
    center_crop=False,
    suction_wrench=None,
):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    if suction_wrench is not None:
        # suction_wrench 是连续力/力矩历史窗口，不能跟图像 token 一起转成 bfloat16。
        # 这里保持 float32，并补 batch 维度，形状固定为 [1, 6, 250]。
        suction_wrench = torch.as_tensor(suction_wrench, dtype=torch.float32, device=DEVICE)
        if suction_wrench.ndim == 2:
            suction_wrench = suction_wrench.unsqueeze(0)
        force_history_length = int(getattr(vla.config, "force_history_length", 250))
        if suction_wrench.shape != (1, 6, force_history_length):
            raise ValueError(
                f"`suction_wrench` must have shape [6, {force_history_length}] or [1, 6, {force_history_length}], "
                f"got {tuple(suction_wrench.shape)}"
            )
        inputs["suction_wrench"] = suction_wrench

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
