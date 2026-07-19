"""
finetune.py

用于对 HF AutoClass 加载的 OpenVLA 模型做参数高效微调（LoRA），基于 HuggingFace PEFT 库实现。

面向新手的阅读提示（本文件只做"微调训练 + 保存 checkpoint"的最小闭环）：
    - 配置解析：使用 draccus，把 dataclass 的字段自动映射为命令行参数。
    - 分布式：使用 accelerate.PartialState 简化多卡信息获取；梯度同步用原生 DDP。
    - 数据：使用 RLDSDataset（Open X-Embodiment 生态常见的 TF/RLDS 流式数据）。
    - 模型：HuggingFace AutoClasses + OpenVLA 的自定义 config/processor/model 注册。
    - LoRA：用 PEFT 把可训练参数限制在 LoRA 适配器中；可选 4bit 量化进一步省显存。
    - 日志：主进程使用 wandb 记录 loss/accuracy/L1。
    - 保存：按步保存 adapter（或全量），并可选 merge LoRA 生成可直接推理的完整权重。

注意事项与简单 Benchmark：
    - 需要安装 PEFT（`pip install peft==0.11.1`）
    - LoRA 配置示例（无量化，rank=32，target_modules=all-linear）：
        + 48 GB GPU 可支持 batch size ≈ 12
        + 80 GB GPU 可支持 batch size ≈ 24

运行方式：
    - 单节点多卡（= $K）：torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - 若需覆盖配置：torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                        --data_root_dir <RLDS 数据目录> \
                        --dataset_name <数据集名称> \
                        --run_root_dir <日志目录> \
                        ...
"""

import os
import random
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import draccus
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.pytorch_utils import Conv1D

import wandb

# prompt_builder 负责把"任务指令/对话模板"拼成语言模型可消费的 prompt（不同模型版本可能模板不同）
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder

# collator 负责把不同长度的 token 序列 padding 到同一长度，便于组成 batch
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

# ActionTokenizer 把连续动作向量离散化为 token（训练时预测 token；评估/推理时再反解为连续动作）
from prismatic.vla.action_tokenizer import ActionTokenizer

# RLDSDataset/RLDSBatchTransform：读取 RLDS 格式数据并把每条轨迹样本转成训练所需张量
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.loading_utils import build_pretrained_load_kwargs, register_openvla_auto_classes

# 关闭 tokenizer 的并行线程提示：
# - HF tokenizer 在多进程/多卡训练中可能反复打印并行化警告/信息
# - 设为 false 可以让日志更干净（不影响训练正确性）
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    # vla_path：HuggingFace Hub 上的模型 repo id（也可以是本地目录路径）
    vla_path: str = "openvla/openvla-7b"                            # OpenVLA 模型在 HF Hub 上的路径

    # 各类目录路径
    # data_root_dir：RLDS 数据根目录（通常包含多个子数据集）
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Open-X 数据集所在目录
    # dataset_name：要训练的具体数据集名称（由 RLDSDataset 识别）
    dataset_name: str = "droid_wipe"                                # 微调所用数据集名称（如 `droid_wipe`）
    # run_root_dir：主输出目录（保存 processor、合并后的权重、统计信息等）
    run_root_dir: Path = Path("runs")                               # 日志与 checkpoint 目录
    # run_dir_name_override：如果提供，则最终 merged 权重目录名直接用这个值，不再使用自动 exp_id
    run_dir_name_override: Optional[str] = None                      # 覆盖最终权重目录名（仅影响 run_dir）
    # adapter_tmp_dir：如果用 LoRA，会先把 adapter 权重单独保存到这里
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # 合并 LoRA 前暂存 adapter 的目录

    # 微调相关超参
    # batch_size：每个进程（每张 GPU）的 DataLoader batch 大小
    batch_size: int = 16                                            # 单次 DataLoader batch 大小
    # max_steps：这里按"梯度步"（考虑 grad_accumulation_steps 后）计数
    max_steps: int = 200_000                                        # 最多训练步数
    # save_steps：每隔多少个"梯度步"保存一次 checkpoint
    save_steps: int = 5000                                          # 每隔多少步保存一次 checkpoint
    # learning_rate：AdamW 的学习率（仅作用于可训练参数；LoRA 时主要是 LoRA 权重）
    learning_rate: float = 5e-4                                     # 微调学习率
    # grad_accumulation_steps：梯度累积（显存不足时常用；等效增大全局 batch）
    grad_accumulation_steps: int = 1                                # 梯度累积步数
    # image_aug：是否开启 RLDSBatchTransform/数据管线中的图像增强
    image_aug: bool = True                                          # 是否启用图像增强
    # shuffle_buffer_size：RLDS shuffle buffer（越大越随机，但占用更多 CPU/RAM）
    shuffle_buffer_size: int = 100_000                              # RLDS dataloader 的 shuffle buffer，大可降内存
    # save_latest_checkpoint_only：历史遗留参数，当前训练保存逻辑已经不再读取这个字段。
    # 保留它只是为了兼容旧命令行 / 旧配置，避免旧脚本传入该参数时因为字段不存在而报错。
    # 现在实际控制 checkpoint 保存行为的是 save_steps、exit_after_save、immutable_adapter_checkpoints，
    # 以及 use_lora 分支；其中 immutable_adapter_checkpoints=True 时会保存独立的 step-* adapter 断点。
    save_latest_checkpoint_only: bool = True                        # 已废弃：当前不再影响 checkpoint 保存或覆盖策略
    # exit_after_save：True 时在每次 save_steps 保存完成后优雅退出进程，交给外层脚本续跑
    # 这样做的目的不是改变训练语义，而是把 TF/RLDS 数据管线的内存回收边界提升到进程级
    exit_after_save: bool = False                                   # 是否在每次保存后退出当前训练进程

    # LoRA 相关参数
    # use_lora：True 时仅训练 LoRA 适配器参数（显存/速度更友好）
    use_lora: bool = True                                           # 是否启用 LoRA 微调
    # lora_rank：LoRA 的秩 r（越大表达能力越强，但参数/显存更高）
    lora_rank: int = 32                                             # LoRA 低秩分解的秩
    # lora_dropout：LoRA 分支的 dropout（有助于正则化；很多场景设为 0）
    lora_dropout: float = 0.0                                       # LoRA dropout
    # use_quantization：4bit 量化加载 base 模型以省显存（仅支持 LoRA 场景）
    use_quantization: bool = False                                  # 是否对 VLA 做 4bit 量化（省显存但略降精度）

    # Force-FiLM 相关参数
    # use_force_film：True 时在视觉 backbone 内创建 ForceFiLMGenerator，并允许 forward 接收力历史窗口
    use_force_film: bool = False                                     # 是否启用基于 PRIMUS 力历史编码的 Force-FiLM 通路
    # train_force_film：True 时把 ForceFiLMGenerator 整体解冻，PRIMUS IMU encoder 和输出映射层会一起训练
    train_force_film: bool = False                                   # 是否在本次微调中训练 ForceFiLMGenerator
    # force_encoder_checkpoint_path：PRIMUS 预训练 IMU 编码器 checkpoint，用于编码 [6,250] 力历史窗口
    force_encoder_checkpoint_path: str = "/data1/ps/pretrained-imu-encoders/checkpoint/best_model.ckpt"
    # force_encoder_embedding_type：PRIMUS 多头输出选择；本实验使用和视频/文本对齐的 mmcl 表征
    force_encoder_embedding_type: str = "mmcl"
    # force_history_length：suction_wrench 历史窗口长度，processed_libero_suction_wrench_history_250 固定为 250
    force_history_length: int = 250
    # force_encoder_learning_rate：PRIMUS / IMU encoder backbone 的独立学习率，要明显小于 LoRA 主学习率
    force_encoder_learning_rate: float = 1e-5
    # force_projection_learning_rate：ForceFiLMGenerator.output_projection 的独立学习率，降低大输出层的 NaN 风险
    force_projection_learning_rate: float = 2e-5
    # immutable_adapter_checkpoints：True 时每个保存边界写入独立 step-* 目录，避免同一路径覆盖导致外层续训误判
    immutable_adapter_checkpoints: bool = False                      # 是否使用 adapter/checkpoints/step-* 不可变断点目录

    # 断点续训配置
    # resume_from_checkpoint：如果提供，从此路径加载 adapter 权重 + 续训状态继续训练
    # 路径应指向 adapter 目录（包含 adapter_config.json、adapter_model.safetensors、resume_state/）
    # 设为 None 则从头开始训练
    resume_from_checkpoint: Optional[str] = None                      # 断点续训的 adapter checkpoint 路径

    # 指标追踪配置
    # wandb_project/entity：Weights & Biases 的项目与组织
    wandb_project: str = "openvla"                                  # W&B Project 名称
    wandb_entity: str = "stanford-voltron"                          # W&B Entity
    # run_id_note：你自定义的备注，会拼到 exp_id 里便于区分实验
    run_id_note: Optional[str] = None                               # 在 run 名称上附加的备注

    # fmt: on


def build_rank_rng_state() -> dict:
    """收集当前 rank 的随机数状态，供 checkpoint 保存与恢复使用。"""
    # 为什么这里要同时保存 Python / NumPy / Torch / CUDA 四套随机状态：
    # 1. Python 的 `random` 可能会被某些第三方库内部使用；
    # 2. NumPy 常被数据处理或评估代码调用；
    # 3. Torch CPU RNG 会影响 CPU 侧随机逻辑；
    # 4. Torch CUDA RNG 会直接影响 GPU 上的 dropout、随机采样等行为。
    #
    # 只有把这些状态都记下来，续训时才尽量贴近中断前的训练轨迹。
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }


def restore_rank_rng_state(rng_state: dict) -> None:
    """恢复当前 rank 的随机数状态。"""
    # 这里的恢复顺序本身没有强约束，但把四套状态全部恢复可以最大程度减少续训随机漂移。
    random.setstate(rng_state["python_rng_state"])
    np.random.set_state(rng_state["numpy_rng_state"])
    torch.random.set_rng_state(rng_state["torch_rng_state"])
    torch.cuda.set_rng_state(rng_state["cuda_rng_state"])


def build_resume_state_dir(adapter_dir: Path) -> Path:
    """返回某个 adapter checkpoint 对应的 resume_state 目录。"""
    return adapter_dir / "resume_state"


def build_rank_rng_state_path(resume_state_dir: Path, rank: int) -> Path:
    """返回当前 rank 专属随机状态文件路径。"""
    return resume_state_dir / f"rng_state_rank_{rank}.pt"


def build_latest_adapter_marker_path(adapter_tmp_dir: Path) -> Path:
    """返回"最近一次成功保存的真实 adapter 目录"标记文件路径。"""
    # 这个文件只服务于 shell 的分段续训逻辑：
    # - adapter_tmp_dir 本身只是"实验级父目录"；
    # - 真实可恢复的 adapter 在更深一层的 exp_id 子目录里；
    # - shell 不应该自己猜 exp_id，而应该读取训练脚本在成功保存后写下的真实路径。
    return adapter_tmp_dir / ".latest_adapter_dir"


def build_step_adapter_checkpoint_dir(adapter_tmp_dir: Path, completed_gradient_steps: int) -> Path:
    """返回联合训练使用的不可变 LoRA+ForceFiLM adapter checkpoint 目录。"""
    # 这个目录只在某个 save_steps 边界完整保存成功后才会变成正式目录。
    # 目录名直接包含 step 数，外层 shell 就能用目录名和 resume_state 双重校验断点是否向前推进。
    return adapter_tmp_dir / "checkpoints" / f"step-{completed_gradient_steps}"


def build_tmp_step_adapter_checkpoint_dir(adapter_tmp_dir: Path, completed_gradient_steps: int) -> Path:
    """返回联合训练保存过程中的临时 checkpoint 目录。"""
    # 先写 .tmp-step-*，成功后再原子式 rename 成 step-*。
    # 这样即使训练进程在保存中途崩溃，外层 shell 也不会把半成品当成可恢复 checkpoint。
    return adapter_tmp_dir / "checkpoints" / f".tmp-step-{completed_gradient_steps}"


def locate_force_film_generator(model: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    """在当前模型里定位唯一的 ForceFiLMGenerator。"""
    # 这个函数同时兼容普通模型、PEFT 包装模型和 DDP 包装模型：
    # - named_modules() 会递归遍历所有子模块；
    # - 我们用模块名后缀和类名双保险匹配，避免某一层包装改变前缀后找不到。
    candidates = []
    for module_name, module in model.named_modules():
        if module_name and (
            module_name.endswith("force_film_generator") or module.__class__.__name__ == "ForceFiLMGenerator"
        ):
            candidates.append((module_name, module))

    # 同一个模块可能因为 PEFT 包装别名被遍历到多次，这里按对象 id 去重。
    dedup_candidates = []
    seen_module_ids = set()
    for module_name, module in candidates:
        if id(module) not in seen_module_ids:
            seen_module_ids.add(id(module))
            dedup_candidates.append((module_name, module))

    if len(dedup_candidates) == 0:
        raise ValueError(
            "train_force_film=True requires a ForceFiLMGenerator, but the current model does not expose one. "
            "Please also set use_force_film=True."
        )
    if len(dedup_candidates) != 1:
        candidate_names = ", ".join(name for name, _ in dedup_candidates)
        raise ValueError(f"Expected exactly one ForceFiLMGenerator, but found: {candidate_names}")

    return dedup_candidates[0]


def unfreeze_force_film_generator(model: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    """解冻整个 ForceFiLMGenerator，让 PRIMUS backbone 和输出映射层都可以和 LoRA 一起训练。"""
    force_module_name, force_module = locate_force_film_generator(model)

    # PEFT 会把 base model 权重整体冻结；这里显式只把 ForceFiLMGenerator 打开。
    # 这样训练集合就是：LoRA adapter 参数 + PRIMUS IMU encoder 参数 + output_projection 参数。
    force_module.requires_grad_(True)
    # 显式切到 train 模式，确保 PRIMUS 里的 GroupNorm/GRU/Linear 和输出映射都按训练路径运行。
    force_module.train()

    trainable_force_params = [
        name for name, param in model.named_parameters()
        if param.requires_grad and name.startswith(f"{force_module_name}.")
    ]
    if len(trainable_force_params) == 0:
        raise ValueError(f"ForceFiLMGenerator `{force_module_name}` was found, but no parameters are trainable.")

    return force_module_name, force_module


def collect_lora_target_linear_module_names(model: torch.nn.Module) -> list[str]:
    """收集 OpenVLA 主干里可以注入 LoRA 的线性层名字，并排除 ForceFiLMGenerator。"""
    # 不再使用 PEFT 的 `target_modules="all-linear"` 魔法字符串。
    # 原因是开启 Force-FiLM 后，视觉 block 会被替换成 FiLMWrappedViTBlock；
    # 在 PEFT 0.11.1 中，"all-linear" 可能误把这个自定义包装 block 当成 target，
    # 但 PEFT 只支持给 Linear/Embedding/Conv2d/Conv1D 等基础层注入 LoRA。
    #
    # 这里改成我们自己显式收集：
    # 1. 只收集真正支持 LoRA 的线性层；
    # 2. 不收集 ForceFiLMGenerator 内部模块，因为 PRIMUS 和输出映射层应该直接全参训练；
    # 3. 返回完整模块路径，避免 suffix 匹配误伤同名容器。
    target_module_names: list[str] = []
    supported_lora_module_types = (torch.nn.Linear, Conv1D)

    for module_name, module in model.named_modules():
        if not module_name:
            continue

        # ForceFiLMGenerator 里包含 PRIMUS IMU encoder 和输出映射层，这两部分都不套 LoRA。
        # 它们会在 PEFT 包装完成后通过 `unfreeze_force_film_generator(...)` 直接全参训练。
        if "force_film_generator" in module_name:
            continue

        if isinstance(module, supported_lora_module_types):
            target_module_names.append(module_name)

    if len(target_module_names) == 0:
        raise ValueError("No supported Linear/Conv1D modules found for LoRA injection.")

    return target_module_names


def save_force_film_generator_state(
    *,
    checkpoint_dir: Path,
    force_module_name: Optional[str],
    force_module: Optional[torch.nn.Module],
    completed_gradient_steps: int,
) -> None:
    """把完整 ForceFiLMGenerator 权重保存到 LoRA checkpoint 旁边。"""
    # LoRA 的 save_pretrained 不会自动保存任意额外可训练模块。
    # 所以联合训练时必须手动把 ForceFiLMGenerator 单独落盘，否则 resume/eval 都会丢掉这部分学习结果。
    if force_module_name is None or force_module is None:
        return

    torch.save(
        {
            "force_module_name": force_module_name,
            "state_dict": force_module.state_dict(),
            "completed_gradient_steps": completed_gradient_steps,
        },
        checkpoint_dir / "force_film_generator.pt",
    )


def load_force_film_generator_state(
    *,
    checkpoint_dir: Path,
    force_module: torch.nn.Module,
) -> None:
    """从 LoRA checkpoint 旁边恢复 ForceFiLMGenerator 权重。"""
    force_checkpoint_path = checkpoint_dir / "force_film_generator.pt"
    if not force_checkpoint_path.exists():
        raise FileNotFoundError(
            "train_force_film=True resume requires ForceFiLMGenerator weights, "
            f"but missing: {force_checkpoint_path}"
        )

    force_checkpoint = torch.load(force_checkpoint_path, map_location="cpu", weights_only=False)
    force_module.load_state_dict(force_checkpoint["state_dict"])


def move_optimizer_state_to_device(optimizer: AdamW, device_id: int) -> None:
    """把 optimizer state 里的张量移动到当前进程负责的 GPU。"""
    # `optimizer.load_state_dict(...)` 并不会总是自动把内部状态迁移到参数所在设备。
    # 如果这里不手动搬运，常见结果是：
    # - 模型参数在 GPU；
    # - AdamW 的动量 / 方差缓冲还留在 CPU；
    # 随后第一次 `optimizer.step()` 就可能报设备不一致错误。
    target_device = torch.device(f"cuda:{device_id}")
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(target_device, non_blocking=True)


def build_trainable_optimizer_groups(
    model: torch.nn.Module,
    cfg: FinetuneConfig,
    force_module: Optional[torch.nn.Module],
) -> Tuple[list, list]:
    """按模块来源拆分 AdamW 参数组，并返回所有可训练参数列表。"""
    # 训练稳定性的核心是“不同模块用不同学习率”：
    # - LoRA 参数可以沿用较大的主学习率；
    # - PRIMUS 是已经预训练好的 IMU encoder，只需要小步微调；
    # - output_projection 直接生成大量 gamma/beta，也需要低学习率避免数值突然放大。
    force_encoder_param_ids = set()
    force_projection_param_ids = set()

    if force_module is not None:
        if hasattr(force_module, "imu_encoder"):
            force_encoder_param_ids = {id(param) for param in force_module.imu_encoder.parameters()}
        if hasattr(force_module, "output_projection"):
            force_projection_param_ids = {id(param) for param in force_module.output_projection.parameters()}

    main_params = []
    force_encoder_params = []
    force_projection_params = []

    for param in model.parameters():
        if not param.requires_grad:
            continue

        param_id = id(param)
        if param_id in force_encoder_param_ids:
            force_encoder_params.append(param)
        elif param_id in force_projection_param_ids:
            force_projection_params.append(param)
        else:
            main_params.append(param)

    optimizer_groups = []
    if main_params:
        optimizer_groups.append({"params": main_params, "lr": cfg.learning_rate})
    if force_encoder_params:
        optimizer_groups.append({"params": force_encoder_params, "lr": cfg.force_encoder_learning_rate})
    if force_projection_params:
        optimizer_groups.append({"params": force_projection_params, "lr": cfg.force_projection_learning_rate})

    trainable_params = main_params + force_encoder_params + force_projection_params
    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found. "
            "If this is a LoRA resume run, please check whether the adapter was loaded with `is_trainable=True`."
        )

    print(
        "AdamW parameter groups: "
        f"main_params={sum(param.numel() for param in main_params)}@lr={cfg.learning_rate}, "
        f"force_encoder_params={sum(param.numel() for param in force_encoder_params)}"
        f"@lr={cfg.force_encoder_learning_rate}, "
        f"force_projection_params={sum(param.numel() for param in force_projection_params)}"
        f"@lr={cfg.force_projection_learning_rate}"
    )
    return optimizer_groups, trainable_params


def merge_latest_lora_into_run_dir(
    cfg: "FinetuneConfig",
    *,
    adapter_dir: Path,
    run_dir: Path,
    processor,
    dataset_statistics,
    completed_gradient_steps: int,
) -> None:
    """在训练正常结束后，把最新 LoRA adapter 合并成可直接推理的完整权重。"""
    print(
        "Starting final LoRA merge after normal training completion. "
        f"Source adapter dir: {adapter_dir}"
    )

    # 最终 merge 时重新加载一份 base 模型，避免直接在训练中的 DDP/LoRA 包装模型上操作。
    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        **build_pretrained_load_kwargs(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # merge 阶段必须保持和训练阶段一致的模型结构，避免结构不一致导致权重挂载失败。
            # 如果本轮训练使用了 Force-FiLM，最终合并权重必须构建同样的 Force-FiLM 结构。
            # 否则下面加载 force_film_generator.pt 时，模型里根本没有对应模块。
            use_force_film=cfg.use_force_film,
            force_encoder_checkpoint_path=cfg.force_encoder_checkpoint_path,
            force_encoder_embedding_type=cfg.force_encoder_embedding_type,
            force_history_length=cfg.force_history_length,
        ),
    )
    # merge 阶段同样从 base checkpoint 构建 Force-FiLM，新模块需要在加载 LoRA/Force-FiLM 小权重前真实落地。
    if cfg.use_force_film:
        base_vla.vision_backbone.load_pretrained_force_encoder()
        base_vla.vision_backbone.zero_initialize_force_film_output_layer()

    # 从最新 adapter 目录加载 LoRA 权重，并将其合并回 base 模型。
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
    merged_vla = merged_vla.merge_and_unload()

    # LoRA merge 只处理 adapter 权重；ForceFiLMGenerator 是我们额外训练的普通 nn.Module。
    # 因此这里在保存最终 merged checkpoint 前，手动把联合训练得到的 ForceFiLM 权重挂回完整模型。
    if cfg.train_force_film:
        merged_force_module_name, merged_force_module = locate_force_film_generator(merged_vla)
        load_force_film_generator_state(checkpoint_dir=adapter_dir, force_module=merged_force_module)
        save_force_film_generator_state(
            checkpoint_dir=run_dir,
            force_module_name=merged_force_module_name,
            force_module=merged_force_module,
            completed_gradient_steps=completed_gradient_steps,
        )

    # 最终 merged checkpoint 固定落到 run_dir，便于评测脚本直接引用。
    save_dataset_statistics(dataset_statistics, run_dir)
    processor.save_pretrained(run_dir)
    merged_vla.save_pretrained(run_dir)
    print(
        "Saved final merged checkpoint after normal completion "
        f"for Step {completed_gradient_steps} at: {run_dir}"
    )

    # 显式释放临时 merge 对象，减少主进程收尾阶段的内存滞留。
    del merged_vla
    del base_vla


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # 打印关键信息：要微调哪个模型、用哪个数据集
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] 确认至少有一块 GPU 并初始化分布式上下文
    # LoRA 仍然建议跑在 GPU 上，accelerate.PartialState 帮我们管理多卡分布式
    # 这里默认"必须有 GPU"：
    # - OpenVLA 体积较大，即便 LoRA 也通常需要 GPU 才现实
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"

    # PartialState 会：
    # - 读取 torchrun/accelerate 注入的环境变量
    # - 提供当前进程的 rank、local rank、是否主进程等信息
    distributed_state = PartialState()

    # local_process_index 就是当前机器上的 local_rank；用它选择"本进程绑定哪张 GPU"
    torch.cuda.set_device(device_id := distributed_state.local_process_index)

    # 主动清一下缓存，减少前序程序残留显存导致的 OOM 概率（不保证一定释放所有占用）
    torch.cuda.empty_cache()

    # 生成唯一实验 ID，并确定日志与模型目录
    # 根据"模型名 + 数据集 + 总 batch + 学习率"等关键信息拼 run 名，方便区分不同实验
    # exp_id：把关键超参拼成字符串，用作 run 目录名/日志名
    # 这样不用打开 wandb 也能从目录名大概看出配置
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        # 记录 LoRA 关键超参（rank/dropout）
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_force_film:
        # Force-FiLM 现在固定调制全部视觉 ViT blocks，不再把 block selection 写进实验名。
        exp_id += "+force-film"
    if cfg.train_force_film:
        # 记录 ForceFiLMGenerator 使用 PRIMUS 力编码器；只打开 use_force_film 但不训练时不会加这个标记。
        exp_id += f"+train-force-primus-{cfg.force_encoder_embedding_type}-h{cfg.force_history_length}"
    if cfg.use_quantization:
        # 标记 4bit 量化
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        # 用户自定义备注
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        # 标记开启图像增强
        exp_id += "--image_aug"

    # 创建目录、准备好存放 processor / adapter 的位置
    # run_dir 保存 processor + 模型，adapter_dir 用来暂存 LoRA 适配器权重
    # run_dir：主输出目录（合并后的模型、processor、统计文件等）
    # adapter_dir：LoRA adapter 的输出目录（训练中保存更小、更省空间）
    run_dir_name = cfg.run_dir_name_override if cfg.run_dir_name_override is not None else exp_id
    run_dir = cfg.run_root_dir / run_dir_name
    computed_adapter_dir = cfg.adapter_tmp_dir / exp_id
    # 续训时优先沿用用户显式传入的 checkpoint 目录。
    # 这样做有两个好处：
    # 1. 后续保存的新 checkpoint 会继续落到同一个 adapter 目录，不会“断点续训后另开新目录”；
    # 2. 即使外部脚本后面改了 adapter_tmp_dir 的命名规则，只要传入旧目录也能继续接着跑。
    adapter_dir = Path(cfg.resume_from_checkpoint) if cfg.resume_from_checkpoint is not None else computed_adapter_dir
    # final merge 需要知道“最后一次完整保存的 adapter 在哪里”。
    # 普通 LoRA 路径下它就是 adapter_dir；不可变 step-* 路径下会在每次保存后更新为最新 step 目录。
    latest_adapter_dir_for_merge = adapter_dir

    # exist_ok=True：目录存在时不报错，便于断点续跑/覆盖
    os.makedirs(run_dir, exist_ok=True)

    # 仅在 LoRA 场景下才构建 4bit 量化配置
    # 仅在 LoRA 下才支持 4bit 量化，节省显存但会有轻微精度损失
    # quantization_config：给 transformers.from_pretrained 用的量化加载配置
    # - None：表示正常 bfloat16 加载
    # - BitsAndBytesConfig：表示用 bitsandbytes 以 4bit 量化权重加载
    quantization_config = None
    if cfg.use_quantization:
        # 本脚本限制：只有 LoRA 微调才允许量化
        # 原因：全参训练 + 4bit 量化通常不成立（梯度/更新都不适配）
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"

        # nf4：常见的 4bit 量化格式；compute_dtype=bfloat16 表示计算用 bf16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # === 1) 先"登记映射"：把 OpenVLA 接到 Transformers 的 Auto* 工厂体系里（这里只登记，不会加载权重）===
    # Transformers 的 AutoConfig / AutoProcessor / AutoModel... 是"工厂"：
    # 它们会先去 cfg.vla_path 这个目录（或 HF Hub repo）读配置文件（例如 config.json），
    # 再根据配置内容决定：应该用哪个 Config 类、哪个 Processor 类、哪个 Model 类来实例化。
    #
    # 统一使用仓库内的注册函数，避免训练入口和评测入口各自维护一份映射表。
    # 这个函数会一次性登记：
    # - AutoConfig        -> OpenVLAConfig
    # - AutoImageProcessor -> PrismaticImageProcessor
    # - AutoProcessor     -> PrismaticProcessor
    # - AutoModelForVision2Seq -> OpenVLAForActionPrediction
    register_openvla_auto_classes()

    # === 2) 加载 processor（不加载模型权重）===
    # cfg.vla_path 不是"只有权重"的路径，而是"模型仓库/目录路径"：里面通常还包含 tokenizer、图像预处理等配置文件。
    # 本地 checkpoint 路径时，我们显式优先使用"仓库里的类定义"，
    # 而不是让 HF 去执行 checkpoint 目录自带的 `processing_prismatic.py`。
    # 这样可以确保我们新增的安全包装层和配置字段在训练时真正生效。
    processor = AutoProcessor.from_pretrained(
        cfg.vla_path,
        **build_pretrained_load_kwargs(cfg.vla_path),
    )

    # === 3) 加载模型（这里才会加载权重）===
    # 这句会根据 cfg.vla_path 中的配置 + 上面登记的映射，构建 OpenVLAForActionPrediction 并加载权重。
    # - torch_dtype=bfloat16：权重/计算以 bf16 为主（通常比 fp16 更稳）
    # - quantization_config：可选 4bit 量化
    # - low_cpu_mem_usage=True：减少 CPU 峰值内存
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        **build_pretrained_load_kwargs(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            # 联合训练时必须在加载阶段就打开 Force-FiLM。
            # 这会让模型构建 ForceFiLMGenerator，并让视觉 forward 能接收 suction_wrench。
            use_force_film=cfg.use_force_film,
            force_encoder_checkpoint_path=cfg.force_encoder_checkpoint_path,
            force_encoder_embedding_type=cfg.force_encoder_embedding_type,
            force_history_length=cfg.force_history_length,
        ),
    )
    # HuggingFace low_cpu_mem_usage 会在 from_pretrained 后处理缺失的新模块参数。
    # Force-FiLM 属于 base checkpoint 里没有的新结构，所以这里在 .to(device) 前再显式落地一次 PRIMUS 和输出层。
    if cfg.use_force_film:
        vla.vision_backbone.load_pretrained_force_encoder()
        vla.vision_backbone.zero_initialize_force_film_output_layer()

    # 量化情况下由 BitsAndBytes 处理设备放置；非量化时手动移动到当前 GPU
    # 量化路径会自动固定在 GPU 上，所以只在"非量化"场景下手动 .to(device)
    # 设备放置（很重要）：
    # - 量化路径：bitsandbytes/transformers 会用自己的 device map 逻辑，通常已放到 GPU
    #             prepare_model_for_kbit_training 会做一些训练前准备（如 layernorm 处理等）
    # - 非量化路径：手动把模型整体挪到本进程绑定的 GPU
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] 基于 PEFT 的 LoraConfig 包装模型，默认对 OpenVLA 主干线性层插入 LoRA 模块
    # LoRA 通过给所有线性层加秩 r 的低秩矩阵，显著减少可训练参数数量
    # LoRA 包装：
    # - target_modules：显式列出 OpenVLA 主干 Linear/Conv1D 层，避免误包 FiLMWrappedViTBlock
    # - ForceFiLMGenerator 不放进 LoRA target；它会在后面直接解冻，PRIMUS 和输出映射层都全参训练
    # - lora_alpha：缩放系数；这里取 min(r, 16) 属于常见经验设置
    # - init_lora_weights="gaussian"：初始化方式
    if cfg.use_lora:
        lora_target_modules = collect_lora_target_linear_module_names(vla)
        print(
            "Collected LoRA target modules before PEFT wrapping: "
            f"{len(lora_target_modules)} modules; first targets: {lora_target_modules[:8]}"
        )
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=lora_target_modules,
            init_lora_weights="gaussian",
        )

        if cfg.resume_from_checkpoint is not None:
            # 断点续训：从已保存的 checkpoint 加载 LoRA 权重和配置
            # PeftModel.from_pretrained 会读取 adapter_config.json 恢复 LoRA 结构，
            # 并加载 adapter_model.safetensors 恢复训练过的权重。
            #
            # [关键点]
            # PEFT 0.11.1 中 `is_trainable=False` 是默认值，这意味着如果这里不显式传 True，
            # 加载出来的 adapter 会处于“只推理、不参与训练”的冻结状态。
            # 断点续训场景必须强制设成可训练，否则看起来像恢复成功，实际上 LoRA 参数不会继续更新。
            print(f"Loading LoRA adapter from checkpoint: {cfg.resume_from_checkpoint}")
            vla = PeftModel.from_pretrained(vla, cfg.resume_from_checkpoint, is_trainable=True)
            print("Loaded LoRA adapter weights from checkpoint.")
        else:
            # 从头训练：用 get_peft_model 初始化新的随机 LoRA 权重
            # get_peft_model 会把 base model 包一层 PEFT wrapper，并设置好 requires_grad
            vla = get_peft_model(vla, lora_config)

        # 先打印 LoRA adapter 的可训练参数/总参数占比；ForceFiLMGenerator 会在下面单独打印。
        vla.print_trainable_parameters()

    force_module_name = None
    force_module = None
    if cfg.train_force_film:
        # 防呆：训练 ForceFiLMGenerator 时必须同时启用 Force-FiLM 结构。
        if not cfg.use_force_film:
            raise ValueError("train_force_film=True requires use_force_film=True.")

        # PEFT 包装之后再解冻 ForceFiLMGenerator。
        # 原因是 get_peft_model 会冻结 base model；如果在它之前解冻，随后仍可能被 PEFT 再次冻结。
        force_module_name, force_module = unfreeze_force_film_generator(vla)
        force_param_count = sum(param.numel() for param in force_module.parameters())
        # 分开统计 PRIMUS backbone 和 output_projection 的可训练参数量。
        # 这条日志是为了启动训练时就能确认：IMU encoder 已经不再被冻结。
        imu_trainable_param_count = (
            sum(param.numel() for param in force_module.imu_encoder.parameters() if param.requires_grad)
            if hasattr(force_module, "imu_encoder")
            else 0
        )
        output_projection_trainable_param_count = (
            sum(param.numel() for param in force_module.output_projection.parameters() if param.requires_grad)
            if hasattr(force_module, "output_projection")
            else 0
        )
        force_trainable_param_count = sum(param.numel() for param in force_module.parameters() if param.requires_grad)
        print(
            "ForceFiLMGenerator will be trained together with LoRA. "
            f"module={force_module_name}, params={force_param_count}, "
            f"trainable_params={force_trainable_param_count}, "
            f"imu_encoder_trainable_params={imu_trainable_param_count}, "
            f"output_projection_trainable_params={output_projection_trainable_param_count}"
        )
        if cfg.resume_from_checkpoint is not None:
            # 断点续训时，PEFT 只会自动恢复 LoRA adapter；
            # ForceFiLMGenerator 是普通子模块，所以要从同一个 checkpoint 目录手动恢复。
            load_force_film_generator_state(
                checkpoint_dir=Path(cfg.resume_from_checkpoint),
                force_module=force_module,
            )
            print(f"Loaded ForceFiLMGenerator weights from checkpoint: {cfg.resume_from_checkpoint}")

    # 使用原生 DDP 包裹模型，实现多卡梯度同步
    # accelerate 负责跨节点通信，这里依旧用原生 DDP 做梯度同步
    # DDP 包装：
    # - device_ids=[device_id]：每个进程只负责一张 GPU
    # - find_unused_parameters=True：允许某些分支参数在某次 forward 没用到（更稳但略慢）
    # - gradient_as_bucket_view=True：可能减少显存开销/提升性能（依赖 PyTorch 版本）
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # 构建优化器。这里不再把所有可训练参数塞进同一个学习率：
    # - LoRA adapter 沿用 cfg.learning_rate；
    # - PRIMUS / IMU encoder 使用 cfg.force_encoder_learning_rate；
    # - output_projection 使用 cfg.force_projection_learning_rate。
    # 这样能保留“OpenVLA LoRA + PRIMUS 全参训练 + projection 全参训练”的训练目标，同时降低 NaN 风险。
    optimizer_groups, trainable_params = build_trainable_optimizer_groups(
        model=vla,
        cfg=cfg,
        force_module=force_module,
    )

    # AdamW：解耦 weight decay 的 Adam（transformer 微调最常用的优化器之一）
    optimizer = AdamW(optimizer_groups, lr=cfg.learning_rate)

    # 初始化 checkpoint 变量为 None，后续 deque 初始化逻辑会用到
    checkpoint = None
    # 断点续训：加载 optimizer state、已完成梯度步数、指标滑窗、随机数状态。
    # completed_gradient_steps 表示“已经成功执行过多少次 optimizer.step()”。
    # 它和“当前正在处理第几个 micro-batch”不是一回事，后面所有日志、保存和停止条件都以它为准。
    completed_gradient_steps = 0
    if cfg.resume_from_checkpoint is not None:
        # 构建 resume_state/checkpoint.pt 的完整路径
        resume_state_dir = build_resume_state_dir(Path(cfg.resume_from_checkpoint))
        resume_ckpt_path = resume_state_dir / "checkpoint.pt"
        if resume_ckpt_path.exists():
            print(f"Loading resume state from: {resume_ckpt_path}")
            # 这里先按 CPU 读入共享 checkpoint，再手动把 optimizer state 迁移到当前 GPU。
            # 这么做有两个原因：
            # 1. 旧格式 checkpoint 里混有 CPU RNG 状态，直接 map 到 CUDA 可能让 RNG 恢复出错；
            # 2. 优化器状态需要迁移到当前 rank 对应的 GPU，而不是笼统地“所有张量都硬搬到同一设备”。
            checkpoint = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
            # 恢复 optimizer state（AdamW 的动量 m 和方差 v）
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device_id)
            # 兼容新老两种字段名：
            # - 新格式主字段是 completed_gradient_steps；
            # - 旧格式使用 gradient_step_idx。
            completed_gradient_steps = checkpoint.get(
                "completed_gradient_steps", checkpoint.get("gradient_step_idx", 0)
            )

            # 先尝试读取“每个 rank 自己的 RNG 状态文件”。
            # 这是新格式 checkpoint 的标准恢复路径，可以避免多卡续训时所有进程误用同一份 CUDA RNG。
            rank_rng_state_path = build_rank_rng_state_path(resume_state_dir, distributed_state.process_index)
            if rank_rng_state_path.exists():
                restore_rank_rng_state(torch.load(rank_rng_state_path, weights_only=False))
                print(
                    f"Restored RNG state for global rank {distributed_state.process_index} "
                    f"from: {rank_rng_state_path}"
                )
            # 如果没有 per-rank RNG 文件，则回退兼容旧格式 checkpoint。
            # 旧格式只有一份共享 CUDA RNG；这能保证“尽量接着训”，但不能保证多卡逐 rank 的随机轨迹完全一致。
            elif "rng_state" in checkpoint and "cuda_rng_state" in checkpoint:
                torch.random.set_rng_state(checkpoint["rng_state"])
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
                print(
                    "WARNING: Loaded legacy shared RNG state from checkpoint.pt. "
                    "Training can resume, but multi-rank RNG recovery is not exact."
                )

            print(f"Resumed training from completed step {completed_gradient_steps}. Optimizer state restored.")
            print(
                "NOTE: RLDS input pipeline state is not checkpointed. "
                "Resume continues from the latest saved model/optimizer step, not from an exact sample cursor."
            )
        else:
            print(f"WARNING: resume_from_checkpoint is set but no resume state found at {resume_ckpt_path}")
            print("Training will start from scratch with loaded adapter weights.")

    # 如果 checkpoint 已经达到或超过当前 max_steps，就没有必要继续进入训练循环了。
    # 提前返回可以避免“已经训满了却又多跑一遍”的误操作。
    if completed_gradient_steps >= cfg.max_steps:
        print(
            f"Checkpoint already contains {completed_gradient_steps} completed steps, "
            f"which is >= max_steps={cfg.max_steps}. Nothing to do."
        )
        return

    # 创建动作 tokenizer，用于离散 token 与连续动作互转
    # 用于把离散 token <-> 连续动作向量互相转换
    # ActionTokenizer 基于语言 tokenizer 构建动作 token 的编码规则
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # 加载 RLDS 数据集（遵循 Open X-Embodiment 格式）。若想改用常规 PyTorch Dataset，可参考下方注释代码；
    # 因 RLDS 自带无限流，不需要我们再手动套 epoch 循环，换自定义数据集时别忘了补上。
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    # RLDSBatchTransform 负责：图像增强、prompt 构造、动作编码等预处理逻辑
    # batch_transform：把 RLDS 原始样本（图像、文本、动作、元信息）加工成训练 batch
    # - image_transform：来自 processor 的图像预处理（resize/normalize 等）
    # - prompt_builder_fn：不同模型版本用不同 prompt 模板
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    # vla_dataset：一个可被 PyTorch DataLoader 迭代的数据集封装
    # 注意：内部可能是 TF/RLDS 的 streaming 管线，因此 num_workers 通常设 0
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        # resize_resolution：图像 resize 的目标分辨率（来自模型配置）
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] 保存动作归一化统计，供推理阶段反归一化使用
    # dataset_statistics：动作归一化/反归一化所需的均值方差等统计
    # 只让主进程写盘，避免多进程抢写同一文件
    # 单独缓存一份 dataset_statistics，避免后续任何控制流调整都去依赖训练末尾的 vla_dataset 对象。
    # 最终 merge 时只需要这份统计信息本身，不需要整个 dataset 继续活着。
    dataset_statistics = vla_dataset.dataset_statistics
    if distributed_state.is_main_process:
        save_dataset_statistics(dataset_statistics, run_dir)

    # 构建 collator + DataLoader
    # collator 会把不同长度的 token 序列 pad 到统一长度，方便堆叠成 batch
    # collator：
    # - model_max_length：最大序列长度
    # - pad_token_id：padding 用的 token id
    # - padding_side="right"：右侧 padding（更常见）
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    # DataLoader：
    # - sampler=None：这里不显式用 DistributedSampler，因为 RLDSDataset 自身可能处理了随机性/分片
    # - num_workers=0：避免 TF/RLDS 的内部线程与 PyTorch worker 叠加导致资源争抢
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # 这里设 0，是因为 RLDS/Tensorflow dataset 本身内部开线程
    )

    # 初始化 W&B 日志
    # wandb.init 也只在主进程做：
    # - 避免创建多个重复 run
    # - 避免多进程同时写日志
    if distributed_state.is_main_process:
        # 续训时在 wandb run name 中标记 "resumed"，便于区分新训练和续训
        run_name = f"ft+{exp_id}"
        if completed_gradient_steps > 0:
            run_name += f"+resumed-from-step{completed_gradient_steps}"
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_name)

    # 使用 deque 滑动窗口存最近的指标，方便在梯度累积时做平滑统计
    # recent_* 用来做"梯度累积窗口内的平均"，让日志更平滑
    # 如果是断点续训且有保存的滑窗数据，则从 checkpoint 恢复；否则初始化为空
    if completed_gradient_steps > 0 and checkpoint is not None:
        # 从 checkpoint 恢复滑窗，保持和中断前一致的平滑效果
        recent_losses = deque(checkpoint["recent_losses"], maxlen=cfg.grad_accumulation_steps)
        recent_action_accuracies = deque(
            checkpoint["recent_action_accuracies"], maxlen=cfg.grad_accumulation_steps
        )
        recent_l1_losses = deque(checkpoint["recent_l1_losses"], maxlen=cfg.grad_accumulation_steps)
    else:
        # 从头训练：使用空滑窗
        recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
        recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
        recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # 到这里为止，checkpoint 中后续还需要用到的内容都已经复制到目标对象里了：
    # - optimizer state 已恢复到 optimizer；
    # - completed_gradient_steps 已取出；
    # - RNG 状态已恢复；
    # - recent_* 滑窗已初始化完成。
    # 因此这里可以安全释放整份共享 checkpoint，避免它在续训场景下继续常驻 CPU RAM。
    if checkpoint is not None:
        del checkpoint
        checkpoint = None

    # 进入训练主循环
    # tqdm 进度条按"梯度步"推进：每做完一次 optimizer.step() 就 progress.update() 一次
    # initial=completed_gradient_steps：如果是断点续训，进度条从“已完成的真实更新步数”开始，而不是从 0 开始
    training_completed_normally = False
    # segment_restart_required=True 表示：
    # - 这次训练段已经在 save_steps 边界成功保存完 checkpoint；
    # - 但整体训练还没达到 max_steps；
    # - 因此当前进程应该以受控退出码结束，交给外层 shell 拉起下一段训练。
    segment_restart_required = False
    with tqdm.tqdm(total=cfg.max_steps, initial=completed_gradient_steps, leave=False) as progress:
        # 进入训练模式（启用 dropout 等训练行为）
        vla.train()

        # 先清空梯度（PyTorch 默认是累加梯度）
        optimizer.zero_grad()
        # enumerate(dataloader)：batch_idx 从 0 开始递增；RLDS 可能是"无限流"，所以靠 max_steps 退出
        for batch_idx, batch in enumerate(dataloader):
            # autocast：混合精度上下文，把部分算子用 bf16 执行以提速/省显存
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # 前向：
                # - input_ids/attention_mask：文本 token 及其 mask
                # - pixel_values：图像张量（显式转 bf16）
                # - labels：监督信号（动作 token 序列），由模型内部计算 loss
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    # suction_wrench 是 processed_libero_suction_wrench_history_250 额外提供的 [6,250] 力/力矩历史窗口。
                    # 只有启用 Force-FiLM 时才传入；否则保持原始 OpenVLA forward 行为不变。
                    suction_wrench=batch["suction_wrench"].to(device_id) if cfg.use_force_film else None,
                    labels=batch["labels"],
                )

                # output.loss：HF 模型约定返回的标量 loss
                loss = output.loss

            # 如果 loss 已经是 NaN/Inf，就不要继续 backward 或保存 checkpoint。
            # 这种 fail-fast 能防止坏梯度写进 optimizer state，避免后续断点全部被污染。
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite training loss at micro_batch={batch_idx}, "
                    f"completed_gradient_steps={completed_gradient_steps}: {loss.item()}. "
                    "Training is stopped before backward/optimizer.step to avoid writing a corrupted checkpoint."
                )

            # 先把 loss 除以累积步数，保证梯度幅度一致
            # 做梯度累积时记得把 loss 除以累积步数，保持总梯度幅度不变
            # 梯度累积的关键：
            # - 如果你累计 N 次 backward 后再 step，需要把每次的 loss 除以 N
            # - 这样等效于"更大 batch"的平均梯度，而不是把梯度放大 N 倍
            normalized_loss = loss / cfg.grad_accumulation_steps

            # 反向传播
            # backward：把计算图上的梯度累加到各参数的 .grad 里
            normalized_loss.backward()

            # 计算动作 token 精度与连续动作的 L1 误差
            # logits 中前半部分是视觉 patch，对应动作 token 的 slice 要跳过 patch 数量并丢掉最后一个 EOS
            # output.logits 的形状通常是 [B, T, V]：
            # - B：batch size
            # - T：序列长度（含视觉 token + 文本/动作 token）
            # - V：词表大小
            # OpenVLA 里序列前面常拼了视觉 patch token；训练动作预测时要把它们切掉
            # ":-1" 的 -1：丢掉最后一个位置（常对应 EOS 的 next-token 预测位置）
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]

            # argmax(dim=2)：对词表维度取最大概率 token id，得到预测的动作 token
            action_preds = action_logits.argmax(dim=2)

            # labels 通常是 [B, T_label]；这里右移 1 位以对齐 next-token 预测
            action_gt = batch["labels"][:, 1:].to(action_preds.device)

            # mask：只在"动作 token 区间"上计算指标（过滤掉 padding/特殊 token）
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # 计算离散动作准确率
            # correct_preds：逐位置判断预测是否正确，并且只在 mask 为 True 的位置计数
            correct_preds = (action_preds == action_gt) & mask

            # accuracy：正确数 / 有效动作 token 数
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # 计算连续动作的 L1 误差（需先反归一化）
            # decode_token_ids_to_actions -> 把动作 token 重新映射为归一化前的连续动作
            # 把动作 token 反解成连续动作向量：
            # - 这里先把 mask 后的 token 拉到 CPU + numpy，因为 decode 通常是 Python/NumPy 实现
            # - 反解得到的动作通常已"反归一化"回真实尺度（具体取决于 tokenizer 实现）
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )

            # L1 loss（MAE）：连续动作误差的直观度量
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # 把指标塞进滑动窗口，后续求平均
            # .item()：把 0-d tensor 取成 Python float，便于 deque/wandb 记录
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # 计算平滑指标：无梯度累积时等同当前值，有累积时为窗口平均
            # 平滑指标：窗口内做平均
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # 满足累积步数后执行优化器更新
            # 当 micro-batch 数凑够一次累积窗口，就更新一次参数
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                # step：根据当前累计梯度更新参数
                optimizer.step()

                # 清空梯度，为下一个累积窗口做准备
                optimizer.zero_grad()

                # 只有真正执行完一次 optimizer.step() 之后，才认为“完成了一个梯度步”。
                # 这一步是这次修复的核心：
                # - 日志步号；
                # - checkpoint 步号；
                # - 训练停止条件；
                # 都必须和真实更新次数完全对齐。
                completed_gradient_steps += 1

                # 进度条前进一步（按梯度步计数）
                progress.update()

                # 每 10 个真实梯度步向 W&B 汇报一次。
                # 这里把日志放到 optimizer.step() 之后，保证看到的 step 一定对应“已经更新过参数”的模型状态。
                if distributed_state.is_main_process and completed_gradient_steps % 10 == 0:
                    wandb.log(
                        {
                            "train_loss": smoothened_loss,
                            "action_accuracy": smoothened_action_accuracy,
                            "l1_loss": smoothened_l1_loss,
                        },
                        step=completed_gradient_steps,
                    )

                # 按配置的 save_steps 存一次 checkpoint（默认覆盖旧版本）。
                # 这里同样只在“真实完成参数更新之后”保存，避免 checkpoint 命名步号与模型实际状态错位。
                if completed_gradient_steps > 0 and completed_gradient_steps % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {completed_gradient_steps}")

                    # 若是 LoRA，先保存 adapter；否则直接保存全量模型。
                    # 普通 LoRA 保存仍然沿用旧逻辑：同一个 adapter_dir 反复覆盖。
                    # 联合训练建议打开 immutable_adapter_checkpoints：每个保存边界写入独立 step-* 目录。
                    if cfg.use_lora and cfg.immutable_adapter_checkpoints:
                        final_save_dir = build_step_adapter_checkpoint_dir(
                            cfg.adapter_tmp_dir, completed_gradient_steps
                        )
                        save_dir = build_tmp_step_adapter_checkpoint_dir(
                            cfg.adapter_tmp_dir, completed_gradient_steps
                        )

                        if final_save_dir.exists():
                            raise FileExistsError(
                                "Refusing to overwrite existing immutable checkpoint: "
                                f"{final_save_dir}"
                            )

                        # 只有主进程负责清理临时目录并创建父目录；其他 rank 等待后再写自己的 RNG。
                        if distributed_state.is_main_process:
                            if save_dir.exists():
                                shutil.rmtree(save_dir)
                            os.makedirs(save_dir, exist_ok=True)
                        dist.barrier()
                    else:
                        final_save_dir = adapter_dir if cfg.use_lora else run_dir
                        save_dir = final_save_dir

                    # save_dir：
                    # - LoRA 普通模式：adapter_dir
                    # - LoRA 不可变模式：先写 .tmp-step-*，完整后 rename 成 step-*
                    # - 非 LoRA：run_dir
                    resume_dir = build_resume_state_dir(save_dir)
                    final_resume_dir = build_resume_state_dir(final_save_dir)

                    # 先由每个 rank 保存自己的随机状态文件。
                    # 这样共享 checkpoint 只需要保存“所有 rank 都一样”的训练状态，
                    # 而容易因为进程不同而不同的 RNG，则单独分文件保存。
                    if cfg.use_lora:
                        os.makedirs(resume_dir, exist_ok=True)
                        torch.save(
                            build_rank_rng_state(),
                            build_rank_rng_state_path(resume_dir, distributed_state.process_index),
                        )

                    # 第一处 barrier：
                    # 等所有 rank 都把自己的 RNG 文件写完，再让主进程写共享 checkpoint。
                    dist.barrier()

                    # 只让主进程执行权重和共享状态写盘。
                    if distributed_state.is_main_process:
                        # 同步保存 processor 与权重
                        # 保存 processor（tokenizer、image processor 等）
                        processor.save_pretrained(run_dir)

                        # vla 是 DDP 包装；实际模型在 vla.module 里
                        vla.module.save_pretrained(save_dir)

                        # 保存续训共享状态（optimizer state、训练步数、指标滑窗等）。
                        # 注意这里不再把 RNG 一股脑塞进共享 checkpoint；
                        # 新格式下 RNG 已经按 rank 单独写盘。
                        if cfg.use_lora:
                            torch.save(
                                {
                                    "format_version": 2,
                                    "completed_gradient_steps": completed_gradient_steps,
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "recent_losses": list(recent_losses),
                                    "recent_action_accuracies": list(recent_action_accuracies),
                                    "recent_l1_losses": list(recent_l1_losses),
                                    "resume_source_adapter_dir": str(final_save_dir),
                                    "wandb_resume_mode": "new_resumed_run",
                                },
                                resume_dir / "checkpoint.pt",
                            )
                            save_force_film_generator_state(
                                checkpoint_dir=save_dir,
                                force_module_name=force_module_name,
                                force_module=force_module,
                                completed_gradient_steps=completed_gradient_steps,
                            )
                            print(
                                "Saved shared resume state for "
                                f"Step {completed_gradient_steps} at: {resume_dir / 'checkpoint.pt'}"
                            )

                            # 不可变 checkpoint 模式下，只有所有文件都写进 .tmp-step-* 后，
                            # 才把目录改名成正式 step-*。正式目录一出现，就代表这个保存边界完整成功。
                            if cfg.immutable_adapter_checkpoints:
                                save_dir.rename(final_save_dir)
                                print(
                                    "Promoted immutable adapter checkpoint for "
                                    f"Step {completed_gradient_steps} to: {final_save_dir}"
                                )

                            # 只有在 adapter、本次共享续训状态都已经真实写盘成功之后，
                            # 才更新"最新 adapter 路径"标记文件。
                            # 这样外层 shell 在收到退出码 20 时，读到的一定是可恢复的真实子目录，
                            # 而不是 adapter_tmp_dir 这种父目录。
                            build_latest_adapter_marker_path(cfg.adapter_tmp_dir).write_text(
                                str(final_save_dir), encoding="utf-8"
                            )
                            if cfg.immutable_adapter_checkpoints:
                                print(
                                    "Saved immutable resume state for "
                                    f"Step {completed_gradient_steps} at: {final_resume_dir / 'checkpoint.pt'}"
                                )

                    if cfg.use_lora:
                        latest_adapter_dir_for_merge = final_save_dir

                    # 第二处 barrier：
                    # 等主进程把 adapter / processor / 共享状态都写完，再继续训练。
                    dist.barrier()

                    # 分段续训模式：
                    # - 只有在 checkpoint 已经完整写盘后，才允许退出；
                    # - 只有在尚未达到 max_steps 时，才需要退出给外层脚本续跑。
                    # 这样可以保证：
                    # 1. 当前训练进度不会丢；
                    # 2. 中间段不会误触发最终 merge；
                    # 3. 断点恢复仍然基于同一套 adapter + resume_state 语义。
                    if cfg.exit_after_save and completed_gradient_steps < cfg.max_steps:
                        if distributed_state.is_main_process:
                            print(
                                "Checkpoint saved at "
                                f"step {completed_gradient_steps}; exiting current segment for restart."
                            )
                        segment_restart_required = True
                        break

                # 如果达到 max_steps，终止训练。
                # 终止条件同样只看“已经完成了多少次真实参数更新”。
                if completed_gradient_steps >= cfg.max_steps:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                    training_completed_normally = True
                    break
        else:
            # DataLoader 正常耗尽也视为一次“正常完成”。
            training_completed_normally = True

    # 分段续训的受控退出路径：
    # - 不做最终 merge，因为整体训练还没结束；
    # - 直接返回专用退出码 20，让外层 shell 识别为“正常分段结束，立即续跑下一段”。
    if segment_restart_required:
        raise SystemExit(20)

    # 只在训练正常结束后做一次最终 merge，避免训练过程中的周期性 CPU 内存暴涨。
    if cfg.use_lora and training_completed_normally:
        dist.barrier()
        if distributed_state.is_main_process:
            merge_latest_lora_into_run_dir(
                cfg,
                adapter_dir=latest_adapter_dir_for_merge,
                run_dir=run_dir,
                processor=processor,
                dataset_statistics=dataset_statistics,
                completed_gradient_steps=completed_gradient_steps,
            )
        dist.barrier()


if __name__ == "__main__":
    # 直接运行脚本时，draccus 会：
    # - 解析命令行参数覆盖 FinetuneConfig 默认值
    # - 把 cfg 传入 finetune(cfg)
    finetune()
