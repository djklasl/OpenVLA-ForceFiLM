"""
train.py

该脚本用于训练 Vision-Language-Action (VLA) Policy，会在预训练 VLM 基础上，利用 Open-X Embodiment 混合数据集继续训练。
训练完全基于原生 PyTorch，并通过 Fully-Sharded Data Parallel (FSDP) 在多 GPU（甚至多节点）上进行；默认假设 CUDA >= 11.0
以获得 BF16 混合精度能力。

先决条件提示：
    - 若需要手动指定 HF / TIMM 缓存目录：在运行前执行 `export HF_HOME="<PATH>"`
        => 示例（可写入 .bashrc 尾部）：`export HF_HOME="/mnt/fsx/skaramcheti/cache"`
    - 若想屏蔽 Tensorflow 的冗余日志：`export TF_CPP_MIN_LOG_LEVEL=3`

运行方式：
    - 单机单卡（调试）：torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - 单机多卡（= $K）：torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# 默认关闭 HF Tokenizer 的多线程，不然多进程训练时会有一堆警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Overwatch 是整个项目的日志 & 分布式状态封装，创建一次全局复用
overwatch = initialize_overwatch(__name__)


"""
dataclass 创建 TrainConfig 实例时的调用顺序为：
1) 先执行 dataclass 自动生成的 __init__，按照下面类中声明的每个字段名、类型和默认值创建参数列表，并把传入的值赋给对应的 self.xxx（例如 self.data_root_dir = data_root_dir 等）；
2) 然后自动调用本方法 __post_init__，在这里做二次处理和合法性校验。
只有在 @dataclass 类定义时声明的字段（即 def __post_init__ 之前的那些）才能被命令行参数解析器（如 Draccus/argparse）识别和赋值
"""
@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig（定义在 `prismatic/conf/vla.py`），可通过 --vla.type `VLARegistry.<VLA>.vla_id` 来切换预设
    #VLARegistry是Enum的子类，里面定义了所有的VLA实验配置
    #等效于 vla : VLAConfig = field(VLAConfig.get_choice_class("prism-dinosiglip-224px+mx-oxe-magic-soup-plus"))
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    )

    # 目录相关路径配置
    data_root_dir: Path = Path(                                     # Open-X 数据存放目录
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # 日志与 checkpoint 存储目录

    # 断点续训相关参数
    pretrained_checkpoint: Optional[Path] = None                    # 预训练 / checkpoint 绝对路径
    is_resume: bool = True                                          # 是否视作继续训练（需要配合 checkpoint 使用）
    resume_step: Optional[int] = None                               # 对应 checkpoint 的 global step
    resume_epoch: Optional[int] = None                              # 对应 checkpoint 的 epoch

    # 运行相关参数
    run_id: Optional[str] = None                                    # 日志 / W&B run 的唯一名称
    run_id_note: Optional[str] = None                               # 附加在 run_id 后的备注
    save_interval: int = 2500                                       # checkpoint 保存间隔（step）
    image_aug: bool = False                                         # 是否在训练时启用图像增强
    seed: int = 7                                                   # 随机种子（复现用）

    # HF Hub 授权（部分模型需要）
    hf_token: Union[str, Path] = Path(".hf_token")                  # 可以是 token 文件，也可以是环境变量名

    # 监控 / 记录相关配置
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # 需要启用的日志后端（如 jsonl、W&B 等）
    wandb_project: str = "openvla"                                  # W&B Project 名称
    wandb_entity: str = "stanford-voltron"                          # W&B Entity

    def __post_init__(self) -> None:
        # 方便下游代码直接通过 cfg.epochs 等访问
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # 分布式训练一定要满足“配置里写的 world size == 实际 torchrun 进程数”
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on

# [Draccus 魔法核心]
# @draccus.wrap() 是整个脚本的入口装饰器。它在后台默默做了三件大事：
# 1. 解析器生成 (Parser Generator): 
#    它会自动读取 `train` 函数的参数类型注解（即 `cfg: TrainConfig`）。
#    然后扫描 `TrainConfig` 这个 dataclass 的所有字段（包括嵌套的 VLAConfig），
#    自动生成一套完整的 argparse 命令行解析规则。
#    (你不再需要手写 parser.add_argument('--learning_rate', ...) 了！)
#
# 2. 命令行参数解析 (CLI Parsing):
#    当你运行脚本时 (python train.py --vla.learning_rate 5e-4 ...)，
#    它会截获终端输入，解析参数，并实例化一个填充好数据的 `TrainConfig` 对象。
#
# 3. 依赖注入与执行 (Injection & Execution):
#    最后，它把创建好的配置对象传给下面的 `cfg` ，并自动执行 train 函数。
@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    # 进入真正的训练入口（torchrun 每个进程都会执行一次）
    overwatch.info("OpenVLA Training :: Warming Up")

    # 每个进程绑定自己负责的 GPU，local_rank 由 torchrun 注入
    torch.cuda.set_device(device_id := overwatch.local_rank())
    # 清空显存缓存，防止上一次运行残留导致 OOM
    torch.cuda.empty_cache()

    # 处理 run_id 以及日志、模型输出目录
    vla_id = cfg.vla.vla_id
    # 如果用户没手动指定 run_id，就自动拼一个“模型+GPU 数+batch+随机种子”的名字
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # 初始化随机性 & 目录结构
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    # 兼容“把 HF Token 写在文件里”或“存到环境变量里”两种方式
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    # 统一随机种子，worker_init_fn 交给 DataLoader 用，保证多进程也能复现
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    # 创建日志 & checkpoint 目录，exist_ok=True 保证复运行也不会报错
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # 保存一份配置快照（同时生成 YAML / JSON，便于复现实验或上传 HF）
    if overwatch.is_rank_zero():
        # 只有 rank0 负责把 “本次训练的配置” 落盘，方便复现 & 上传 HF
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # 如果传入 checkpoint 就加载 VLA checkpoint，否则根据 base VLM ID/path 载入初始模型
    #   =>> 同时检查参数 dtype 是否保持 FP32
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        # [Validate] checkpoint 文件名里的 step/epoch 必须与 resume_step / resume_epoch 一致
        #   =>> 这是额外的校验，防止用错断点
        if cfg.is_resume:
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch

        #使用 load_vla 加载已经训练过的 OpenVLA 模型的checkpoint
        vlm = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=True)

    else:
        #使用 load 加载基础模型 (如 Prismatic VLM)
        vlm = load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)

    # [Validate] 确认模型仍保持全精度参数
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # 根据 vision / LLM 是否冻结选择训练阶段（兼容不同微调策略）
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-full-train"  # 两个主干都开训，最“重”的微调方式
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-train"  # 视觉 backbone 冻住，只训练语言部分
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"  # 典型“三明治”微调，视觉 + 投影 + LLM 最后一层
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # 只训练 LLM 最后一层，成本最低
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # 显式调用 freeze_backbones，日志里能看到哪些模块被冻结/解冻
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vlm.freeze_backbones(stage)

    # 打印模型总参数量与可训练参数量，便于监控
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    # 构建 VLA 数据集与 collator
    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    # 该 helper 会根据 data_mix 组装 RLDS 数据、返回动作 tokenizer 以及 collator
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vlm.vision_backbone.get_image_transform(),
        tokenizer=vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # 保存数据集统计，推理时需要用它来反归一化动作
    if overwatch.is_rank_zero():
        # 推理还原动作需要这些 mean/std，只有主进程保存一次即可
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # 创建训练策略对象
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    # get_train_strategy 会根据 FSDP/DDP 等不同策略封装优化器、调度器、梯度混合等细节
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    # 这里做真正的初始化，比如把模型包进 FSDP、构建优化器、估计训练样本数等
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

    # 创建指标记录器，用于向 jsonl / W&B 等后端推送训练指标
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    # VLAMetrics 负责把 loss/accuracy 等指标写到 jsonl 或 W&B，便于观察训练曲线
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    # 启动训练 loop
    overwatch.info("Starting VLA Training Loop")
    # run_vla_training 内部完成 DataLoader -> 前向 -> 反向 -> 保存 ckpt 的整套流程
    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        action_tokenizer,
        metrics,
        save_interval=cfg.save_interval,
    )

    # 训练完成后的收尾
    overwatch.info("Done with Training =>> Finalizing Metrics")
    # 关闭指标写入（例如 flush jsonl，结束 W&B run）
    metrics.finalize()

    # 所有流程执行完毕
    overwatch.info("... and that's all, folks!")
    # 全部 rank 在这里同步一下，确保没有任务仍在写文件
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
