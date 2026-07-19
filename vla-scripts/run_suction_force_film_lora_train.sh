# 这个脚本用于启动 suction 数据集上的 OpenVLA LoRA + ForceFiLM 联合训练，输入是官方 openvla-7b checkpoint 和 processed_libero_suction RLDS 数据，输出是合并后的 VLA 权重、LoRA+ForceFiLM 不可变断点以及 force_film_generator.pt。
#!/usr/bin/env bash
set -euo pipefail

# 统一切回仓库根目录，避免从其他目录启动时相对路径失效。
# 仓库已从 /data1/ps/openvla 搬运到 /data/ps/openvla_force，这里同步改为新路径。
cd /data/ps/openvla_force

# 激活项目使用的 conda 环境；仓库规范要求所有相关命令都先进入 openvla 环境。
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate openvla

# 让训练优先使用 conda 环境内随 PyTorch 安装的 cuDNN / cuBLAS，
# 避免系统 CUDA 动态库抢到前面导致二进制版本混用。
CONDA_CUDNN_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib"
CONDA_CUBLAS_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib"
CONDA_LIB="$CONDA_PREFIX/lib"
OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
FILTERED_LD_LIBRARY_PATH="${OLD_LD_LIBRARY_PATH//\/usr\/local\/cuda-12.8\/lib64:}"
FILTERED_LD_LIBRARY_PATH="${FILTERED_LD_LIBRARY_PATH//\/usr\/local\/cuda-12.8\/lib64/}"
export LD_LIBRARY_PATH="$CONDA_CUDNN_LIB:$CONDA_CUBLAS_LIB:$CONDA_LIB${FILTERED_LD_LIBRARY_PATH:+:$FILTERED_LD_LIBRARY_PATH}"

# 默认使用 4 张 GPU；如果当前机器卡号占用情况变化，只改这一行即可。
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 限制 CPU / TensorFlow 线程数，降低 RLDS 数据管线长期运行时的内存和线程压力。
export OMP_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export TOKENIZERS_PARALLELISM=false

# ========================
# 训练参数集中放在这里，后面命令直接引用。
# ========================
# 使用官方 OpenVLA-7B 本地 snapshot；这个目录只读加载，不会被训练脚本覆盖。
VLA_PATH="/home/user/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/"
# suction 数据集所在的 RLDS 根目录。
# 新仓库中数据直接放在顶层 ./rlds，不再有 data_regeneration/ 这一层。
DATA_ROOT_DIR="./rlds"
# 本次联合训练使用每帧都带 [6,250] suction_wrench 历史窗口的 suction 任务数据。
DATASET_NAME="processed_libero_suction_wrench_history_250"
# 最终合并后完整模型的输出根目录。
# primus_trainable_rank256 表示 PRIMUS / IMU encoder backbone 会全参训练，
# output_projection 也会全参训练；postattn 表示 ForceFiLM 放在 attention 之后、第二个 norm 之前。
RUN_ROOT_DIR="./runs_libero_suction_wrench_history_250_primus_trainable_rank256_(post_attn)_film_lora"
# LoRA 低秩分解秩，沿用当前 suction LoRA 常用配置。
LORA_RANK=32
# 每张 GPU 的 batch size；联合训练额外多了 PRIMUS 力编码器和 FiLM 输出映射，但显存主要仍由 OpenVLA 决定。
BATCH_SIZE=8
# 梯度累积步数；这里保持 2，等效全局 batch 为 BATCH_SIZE * 4*2。
GRAD_ACCUMULATION_STEPS=2
# LoRA 主学习率。PRIMUS 和 output_projection 会在 finetune.py 里使用更小的独立学习率。
LEARNING_RATE="5e-4"
# PRIMUS / IMU encoder backbone 的独立学习率；它已经有预训练权重，所以只做小步微调。
FORCE_ENCODER_LEARNING_RATE="1e-5"
# output_projection 直接生成所有视觉层的 gamma/beta，因此学习率需要明显小于 LoRA 主学习率。
FORCE_PROJECTION_LEARNING_RATE="2e-5"
# 是否启用 RLDS 图像增强。
IMAGE_AUG="True"
# RLDS shuffle buffer，越大越随机，但 CPU 内存占用越高。
SHUFFLE_BUFFER_SIZE=100000
# 每 10000 个真实 optimizer step 保存一次断点。
SAVE_STEPS=10000
# 总训练步数；100000 能和 save_steps 整除，最终 merge 会使用最后一个完整 checkpoint。
MAX_STEPS=110000
# 最终 run 目录名；postattn 明确标记 ForceFiLM 调制 attention 后的中间视觉特征。
# 注意必须写成 ${MAX_STEPS} 才会被 bash 展开成 110000，否则目录名会变成字面值 {MAX_STEPS}。
OUTPUT_DIR_NAME="${MAX_STEPS}"
# LoRA+ForceFiLM 的中间断点目录；和原 LoRA-only / 冻结 PRIMUS 的 ForceFiLM 实验完全隔离。
ADAPTER_TMP_DIR="./adapter-tmp/suction_primus_trainable_rank256_(post_attn)_film_lora/${OUTPUT_DIR_NAME}"
# W&B 组织。
WANDB_ENTITY="speng0442-huazhong-university-of-science-and-technology"
# PRIMUS 预训练 IMU 编码器 checkpoint；训练初始化时从这里加载，随后 IMU encoder backbone 会参与微调。
# 输入是 [6,250] 力历史窗口，输出是 512 维 mmcl 特征。
FORCE_ENCODER_CHECKPOINT_PATH="/data1/ps/pretrained-imu-encoders/checkpoint/best_model.ckpt"
# 使用 PRIMUS 多头输出中的 mmcl 分支。
FORCE_ENCODER_EMBEDDING_TYPE="mmcl"
# suction_wrench 历史窗口长度，和数据集名保持一致。
FORCE_HISTORY_LENGTH=250
# 从头训练时保持空；如果要手动续训，填入 checkpoints/step-xxxxx 目录。
RESUME_CHECKPOINT=""

# W&B project 用输出根目录和 step 数区分，避免和 LoRA-only 实验混在一起。
WANDB_PROJECT="$(basename "$RUN_ROOT_DIR")--${OUTPUT_DIR_NAME}"

echo "[run-name] base checkpoint = $VLA_PATH"
echo "[run-name] dataset = ${DATA_ROOT_DIR}/${DATASET_NAME}"
echo "[run-name] final merged checkpoint dir = ${RUN_ROOT_DIR}/${OUTPUT_DIR_NAME}"
echo "[run-name] adapter checkpoint root = ${ADAPTER_TMP_DIR}/checkpoints"
echo "[run-name] wandb project = $WANDB_PROJECT"
echo "[run-name] lr = LoRA:${LEARNING_RATE}, PRIMUS:${FORCE_ENCODER_LEARNING_RATE}, projection:${FORCE_PROJECTION_LEARNING_RATE}"

# 启动前打印关键环境信息，方便确认 Python / PyTorch / CUDA / cuDNN 是否来自 openvla 环境。
python - <<'PYINFO'
import os
import sys
import torch
print('[env-check] python=', sys.executable)
print('[env-check] torch=', torch.__version__)
print('[env-check] torch_cuda=', torch.version.cuda)
print('[env-check] cudnn_version=', torch.backends.cudnn.version())
print('[env-check] LD_LIBRARY_PATH=', os.environ.get('LD_LIBRARY_PATH', ''))
PYINFO

# 构建传给 finetune.py 的基础参数；每一段续训只会在后面额外追加 resume_from_checkpoint。
TORCHRUN_ARGS=(
  --vla_path "$VLA_PATH"
  --data_root_dir "$DATA_ROOT_DIR"
  --dataset_name "$DATASET_NAME"
  --run_root_dir "$RUN_ROOT_DIR"
  --run_dir_name_override "$OUTPUT_DIR_NAME"
  --adapter_tmp_dir "$ADAPTER_TMP_DIR"
  --use_lora "True"
  --lora_rank "$LORA_RANK"
  --batch_size "$BATCH_SIZE"
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --image_aug "$IMAGE_AUG"
  --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE"
  --save_steps "$SAVE_STEPS"
  --max_steps "$MAX_STEPS"
  --exit_after_save "True"
  --use_force_film "True"
  --train_force_film "True"
  --force_encoder_checkpoint_path "$FORCE_ENCODER_CHECKPOINT_PATH"
  --force_encoder_embedding_type "$FORCE_ENCODER_EMBEDDING_TYPE"
  --force_history_length "$FORCE_HISTORY_LENGTH"
  --force_encoder_learning_rate "$FORCE_ENCODER_LEARNING_RATE"
  --force_projection_learning_rate "$FORCE_PROJECTION_LEARNING_RATE"
  --immutable_adapter_checkpoints "True"
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
)

# 最新 checkpoint marker：finetune.py 只有在 adapter、force_film_generator.pt、resume_state 全部写盘成功后才更新它。
LATEST_ADAPTER_MARKER="${ADAPTER_TMP_DIR}/.latest_adapter_dir"
# 不可变 step checkpoint 根目录。
CHECKPOINT_ROOT="${ADAPTER_TMP_DIR}/checkpoints"

# 从头训练时不允许复用已有 step-* 断点，否则外层续跑无法区分旧结果和本轮刚保存的结果。
if [ -z "$RESUME_CHECKPOINT" ]; then
  if compgen -G "${CHECKPOINT_ROOT}/step-*" > /dev/null; then
    echo "[error] Found existing immutable checkpoints under ${CHECKPOINT_ROOT}."
    echo "[error] Use a new OUTPUT_DIR_NAME, remove the old run manually, or set RESUME_CHECKPOINT explicitly."
    exit 1
  fi

  if [ -f "$LATEST_ADAPTER_MARKER" ]; then
    echo "[error] Found existing latest adapter marker: $LATEST_ADAPTER_MARKER"
    echo "[error] Use a new OUTPUT_DIR_NAME, remove the old run manually, or set RESUME_CHECKPOINT explicitly."
    exit 1
  fi
fi

# 读取 checkpoint 完成的训练步数。
# 对 step-* 目录优先从目录名解析；目录名不是 step-* 时才回退读取 resume_state。
read_checkpoint_step() {
  local checkpoint_dir="$1"
  local checkpoint_base
  checkpoint_base="$(basename "$checkpoint_dir")"

  if [[ "$checkpoint_base" == step-* ]]; then
    local step_str="${checkpoint_base#step-}"
    step_str="${step_str#0}"
    if [ -z "$step_str" ]; then
      step_str="0"
    fi
    echo "$step_str"
    return 0
  fi

  python - "$checkpoint_dir" <<'PYSTEP'
import sys
import torch
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
resume_ckpt_path = checkpoint_dir / "resume_state" / "checkpoint.pt"
resume_ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
print(int(resume_ckpt.get("completed_gradient_steps", 0)))
PYSTEP
}

# 检查 LoRA+ForceFiLM checkpoint 是否完整可恢复。
checkpoint_dir_is_valid() {
  local checkpoint_dir="$1"

  [ -n "$checkpoint_dir" ] || return 1
  [ -f "${checkpoint_dir}/adapter_config.json" ] || return 1
  [ -f "${checkpoint_dir}/adapter_model.safetensors" ] || return 1
  [ -f "${checkpoint_dir}/force_film_generator.pt" ] || return 1
  [ -f "${checkpoint_dir}/resume_state/checkpoint.pt" ] || return 1
  [ -f "${checkpoint_dir}/resume_state/rng_state_rank_0.pt" ] || return 1
  [ -f "${checkpoint_dir}/resume_state/rng_state_rank_1.pt" ] || return 1
  [ -f "${checkpoint_dir}/resume_state/rng_state_rank_2.pt" ] || return 1
  [ -f "${checkpoint_dir}/resume_state/rng_state_rank_3.pt" ] || return 1
  return 0
}

# 分段续训主循环：
# - torchrun 可能把训练脚本内部的 SystemExit(20) 包装成外层 exit code 1；
# - 所以这里不只看退出码，而是严格检查“新的完整 step checkpoint 是否正好推进 SAVE_STEPS”。
while true; do
  CURRENT_TORCHRUN_ARGS=("${TORCHRUN_ARGS[@]}")
  PREV_STEP=0

  if [ -n "$RESUME_CHECKPOINT" ]; then
    if ! checkpoint_dir_is_valid "$RESUME_CHECKPOINT"; then
      echo "[error] Resume checkpoint is incomplete: $RESUME_CHECKPOINT"
      exit 1
    fi

    CURRENT_TORCHRUN_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
    PREV_STEP="$(read_checkpoint_step "$RESUME_CHECKPOINT")"
    echo "[resume] Resuming from checkpoint: $RESUME_CHECKPOINT"
    echo "[resume] Previous completed step: $PREV_STEP"
  else
    echo "[resume] Starting a fresh segmented LoRA+ForceFiLM training run."
  fi

  if torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py "${CURRENT_TORCHRUN_ARGS[@]}"; then
    EXIT_CODE=0
  else
    EXIT_CODE=$?
  fi

  if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[segment] LoRA+ForceFiLM training finished normally."
    echo "[segment] Final merged checkpoint should be available at ${RUN_ROOT_DIR}/${OUTPUT_DIR_NAME}."
    break
  fi

  if [ ! -f "$LATEST_ADAPTER_MARKER" ]; then
    echo "[error] Training exited with code $EXIT_CODE, and latest adapter marker is missing: $LATEST_ADAPTER_MARKER"
    exit "$EXIT_CODE"
  fi

  RESUME_CHECKPOINT="$(cat "$LATEST_ADAPTER_MARKER")"
  if ! checkpoint_dir_is_valid "$RESUME_CHECKPOINT"; then
    echo "[error] Training exited with code $EXIT_CODE, and resolved checkpoint is incomplete: $RESUME_CHECKPOINT"
    exit "$EXIT_CODE"
  fi

  NEW_STEP="$(read_checkpoint_step "$RESUME_CHECKPOINT")"
  EXPECTED_STEP=$((PREV_STEP + SAVE_STEPS))
  if [ "$NEW_STEP" -eq "$EXPECTED_STEP" ]; then
    echo "[segment] torchrun exited with code $EXIT_CODE, but the expected immutable checkpoint was saved."
    echo "[segment] Previous step: $PREV_STEP"
    echo "[segment] Save interval: $SAVE_STEPS"
    echo "[segment] New step: $NEW_STEP"
    echo "[segment] Restarting from LoRA+ForceFiLM checkpoint: $RESUME_CHECKPOINT"
    continue
  fi

  echo "[error] Training exited with code $EXIT_CODE, but the checkpoint step is not the expected next save boundary."
  echo "[error] Previous step: $PREV_STEP"
  echo "[error] Save interval: $SAVE_STEPS"
  echo "[error] Expected checkpoint step: $EXPECTED_STEP"
  echo "[error] Resolved checkpoint step: $NEW_STEP"
  exit "$EXIT_CODE"
done
