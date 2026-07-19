#!/usr/bin/env bash
set -euo pipefail

# 这个脚本用于正式启动 suction 版 OpenVLA 的 LoRA 微调。

# 统一切回仓库根目录，避免相对路径失效。
cd /data1/ps/openvla

# 激活仓库使用的 conda 环境。
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate openvla

# 让训练进程优先使用 openvla 环境内随 PyTorch 打包的 cuDNN / cuBLAS，
# 避免系统 /usr/local/cuda-12.8/lib64 抢到前面导致版本混用。
CONDA_CUDNN_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib"
CONDA_CUBLAS_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib"
CONDA_LIB="$CONDA_PREFIX/lib"
OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
FILTERED_LD_LIBRARY_PATH="${OLD_LD_LIBRARY_PATH//\/usr\/local\/cuda-12.8\/lib64:}"
FILTERED_LD_LIBRARY_PATH="${FILTERED_LD_LIBRARY_PATH//\/usr\/local\/cuda-12.8\/lib64/}"
export LD_LIBRARY_PATH="$CONDA_CUDNN_LIB:$CONDA_CUBLAS_LIB:$CONDA_LIB${FILTERED_LD_LIBRARY_PATH:+:$FILTERED_LD_LIBRARY_PATH}"

# 固定使用当前空闲的四张 48G 4090。
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 限制 CPU / TensorFlow 线程数，降低 RLDS 数据管线长期运行时的内存和线程压力。
export OMP_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export TOKENIZERS_PARALLELISM=false

# 如果这台机器还没有做过 wandb 登录，请先手动执行：
#   conda activate openvla
#   wandb login
# ========================
# 训练参数集中放在这里，后面命令直接引用。
# ========================
VLA_PATH="/home/user/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/"
DATA_ROOT_DIR="./data_regeneration/rlds"
DATASET_NAME="processed_libero_suction_wrench_history_250"
RUN_ROOT_DIR="./runs_libero_suction_wrench_history_250_base_pretrained_ckpt_original_openvla"
LORA_RANK=32
BATCH_SIZE=8
GRAD_ACCUMULATION_STEPS=2
LEARNING_RATE="5e-4"
IMAGE_AUG="True"
SHUFFLE_BUFFER_SIZE=100000
SAVE_STEPS=10000
MAX_STEPS=110000
OUTPUT_DIR_NAME="${MAX_STEPS}"
# 临时 LoRA adapter 父目录也按这次实验单独区分，避免后续同超参重跑时复用旧目录。
ADAPTER_TMP_DIR="./adapter-tmp/suction_lora/${OUTPUT_DIR_NAME}"
WANDB_ENTITY="speng0442-huazhong-university-of-science-and-technology"

# 断点续训配置：
# - 置空表示从原生 OpenVLA 权重重新开始训练；
# - 如果要从某个 1w 步断点续训，把这里改成对应 step-* adapter checkpoint 目录；
# - 路径必须是包含 adapter_config.json、adapter_model.safetensors、resume_state/checkpoint.pt 的目录。
RESUME_CHECKPOINT=""

WANDB_PROJECT="$(basename "$RUN_ROOT_DIR")--${OUTPUT_DIR_NAME}"

echo "[run-name] model checkpoint = $VLA_PATH"
echo "[run-name] output dir name = $OUTPUT_DIR_NAME"
echo "[run-name] final merged checkpoint dir = ${RUN_ROOT_DIR}/${OUTPUT_DIR_NAME}"
echo "[run-name] adapter parent dir = ${ADAPTER_TMP_DIR}"
echo "[run-name] immutable adapter checkpoints dir = ${ADAPTER_TMP_DIR}/checkpoints/step-*"
echo "[run-name] latest adapter marker = ${ADAPTER_TMP_DIR}/.latest_adapter_dir"
echo "[run-name] wandb project = $WANDB_PROJECT"

# 启动前打印关键运行时信息，确认当前训练进程实际看到的是哪套 CUDA / cuDNN。
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

# 构建训练参数数组，便于条件性地追加断点续训参数
TORCHRUN_ARGS=(
  --vla_path "$VLA_PATH"
  --data_root_dir "$DATA_ROOT_DIR"
  --dataset_name "$DATASET_NAME"
  --run_root_dir "$RUN_ROOT_DIR"
  --run_dir_name_override "$OUTPUT_DIR_NAME"
  --adapter_tmp_dir "$ADAPTER_TMP_DIR"
  --lora_rank "$LORA_RANK"
  --batch_size "$BATCH_SIZE"
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --image_aug "$IMAGE_AUG"
  --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE"
  --save_steps "$SAVE_STEPS"
  --max_steps "$MAX_STEPS"
  --exit_after_save "True"
  --immutable_adapter_checkpoints "True"
  --wandb_project "$WANDB_PROJECT"
  --wandb_entity "$WANDB_ENTITY"
)

# 最新 adapter marker：finetune.py 只有在 adapter 权重和 resume_state 都写盘成功后才会更新这个文件。
LATEST_ADAPTER_MARKER="${ADAPTER_TMP_DIR}/.latest_adapter_dir"

# 读取某个 LoRA adapter checkpoint 已经完成的梯度步数。
# 这里不依赖目录名，因为 LoRA adapter 目录名是实验名，不是 step-10000 这种不可变步数目录。
read_checkpoint_step() {
  local checkpoint_dir="$1"

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

# 检查 LoRA adapter checkpoint 是否完整可恢复。
# 除了 adapter 权重本身，还要求 4 个 rank 的 RNG 状态都存在；这样可以避免半写入 checkpoint 被误判成功。
checkpoint_dir_is_valid() {
  local checkpoint_dir="$1"

  [ -n "$checkpoint_dir" ] || return 1
  [ -f "${checkpoint_dir}/adapter_config.json" ] || return 1
  [ -f "${checkpoint_dir}/adapter_model.safetensors" ] || return 1
  [ -f "${checkpoint_dir}/resume_state/checkpoint.pt" ] || return 1
  [ -f "${checkpoint_dir}/resume_state/rng_state_rank_0.pt" ] || return 1
  [ -f "${checkpoint_dir}/resume_state/rng_state_rank_1.pt" ] || return 1
  [ -f "${checkpoint_dir}/resume_state/rng_state_rank_2.pt" ] || return 1
  [ -f "${checkpoint_dir}/resume_state/rng_state_rank_3.pt" ] || return 1
  return 0
}

# 分段续训控制：
# - 不再把 torchrun 的外层退出码 20 当成唯一成功协议；
# - torchrun 可能把子进程的 SystemExit(20) 包装成外层 exit code 1；
# - 因此外层脚本只认“最新 adapter marker 指向完整 checkpoint，且步数严格等于上一段 + SAVE_STEPS”。
while true; do
  CURRENT_TORCHRUN_ARGS=("${TORCHRUN_ARGS[@]}")
  PREV_STEP=0

  # 只有在确实存在续训路径时，才把 resume 参数传给 finetune.py。
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
    echo "[resume] Starting a fresh segmented training run."
  fi

  # 启动正式 LoRA 微调。
  # 关键参数说明：
  # --run_dir_name_override 改最终 merged 权重目录名；adapter-tmp 这里也显式按本次实验分目录。
  # --save_steps=10000 且 --max_steps=110000，确保训练过程中会规律保存“最新 adapter + resume_state”。
  # --exit_after_save=True 时，训练会在每次成功保存后退出当前 torchrun 段；外层脚本通过 checkpoint 完整性续跑。
  # 训练真正完成时退出码为 0，并只在最后一段自动 merge 到 run_dir。
  # --nproc-per-node=4 与 CUDA_VISIBLE_DEVICES=2,3,4,5 对齐，真正使用 4 张卡训练。
  if torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py "${CURRENT_TORCHRUN_ARGS[@]}"; then
    EXIT_CODE=0
  else
    EXIT_CODE=$?
  fi

  if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[segment] Training finished normally. Final merge should already be available at ${RUN_ROOT_DIR}/${OUTPUT_DIR_NAME}."
    break
  fi

  if [ ! -f "$LATEST_ADAPTER_MARKER" ]; then
    echo "[error] Training exited with code $EXIT_CODE, and latest adapter marker is missing: $LATEST_ADAPTER_MARKER"
    exit "$EXIT_CODE"
  fi

  RESUME_CHECKPOINT="$(cat "$LATEST_ADAPTER_MARKER")"
  if ! checkpoint_dir_is_valid "$RESUME_CHECKPOINT"; then
    echo "[error] Training exited with code $EXIT_CODE, and resolved adapter checkpoint is incomplete: $RESUME_CHECKPOINT"
    exit "$EXIT_CODE"
  fi

  NEW_STEP="$(read_checkpoint_step "$RESUME_CHECKPOINT")"
  EXPECTED_STEP=$((PREV_STEP + SAVE_STEPS))
  if [ "$NEW_STEP" -eq "$EXPECTED_STEP" ]; then
    echo "[segment] torchrun exited with code $EXIT_CODE, but the expected complete LoRA adapter checkpoint was saved."
    echo "[segment] Previous step: $PREV_STEP"
    echo "[segment] Save interval: $SAVE_STEPS"
    echo "[segment] New step: $NEW_STEP"
    echo "[segment] Restarting from LoRA adapter checkpoint: $RESUME_CHECKPOINT"
    continue
  fi

  echo "[error] Training exited with code $EXIT_CODE, but the checkpoint step is not the expected next save boundary."
  echo "[error] Previous step: $PREV_STEP"
  echo "[error] Save interval: $SAVE_STEPS"
  echo "[error] Expected checkpoint step: $EXPECTED_STEP"
  echo "[error] Resolved checkpoint step: $NEW_STEP"
  exit "$EXIT_CODE"
done
