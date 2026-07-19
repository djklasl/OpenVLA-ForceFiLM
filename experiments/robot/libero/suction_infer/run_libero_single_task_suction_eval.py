"""
run_libero_single_task_suction_eval.py

这个脚本是 suction 并发评测的 worker 执行器。

当前用途：
- 只接收 parallel launcher 显式传入的参数
- 作为单个 worker，只评测分配给自己的 task 子集
- 在显式开启视频保存时，保存三图拼接监看视频，并按“成功/失败 + 是否出现吸附失败”分目录存放
- 输出本 worker 的本地日志和 summary.json，供 launcher 汇总
"""

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

ROOT_DIR = Path(__file__).resolve().parents[4]
LIBERO_DIR = ROOT_DIR / "LIBERO"

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(LIBERO_DIR))

import numpy as np
import tqdm
import wandb
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from PIL import Image, ImageDraw, ImageFont

from experiments.robot.libero.libero_utils import (
    _build_rollout_video_filename,
    create_combined_frame,
    create_suction_wrench_history,
    extract_suction_wrench,
    get_libero_dummy_action,
    get_libero_image,
    finalize_rollout_video,
    open_rollout_video_writer,
    save_rollout_video,
    update_suction_wrench_history,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


ACTION_WRENCH_HISTORY_CSV_COLUMNS = [
    "force_x",
    "force_y",
    "force_z",
    "torque_x",
    "torque_y",
    "torque_z",
    "action_delta_x",
    "action_delta_y",
    "action_delta_z",
    "action_delta_roll",
    "action_delta_pitch",
    "action_delta_yaw",
    "action_suction",
]

SUCTION_FAILURE_OVERLAY_TEXT = "吸附失败"
SUCTION_REWIND_ACTION_COUNT_OVERLAY_PREFIX = "回溯动作剩余"
CONSTANT_FORCE_FILM_INPUT_VALUE = 20.0
# 吸附失败回溯结束后，用这个固定方向表给末端追加一个小扰动。
# 表里只写 action 的前三维，对应 x/y/z 方向；旋转维度和 suction 维度在构造 action 时单独处理。
SUCTION_ESCAPE_PERTURB_DIRECTIONS = [
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
]
SUCTION_FAILURE_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

ROLLOUT_VIDEO_CATEGORY_SUCCESS_NO_SUCTION_FAILURE = "success_no_suction_failure"
ROLLOUT_VIDEO_CATEGORY_SUCCESS_WITH_SUCTION_FAILURE = "success_with_suction_failure"
ROLLOUT_VIDEO_CATEGORY_FAILURE_NO_SUCTION_FAILURE = "failure_no_suction_failure"
ROLLOUT_VIDEO_CATEGORY_FAILURE_WITH_SUCTION_FAILURE = "failure_with_suction_failure"


def build_constant_force_film_input(reference_suction_wrench: np.ndarray) -> np.ndarray:
    """根据真实力历史的形状，构造一份全 20.0 的 Force-FiLM 输入。"""
    reference_suction_wrench = np.asarray(reference_suction_wrench)
    # 这里只按真实历史窗口的 shape 构造模型输入，不改真实 suction_wrench_history 本身。
    # 因此 CSV、视频显示、吸附失败检测、回溯和环境状态仍然使用真实六维力。
    # 是否真正把这份常数输入传给模型，由每条 episode 的“是否已经第一次接触”状态决定。
    return np.full(reference_suction_wrench.shape, CONSTANT_FORCE_FILM_INPUT_VALUE, dtype=np.float32)


def get_rollout_video_category(success: bool, has_suction_failure: bool) -> str:
    """根据最终成功状态和 episode 内是否出现过吸附失败，返回四分类视频目录名。"""
    if success and not has_suction_failure:
        # “成功且无吸附失败”：最终任务成功，并且整条轨迹里没有触发过 suction_failed。
        return ROLLOUT_VIDEO_CATEGORY_SUCCESS_NO_SUCTION_FAILURE
    if success and has_suction_failure:
        # “成功且有吸附失败”：最终任务成功，但过程中至少一次触发 suction_failed。
        return ROLLOUT_VIDEO_CATEGORY_SUCCESS_WITH_SUCTION_FAILURE
    if not success and not has_suction_failure:
        # “失败且无吸附失败”：最终任务失败，并且没有检测到吸附失败状态。
        return ROLLOUT_VIDEO_CATEGORY_FAILURE_NO_SUCTION_FAILURE
    # “失败且有吸附失败”：最终任务失败，且过程中至少一次触发 suction_failed。
    return ROLLOUT_VIDEO_CATEGORY_FAILURE_WITH_SUCTION_FAILURE


def create_action_history(history_length: int = 250) -> np.ndarray:
    """创建左侧补零的 action 历史窗口，形状是 [7, history_length]。"""
    return np.zeros((7, int(history_length)), dtype=np.float32)


def update_action_history(history: np.ndarray, action: Any) -> np.ndarray:
    """把当前控制时刻要执行的 7 维 action 追加到历史窗口最后一列。"""
    history = np.asarray(history, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if history.ndim != 2 or history.shape[0] != 7:
        raise ValueError(f"`history` must have shape [7, T], got {history.shape}")
    if action.shape[0] != 7:
        raise ValueError(f"`action` must have 7 values, got {action.shape}")
    history = np.roll(history, shift=-1, axis=1)
    history[:, -1] = action
    return history


def apply_contact_force_suction_override(
    *,
    action: Any,
    suction_wrench: Any,
    enabled: bool,
) -> tuple[np.ndarray, bool]:
    """当当前 Fz 大于 0 时，按开关强制把 VLA action 的吸盘维度置为 1。"""
    # 这里复制一份 action，避免在原数组上原地修改，导致调用方后续排查时分不清“模型原始输出”和“最终执行动作”。
    action_array = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    # suction_wrench 是 6 维力/力矩：[Fx, Fy, Fz, Tx, Ty, Tz]，这里同样转成一维数组方便取第 3 维 Fz。
    suction_wrench_array = np.asarray(suction_wrench, dtype=np.float32).reshape(-1)
    if action_array.shape[0] != 7:
        raise ValueError(f"`action` must have 7 values, got {action_array.shape}")
    if suction_wrench_array.shape[0] != 6:
        raise ValueError(f"`suction_wrench` must have 6 values, got {suction_wrench_array.shape}")

    # 只看 Fz，而不是看六维任意分量；这和吸附失败检测里“正向接触”的定义保持一致。
    current_fz = float(suction_wrench_array[2])
    should_override = bool(enabled) and current_fz > 0.0
    if should_override:
        # action 第 7 维是 suction 控制值。Fz>0 表示吸盘正在挤压/接触物体，此时强制开启吸盘。
        action_array[6] = 1.0
    return action_array


def save_action_wrench_history_csv(
    *,
    csv_path: Union[str, Path],
    suction_wrench_history: np.ndarray,
    action_history: np.ndarray,
    log_file=None,
) -> None:
    """把最终 250 长度的 wrench/action 同步窗口保存为 CSV。"""
    suction_wrench_history = np.asarray(suction_wrench_history, dtype=np.float32)
    action_history = np.asarray(action_history, dtype=np.float32)
    if suction_wrench_history.ndim != 2 or suction_wrench_history.shape[0] != 6:
        raise ValueError(f"`suction_wrench_history` must have shape [6, T], got {suction_wrench_history.shape}")
    if action_history.ndim != 2 or action_history.shape[0] != 7:
        raise ValueError(f"`action_history` must have shape [7, T], got {action_history.shape}")
    if suction_wrench_history.shape[1] != action_history.shape[1]:
        raise ValueError(
            "`suction_wrench_history` and `action_history` must have the same history length, "
            f"got {suction_wrench_history.shape[1]} and {action_history.shape[1]}"
        )

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # 两个窗口内部都是 [通道数, 时间]，这里转置成 [时间, 通道数]，保证 CSV 每一行是同一个控制时刻。
    csv_rows = np.concatenate([suction_wrench_history.T, action_history.T], axis=1)
    np.savetxt(
        csv_path,
        csv_rows,
        delimiter=",",
        header=",".join(ACTION_WRENCH_HISTORY_CSV_COLUMNS),
        comments="",
        fmt="%.8f",
    )
    print(f"Saved action/wrench history CSV at path {csv_path}")
    if log_file is not None:
        log_file.write(f"Saved action/wrench history CSV at path {csv_path}\n")


def load_suction_failure_overlay_font(font_size: int) -> ImageFont.ImageFont:
    """加载能显示中文的字体；如果系统字体缺失，就退回 PIL 默认字体。"""
    for font_path in SUCTION_FAILURE_FONT_CANDIDATES:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def draw_suction_failure_overlay(frame: np.ndarray) -> np.ndarray:
    """在视频帧上用红色中文标记“吸附失败”，输出尺寸和输入帧保持一致。"""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # create_combined_frame 输出的是左侧力图、中间主视角、右侧手眼图三列拼接。
    # 这里优先把提示放到中间主视角左上角，既醒目，也不会盖住左侧六维力数值区域。
    overlay_frame = frame.copy()
    image = Image.fromarray(overlay_frame)
    draw = ImageDraw.Draw(image)
    font_size = max(24, min(48, int(image.height * 0.14)))
    font = load_suction_failure_overlay_font(font_size)
    x = image.height + 12 if image.width >= image.height * 3 else 12
    y = 12
    draw.text(
        (x, y),
        SUCTION_FAILURE_OVERLAY_TEXT,
        fill=(255, 0, 0),
        font=font,
        stroke_width=2,
        stroke_fill=(255, 255, 255),
    )
    return np.asarray(image)


def draw_suction_rewind_action_count_overlay(frame: np.ndarray, rewind_action_count: int) -> np.ndarray:
    """在视频帧上显示当前还需要执行多少步回溯 action；计数为 0 时直接返回原图。"""
    if rewind_action_count <= 0:
        return frame

    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # 这里的“剩余”包含当前控制时刻马上要执行的这一条回溯 action。
    # 因此单步回溯也会显示 1，避免视频里最后一步回溯看起来像 VLA 正常动作。
    overlay_frame = frame.copy()
    image = Image.fromarray(overlay_frame)
    draw = ImageDraw.Draw(image)
    font_size = max(20, min(40, int(image.height * 0.10)))
    font = load_suction_failure_overlay_font(font_size)
    x = image.height + 12 if image.width >= image.height * 3 else 12
    # 放在“吸附失败”提示的下方区域；即使当前帧没有吸附失败红字，也固定在这里，便于跨帧比较。
    y = 12 + max(32, int(image.height * 0.15))
    draw.text(
        (x, y),
        f"{SUCTION_REWIND_ACTION_COUNT_OVERLAY_PREFIX}: {rewind_action_count}",
        fill=(255, 180, 0),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    return np.asarray(image)


def detect_suction_failure_from_history(
    *,
    suction_wrench_history: np.ndarray,
    action_history: np.ndarray,
    contact_fz_threshold: float,
    free_fz_threshold: float,
    lift_action_z_threshold: float,
    contact_history_steps: int,
    min_contact_steps: int,
) -> tuple[bool, dict[str, float | int | bool]]:
    """根据同步的 wrench/action 历史窗口判断当前控制时刻是否是吸附失败。"""
    suction_wrench_history = np.asarray(suction_wrench_history, dtype=np.float32)
    action_history = np.asarray(action_history, dtype=np.float32)
    if suction_wrench_history.ndim != 2 or suction_wrench_history.shape[0] != 6:
        raise ValueError(f"`suction_wrench_history` must have shape [6, T], got {suction_wrench_history.shape}")
    if action_history.ndim != 2 or action_history.shape[0] != 7:
        raise ValueError(f"`action_history` must have shape [7, T], got {action_history.shape}")
    if suction_wrench_history.shape[1] != action_history.shape[1]:
        raise ValueError(
            "`suction_wrench_history` and `action_history` must have the same history length, "
            f"got {suction_wrench_history.shape[1]} and {action_history.shape[1]}"
        )
    if contact_history_steps <= 0:
        raise ValueError(f"`contact_history_steps` must be positive, got {contact_history_steps}")
    if min_contact_steps <= 0:
        raise ValueError(f"`min_contact_steps` must be positive, got {min_contact_steps}")

    # suction_wrench 的第 3 维是 Fz。只看最近一段历史，防止很早之前的接触影响当前判断。
    history_steps = min(int(contact_history_steps), suction_wrench_history.shape[1])
    fz_history = suction_wrench_history[2]
    recent_fz = fz_history[-history_steps:]
    contact_count = int(np.count_nonzero(recent_fz > float(contact_fz_threshold)))

    # 当前列代表本控制时刻：当前读到的 Fz，以及马上要执行的 action_z。
    current_fz = float(fz_history[-1])
    current_action_z = float(action_history[2, -1])
    has_recent_contact = contact_count >= int(min_contact_steps)
    is_currently_free = abs(current_fz) <= float(free_fz_threshold)
    is_lifting = current_action_z > float(lift_action_z_threshold)
    suction_failed = has_recent_contact and is_currently_free and is_lifting

    debug_info = {
        "current_fz": current_fz,
        "current_action_z": current_action_z,
        "contact_count": contact_count,
        "contact_history_steps": history_steps,
        "has_recent_contact": has_recent_contact,
        "is_currently_free": is_currently_free,
        "is_lifting": is_lifting,
    }
    return suction_failed, debug_info


def build_suction_failure_rewind_action_sequence(
    *,
    suction_wrench_history: np.ndarray,
    action_history: np.ndarray,
    contact_fz_threshold: float,
    pre_contact_steps: int,
) -> list[np.ndarray]:
    """根据最近一段接触历史，构造吸附失败后的多步回溯 action 序列。"""
    suction_wrench_history = np.asarray(suction_wrench_history, dtype=np.float32)
    action_history = np.asarray(action_history, dtype=np.float32)
    if suction_wrench_history.ndim != 2 or suction_wrench_history.shape[0] != 6:
        raise ValueError(f"`suction_wrench_history` must have shape [6, T], got {suction_wrench_history.shape}")
    if action_history.ndim != 2 or action_history.shape[0] != 7:
        raise ValueError(f"`action_history` must have shape [7, T], got {action_history.shape}")
    if suction_wrench_history.shape[1] != action_history.shape[1]:
        raise ValueError(
            "`suction_wrench_history` and `action_history` must have the same history length, "
            f"got {suction_wrench_history.shape[1]} and {action_history.shape[1]}"
        )
    if pre_contact_steps < 0:
        raise ValueError(f"`pre_contact_steps` must be non-negative, got {pre_contact_steps}")

    fz_history = suction_wrench_history[2]
    failure_idx = int(suction_wrench_history.shape[1] - 1)
    # 先找到距离当前失败点最近的一段正向接触区间末尾。
    contact_end_idx = None
    for idx in range(failure_idx, -1, -1):
        if fz_history[idx] > float(contact_fz_threshold):
            contact_end_idx = idx
            break
    if contact_end_idx is None:
        # 理论上只有检测到“近期接触后失败”才会调用这里。
        # 这里保留保护分支，避免阈值配置变化或历史窗口异常时把 None 强行转成 int。
        return []

    # 再向前找到这段连续接触区间的起点，避免多次失败时回到更早的旧接触段。
    contact_start_idx = int(contact_end_idx)
    while contact_start_idx > 0 and fz_history[contact_start_idx - 1] > float(contact_fz_threshold):
        contact_start_idx -= 1

    rewind_idx = max(0, contact_start_idx - int(pre_contact_steps))
    sum_start_idx = int(rewind_idx)
    # 当前检测发生在 env.step(action) 前，所以最后一列“尚未执行”的 action 固定不纳入求和。
    sum_end_idx = failure_idx
    if sum_end_idx <= sum_start_idx:
        # 没有任何已经执行过的历史 action 可以用来求和，说明回溯区间为空。
        return []

    action_sum = np.sum(action_history[:6, sum_start_idx:sum_end_idx], axis=1)
    total_rewind_delta = -action_sum.astype(np.float32)
    if not np.any(np.abs(total_rewind_delta) > 1e-6):
        # 历史 action 总和几乎为 0 时，没有必要插入额外回溯动作。
        return []

    rewind_actions: list[np.ndarray] = []
    remaining_delta = total_rewind_delta.copy()
    suction_value = float(action_history[6, rewind_idx])
    # robosuite 的 OSC_POSE 控制器会把前 6 维输入裁剪到 [-1, 1]。
    # 因此每一步都尽量取当前剩余量里可执行的最大值，步数就是 ceil(max(abs(total_rewind_delta)))。
    while np.any(np.abs(remaining_delta) > 1e-6):
        step_delta = np.clip(remaining_delta, -1.0, 1.0).astype(np.float32)
        rewind_action = np.zeros(7, dtype=np.float32)
        rewind_action[:6] = step_delta
        # 第 7 维是 suction/gripper 控制值，不做求和拆分，直接恢复为回溯点那一刻的值。
        rewind_action[6] = suction_value
        rewind_actions.append(rewind_action)
        remaining_delta = (remaining_delta - step_delta).astype(np.float32)

    return rewind_actions


def build_suction_failure_escape_perturb_action(
    *,
    suction_failure_count: int,
    perturb_scale_step: float,
) -> np.ndarray:
    """根据当前 episode 内的吸附失败次数，构造一个确定性的三维脱困扰动 action。"""
    if suction_failure_count <= 0:
        raise ValueError(f"`suction_failure_count` must be positive, got {suction_failure_count}")
    if perturb_scale_step < 0:
        raise ValueError(f"`perturb_scale_step` must be non-negative, got {perturb_scale_step}")

    # 第 1 次失败使用方向表第 0 个方向，第 2 次失败使用第 1 个方向，超过表长后循环复用。
    direction_idx = int(suction_failure_count - 1) % len(SUCTION_ESCAPE_PERTURB_DIRECTIONS)
    direction = np.asarray(SUCTION_ESCAPE_PERTURB_DIRECTIONS[direction_idx], dtype=np.float32)
    # 幅值直接等于“失败次数 * 步长”，这样失败越多，脱困扰动越明显，逻辑也方便从视频里核对。
    scale = float(suction_failure_count) * float(perturb_scale_step)

    perturb_action = np.zeros(7, dtype=np.float32)
    # 这里只改 x/y 两个平移维度；roll/pitch/yaw 保持 0，避免额外引入旋转扰动。
    perturb_action[:2] = direction * scale
    # 第 7 维继续沿用当前回溯队列里的 suction/gripper 值，不在扰动阶段额外强行开关吸盘。
    perturb_action[2] = -1.0
    perturb_action[6] = -1.0
    return perturb_action


def _mujoco_joint_sizes(joint_type: int) -> tuple[int, int]:
    if joint_type == 0:
        return 7, 6
    if joint_type == 1:
        return 4, 3
    if joint_type in (2, 3):
        return 1, 1
    raise ValueError(f"Unknown MuJoCo joint type: {joint_type}")


def _build_joint_index(model) -> dict[str, tuple[int, int, int]]:
    mapping: dict[str, tuple[int, int, int]] = {}
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name
        mapping[name] = (
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
            int(model.jnt_type[joint_id]),
        )
    return mapping


def convert_default_init_state_to_suction(*, init_state_default, env_default: ControlEnv, env_suction: ControlEnv) -> np.ndarray:
    if hasattr(init_state_default, "detach"):
        init_state_default = init_state_default.detach().cpu().numpy()
    state_d = np.asarray(init_state_default, dtype=np.float64).reshape(-1)

    model_d = env_default.sim.model
    model_s = env_suction.sim.model

    expected_len_d = 1 + int(model_d.nq) + int(model_d.nv)
    expected_len_s = 1 + int(model_s.nq) + int(model_s.nv)
    if len(state_d) != expected_len_d:
        raise ValueError(f"Default init_state length mismatch: got {len(state_d)}, expected {expected_len_d}")

    time = float(state_d[0])
    qpos_d = state_d[1 : 1 + int(model_d.nq)]
    qvel_d = state_d[1 + int(model_d.nq) :]

    qpos_s = np.asarray(env_suction.sim.data.qpos, dtype=np.float64).copy()
    qvel_s = np.asarray(env_suction.sim.data.qvel, dtype=np.float64).copy()

    if len(qpos_s) != int(model_s.nq) or len(qvel_s) != int(model_s.nv):
        raise RuntimeError("Unexpected MuJoCo qpos/qvel sizes in suction env")

    joints_d = _build_joint_index(model_d)

    for joint_id in range(model_s.njnt):
        joint_name = model_s.joint(joint_id).name
        qposadr_s = int(model_s.jnt_qposadr[joint_id])
        dofadr_s = int(model_s.jnt_dofadr[joint_id])
        joint_type_s = int(model_s.jnt_type[joint_id])
        qpos_sz_s, qvel_sz_s = _mujoco_joint_sizes(joint_type_s)

        if joint_name == "gripper0_suction_joint":
            qpos_s[qposadr_s : qposadr_s + qpos_sz_s] = 0.0
            qvel_s[dofadr_s : dofadr_s + qvel_sz_s] = 0.0
            continue

        if joint_name not in joints_d:
            continue

        qposadr_d, dofadr_d, joint_type_d = joints_d[joint_name]
        qpos_sz_d, qvel_sz_d = _mujoco_joint_sizes(joint_type_d)
        if (qpos_sz_d, qvel_sz_d) != (qpos_sz_s, qvel_sz_s):
            raise ValueError(
                f"Joint size mismatch for {joint_name}: default {(qpos_sz_d, qvel_sz_d)} vs suction {(qpos_sz_s, qvel_sz_s)}"
            )

        qpos_s[qposadr_s : qposadr_s + qpos_sz_s] = qpos_d[qposadr_d : qposadr_d + qpos_sz_d]
        qvel_s[dofadr_s : dofadr_s + qvel_sz_s] = qvel_d[dofadr_d : dofadr_d + qvel_sz_d]

    state_s = np.concatenate(([time], qpos_s, qvel_s), axis=0)
    if len(state_s) != expected_len_s:
        raise RuntimeError(f"Converted suction state length mismatch: got {len(state_s)}, expected {expected_len_s}")
    return state_s


@dataclass
class GenerateConfig:
    model_family: str
    pretrained_checkpoint: Union[str, Path]
    load_in_8bit: bool
    load_in_4bit: bool
    center_crop: bool
    task_suite_name: str
    gripper_type: str
    unnorm_key: str
    num_steps_wait: int
    max_steps: int
    num_trials: int
    resolution: int
    control_freq: int
    # 是否保存所有 rollout 视频：
    # - False：完全不保存任何视频
    # - True：保存所有 rollout 视频（成功和失败都会保存）
    save_all_rollout_videos: bool
    # 是否在保存 rollout 视频时，同步保存同名 CSV。
    # CSV 的每一行包含同一个控制时刻的 suction wrench 与即将执行的 action。
    save_action_wrench_history_csv: bool
    # 保存后的视频播放帧率。只改变视频播放速度，不改变环境控制频率。
    rollout_video_fps: int = 30
    # 是否启用吸附失败检测。检测结果可用于视频红字提示，也可触发回溯动作。
    enable_suction_failure_detection: bool = False
    # Fz 大于这个值时，认为吸盘和物体发生过正向接触/挤压。
    suction_failure_contact_fz_threshold: float = 0.0
    # 当前 Fz 的绝对值小于等于这个值时，认为当前没有接触、也没有吸附负载。
    suction_failure_free_fz_threshold: float = 1e-6
    # 当前 action_z 大于这个值时，认为吸盘正在向上离开。
    suction_failure_lift_action_z_threshold: float = 0.0
    # 检测正向接触时只回看最近多少个控制时刻。
    suction_failure_contact_history_steps: int = 50
    # 最近历史里至少多少个 Fz 正向接触点，才认为真的发生过吸附尝试。
    suction_failure_min_contact_steps: int = 3
    # 是否把检测到的吸附失败直接用红色中文叠加到保存视频里。
    show_suction_failure_overlay_in_video: bool = False
    # 是否在 VLA action 分支里启用“Fz>0 时强制 action_suction=1”。
    # 这个规则只覆盖模型刚输出的动作，不覆盖等待阶段 dummy action，也不覆盖回溯队列里的 action。
    enable_contact_force_suction_override: bool = False
    # 是否在吸附失败后用历史 action 求和取反生成多步回溯 action。
    enable_suction_failure_rewind: bool = False
    # 找到最近接触段起点后，再往前回退多少个控制时刻作为回溯点。
    suction_rewind_pre_contact_steps: int = 5
    # 是否在回溯动作结束后追加一个由失败次数决定的 x/y/z 确定性扰动 action。
    enable_suction_failure_escape_perturb: bool = False
    # 每发生一次吸附失败，扰动幅值增加多少；实际幅值 = 当前 episode 内失败次数 * 这个步长。
    suction_escape_perturb_scale_step: float = 0.01
    # `use_force_film`：是否启用基于 suction_wrench 的动态 FiLM。
    use_force_film: bool = False
    # `force_film_checkpoint`：只包含 ForceFiLMGenerator 的轻量 checkpoint 目录或 pt 文件。
    force_film_checkpoint: Optional[str] = None
    # 评测端只从 force_film_generator.pt 完整恢复 ForceFiLMGenerator，不再单独读取 PRIMUS best_model.ckpt。
    force_encoder_checkpoint_path: Optional[str] = None
    force_encoder_embedding_type: str = "mmcl"
    force_history_length: int = 250
    # 是否只在第一次接触物体之前，把“输入模型的六维力历史”强制替换成固定常数 20.0。
    # 第一次检测到 Fz != 0.0 后，本条 episode 会立刻恢复真实 suction_wrench_history 输入。
    # 这里不修改真实 suction_wrench_history 本身，所以 CSV、吸附失败检测、视频显示和环境状态仍然使用真实力数据。
    constant_force_film_input_before_first_contact: bool = False
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    local_video_dir: str = "./infer"
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    seed: int = 7
    worker_id: Optional[int] = None
    task_orders: Optional[list[int]] = None
    shared_run_id: Optional[str] = None
    summary_json_path: Optional[str] = None
    worker_log_path: Optional[str] = None
    progress_json_path: Optional[str] = None
    resume_state_json_path: Optional[str] = None
    worker_video_subdir: Optional[str] = None
    worker_use_wandb: bool = False


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_task_orders(task_orders: Optional[str]) -> Optional[list[int]]:
    if task_orders is None:
        return None
    values = [item.strip() for item in task_orders.split(",") if item.strip()]
    return [int(item) for item in values]


def build_config_from_cli() -> GenerateConfig:
    parser = argparse.ArgumentParser(description="Run one suction LIBERO eval worker with explicit launcher-provided arguments.")
    parser.add_argument("--model-family", type=str, required=True)
    parser.add_argument("--pretrained-checkpoint", type=str, required=True)
    parser.add_argument("--load-in-8bit", type=parse_bool, required=True)
    parser.add_argument("--load-in-4bit", type=parse_bool, required=True)
    parser.add_argument("--center-crop", type=parse_bool, required=True)
    parser.add_argument("--task-suite-name", type=str, required=True)
    parser.add_argument("--gripper-type", type=str, required=True)
    parser.add_argument("--unnorm-key", type=str, required=True)
    parser.add_argument("--num-steps-wait", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--control-freq", type=int, required=True)
    # 视频保存开关：
    # - False：一个视频都不落盘
    # - True：所有 rollout 视频都落盘
    parser.add_argument("--save-all-rollout-videos", type=parse_bool, required=True)
    parser.add_argument("--rollout-video-fps", type=int, default=30)
    parser.add_argument("--save-action-wrench-history-csv", type=parse_bool, required=True)
    parser.add_argument("--enable-suction-failure-detection", type=parse_bool, default=False)
    parser.add_argument("--suction-failure-contact-fz-threshold", type=float, default=0.0)
    parser.add_argument("--suction-failure-free-fz-threshold", type=float, default=1e-6)
    parser.add_argument("--suction-failure-lift-action-z-threshold", type=float, default=0.0)
    parser.add_argument("--suction-failure-contact-history-steps", type=int, default=50)
    parser.add_argument("--suction-failure-min-contact-steps", type=int, default=3)
    parser.add_argument("--show-suction-failure-overlay-in-video", type=parse_bool, default=False)
    parser.add_argument("--enable-contact-force-suction-override", type=parse_bool, default=False)
    parser.add_argument("--enable-suction-failure-rewind", type=parse_bool, default=False)
    parser.add_argument("--suction-rewind-pre-contact-steps", type=int, default=5)
    parser.add_argument("--enable-suction-failure-escape-perturb", type=parse_bool, default=False)
    parser.add_argument("--suction-escape-perturb-scale-step", type=float, default=0.01)
    parser.add_argument("--use-force-film", type=parse_bool, default=False)
    parser.add_argument("--force-film-checkpoint", type=str)
    parser.add_argument("--force-encoder-checkpoint-path", type=str, default=None)
    parser.add_argument("--force-encoder-embedding-type", type=str, default="mmcl")
    parser.add_argument("--force-history-length", type=int, default=250)
    parser.add_argument("--constant-force-film-input-before-first-contact", type=parse_bool, default=False)
    parser.add_argument("--run-id-note", type=str)
    parser.add_argument("--local-log-dir", type=str, required=True)
    parser.add_argument("--local-video-dir", type=str, required=True)
    parser.add_argument("--use-wandb", type=parse_bool, required=True)
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--task-orders", type=str, required=True)
    parser.add_argument("--shared-run-id", type=str, required=True)
    parser.add_argument("--summary-json-path", type=str, required=True)
    parser.add_argument("--worker-log-path", type=str, required=True)
    parser.add_argument("--progress-json-path", type=str)
    parser.add_argument("--resume-state-json-path", type=str)
    parser.add_argument("--worker-video-subdir", type=str, required=True)
    parser.add_argument("--worker-use-wandb", type=parse_bool, required=True)
    args = parser.parse_args()

    if (args.use_wandb or args.worker_use_wandb) and (not args.wandb_project or not args.wandb_entity):
        raise ValueError("wandb_project and wandb_entity must be provided when W&B logging is enabled")
    if args.use_force_film and args.force_film_checkpoint is None:
        raise ValueError("`--force-film-checkpoint` must be provided when `--use-force-film true`.")

    return GenerateConfig(
        model_family=args.model_family,
        pretrained_checkpoint=args.pretrained_checkpoint,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        center_crop=args.center_crop,
        task_suite_name=args.task_suite_name,
        gripper_type=args.gripper_type,
        unnorm_key=args.unnorm_key,
        num_steps_wait=args.num_steps_wait,
        max_steps=args.max_steps,
        num_trials=args.num_trials,
        resolution=args.resolution,
        control_freq=args.control_freq,
        save_all_rollout_videos=args.save_all_rollout_videos,
        rollout_video_fps=args.rollout_video_fps,
        save_action_wrench_history_csv=args.save_action_wrench_history_csv,
        enable_suction_failure_detection=args.enable_suction_failure_detection,
        suction_failure_contact_fz_threshold=args.suction_failure_contact_fz_threshold,
        suction_failure_free_fz_threshold=args.suction_failure_free_fz_threshold,
        suction_failure_lift_action_z_threshold=args.suction_failure_lift_action_z_threshold,
        suction_failure_contact_history_steps=args.suction_failure_contact_history_steps,
        suction_failure_min_contact_steps=args.suction_failure_min_contact_steps,
        show_suction_failure_overlay_in_video=args.show_suction_failure_overlay_in_video,
        enable_contact_force_suction_override=args.enable_contact_force_suction_override,
        enable_suction_failure_rewind=args.enable_suction_failure_rewind,
        suction_rewind_pre_contact_steps=args.suction_rewind_pre_contact_steps,
        enable_suction_failure_escape_perturb=args.enable_suction_failure_escape_perturb,
        suction_escape_perturb_scale_step=args.suction_escape_perturb_scale_step,
        use_force_film=args.use_force_film,
        force_film_checkpoint=args.force_film_checkpoint,
        force_encoder_checkpoint_path=None
        if args.force_encoder_checkpoint_path in {None, "", "None", "none", "null"}
        else args.force_encoder_checkpoint_path,
        force_encoder_embedding_type=args.force_encoder_embedding_type,
        force_history_length=args.force_history_length,
        constant_force_film_input_before_first_contact=args.constant_force_film_input_before_first_contact,
        run_id_note=args.run_id_note,
        local_log_dir=args.local_log_dir,
        local_video_dir=args.local_video_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
        worker_id=args.worker_id,
        task_orders=parse_task_orders(args.task_orders),
        shared_run_id=args.shared_run_id,
        summary_json_path=args.summary_json_path,
        worker_log_path=args.worker_log_path,
        progress_json_path=args.progress_json_path,
        resume_state_json_path=args.resume_state_json_path,
        worker_video_subdir=args.worker_video_subdir,
        worker_use_wandb=args.worker_use_wandb,
    )


def get_task_suite(task_suite_name: str):
    benchmark_dict = benchmark.get_benchmark_dict()
    return benchmark_dict[task_suite_name]()


def build_policy_task_description(task_language: str) -> str:
    task_language = task_language.strip()
    if not task_language.endswith("."):
        task_language += "."
    return task_language


def create_single_task_env(cfg: GenerateConfig, task_bddl_path: Path, gripper_type: str) -> ControlEnv:
    env = ControlEnv(
        bddl_file_name=str(task_bddl_path),
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types=gripper_type,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_heights=cfg.resolution,
        camera_widths=cfg.resolution,
        control_freq=cfg.control_freq,
        ignore_done=False,
    )
    env.seed(0)
    return env


def should_use_wandb(cfg: GenerateConfig) -> bool:
    if cfg.worker_id is not None:
        return cfg.worker_use_wandb
    return cfg.use_wandb


def get_selected_task_orders(cfg: GenerateConfig, task_suite) -> list[int]:
    if cfg.task_orders is None:
        return list(range(task_suite.n_tasks))
    task_orders = list(cfg.task_orders)
    invalid = [task_order for task_order in task_orders if task_order < 0 or task_order >= task_suite.n_tasks]
    if invalid:
        raise ValueError(f"Task orders out of range for suite {cfg.task_suite_name}: {invalid}")
    return task_orders


def build_run_id(cfg: GenerateConfig) -> str:
    if cfg.shared_run_id is not None:
        if cfg.worker_id is not None:
            return f"{cfg.shared_run_id}--worker{cfg.worker_id:02d}"
        return cfg.shared_run_id

    run_id = f"EVAL-suite-{cfg.task_suite_name}-{cfg.model_family}-{cfg.gripper_type}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def write_summary_json(cfg: GenerateConfig, summary: dict) -> None:
    if cfg.summary_json_path is None:
        return
    summary_path = Path(cfg.summary_json_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def read_json_file(json_path: Union[str, Path]) -> dict[str, Any]:
    """读取 JSON 文件并返回字典。"""
    return json.loads(Path(json_path).read_text())


def write_progress_json(cfg: GenerateConfig, progress_payload: dict[str, Any]) -> None:
    """把 worker 当前可恢复进度写到 progress.json。"""
    # 这里单独维护一个 progress.json，而不是只等程序结束后写 summary.json，
    # 是因为 summary.json 只在 finally 阶段统一落盘；
    # 如果程序在中途被 kill / 断电 / OOM，中间那段已经完成的 episode 就没有结构化状态可恢复。
    if cfg.progress_json_path is None:
        return

    progress_path = Path(cfg.progress_json_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    # 加一层时间戳，方便排查“最后一次成功写进度是什么时候”。
    payload_to_write = dict(progress_payload)
    payload_to_write["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    progress_path.write_text(json.dumps(payload_to_write, ensure_ascii=False, indent=2))


def load_resume_state(cfg: GenerateConfig) -> Optional[dict[str, Any]]:
    """如果 launcher 给当前 worker 下发了恢复计划，就把它读回来。"""
    if cfg.resume_state_json_path is None:
        return None

    resume_state_path = Path(cfg.resume_state_json_path)
    if not resume_state_path.exists():
        raise FileNotFoundError(f"Resume state json not found: {resume_state_path}")
    return read_json_file(resume_state_path)


def build_task_stats_entry(
    *,
    task_id: int,
    task_name: str,
    episodes: int,
    successes: int,
    task_bddl_path: str,
) -> dict[str, Any]:
    """统一构造某个 task 的统计字典。"""
    success_rate = float(successes) / float(episodes) if episodes else 0.0
    return {
        "task_id": task_id,
        "task_name": task_name,
        "episodes": episodes,
        "successes": successes,
        "success_rate": success_rate,
        "task_bddl_path": task_bddl_path,
    }


def build_initial_progress_payload(
    cfg: GenerateConfig,
    *,
    run_id: str,
    selected_task_orders: list[int],
    local_log_filepath: str,
    resume_state: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """把 fresh run / resumed run 统一成一份可持续覆盖写入的 progress 状态。"""
    # fresh run 时，没有任何历史状态，下面这些字段就全部从 0 / 空集合开始。
    # resumed run 时，launcher 会先把“已经完成到哪里”整理成标准化 resume_state，
    # worker 直接接着这个状态继续往下更新。
    if resume_state is None:
        return {
            "run_id": run_id,
            "shared_run_id": cfg.shared_run_id,
            "worker_id": cfg.worker_id,
            "task_orders": selected_task_orders,
            "checkpoint": str(cfg.pretrained_checkpoint),
            "use_force_film": bool(cfg.use_force_film),
            "force_film_checkpoint": cfg.force_film_checkpoint,
            "force_encoder_checkpoint_path": cfg.force_encoder_checkpoint_path,
            "force_encoder_embedding_type": cfg.force_encoder_embedding_type,
            "force_history_length": cfg.force_history_length,
            "constant_force_film_input_before_first_contact": bool(
                cfg.constant_force_film_input_before_first_contact
            ),
            "completed": False,
            "resume_source": "fresh",
            "completed_task_ids": [],
            "current_task_id": None,
            "next_episode_idx_by_task": {str(task_id): 0 for task_id in selected_task_orders},
            "task_stats": {},
            "total_episodes": 0,
            "total_successes": 0,
            "total_rollout_videos": 0,
            "total_success_videos": 0,
            "total_failed_videos": 0,
            "local_log_filepath": local_log_filepath,
            "local_video_dir": cfg.local_video_dir,
            "worker_video_subdir": cfg.worker_video_subdir,
            "error": None,
        }

    progress_payload = dict(resume_state)
    progress_payload["run_id"] = run_id
    progress_payload["shared_run_id"] = cfg.shared_run_id
    progress_payload["worker_id"] = cfg.worker_id
    progress_payload["task_orders"] = selected_task_orders
    progress_payload["checkpoint"] = str(cfg.pretrained_checkpoint)
    progress_payload["use_force_film"] = bool(cfg.use_force_film)
    progress_payload["force_film_checkpoint"] = cfg.force_film_checkpoint
    progress_payload["force_encoder_checkpoint_path"] = cfg.force_encoder_checkpoint_path
    progress_payload["force_encoder_embedding_type"] = cfg.force_encoder_embedding_type
    progress_payload["force_history_length"] = cfg.force_history_length
    progress_payload["constant_force_film_input_before_first_contact"] = bool(
        cfg.constant_force_film_input_before_first_contact
    )
    progress_payload["completed"] = False
    progress_payload["local_log_filepath"] = local_log_filepath
    progress_payload["local_video_dir"] = cfg.local_video_dir
    progress_payload["worker_video_subdir"] = cfg.worker_video_subdir
    progress_payload["error"] = None
    progress_payload.setdefault("resume_source", "resume")
    progress_payload.setdefault("completed_task_ids", [])
    progress_payload.setdefault("current_task_id", None)
    progress_payload.setdefault("next_episode_idx_by_task", {})
    progress_payload.setdefault("task_stats", {})
    progress_payload.setdefault("total_episodes", 0)
    progress_payload.setdefault("total_successes", 0)
    progress_payload.setdefault("total_rollout_videos", 0)
    progress_payload.setdefault("total_success_videos", 0)
    progress_payload.setdefault("total_failed_videos", 0)
    return progress_payload


def sync_progress_payload(
    progress_payload: dict[str, Any],
    *,
    completed_task_ids: set[int],
    current_task_id: Optional[int],
    next_episode_idx_by_task: dict[int, int],
    task_stats_map: dict[int, dict[str, Any]],
    total_episodes: int,
    total_successes: int,
    total_rollout_videos: int,
    total_success_videos: int,
    total_failed_videos: int,
    completed: bool,
    error: Optional[str],
) -> None:
    """把运行时状态同步回统一的 progress_payload。"""
    progress_payload["completed_task_ids"] = sorted(completed_task_ids)
    progress_payload["current_task_id"] = current_task_id
    progress_payload["next_episode_idx_by_task"] = {
        str(task_id): int(next_episode_idx_by_task[task_id]) for task_id in sorted(next_episode_idx_by_task)
    }
    progress_payload["task_stats"] = {
        str(task_id): dict(task_stats_map[task_id]) for task_id in sorted(task_stats_map)
    }
    progress_payload["total_episodes"] = int(total_episodes)
    progress_payload["total_successes"] = int(total_successes)
    progress_payload["total_rollout_videos"] = int(total_rollout_videos)
    progress_payload["total_success_videos"] = int(total_success_videos)
    progress_payload["total_failed_videos"] = int(total_failed_videos)
    progress_payload["completed"] = completed
    progress_payload["error"] = error


def build_completed_task_results(
    *,
    task_stats_map: dict[int, dict[str, Any]],
    completed_task_ids: set[int],
    num_trials: int,
) -> list[dict[str, Any]]:
    """把“已完成 task 的统计”整理成 summary.json 期望的列表格式。"""
    inferred_completed_task_ids = {
        task_id for task_id, task_stats in task_stats_map.items() if int(task_stats.get("episodes", 0)) >= num_trials
    }
    final_completed_task_ids = set(completed_task_ids) | inferred_completed_task_ids
    return [dict(task_stats_map[task_id]) for task_id in sorted(final_completed_task_ids) if task_id in task_stats_map]


def eval_libero_suite(cfg: GenerateConfig) -> None:

    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    checkpoint_str = str(cfg.pretrained_checkpoint)
    if "image_aug" in checkpoint_str:
        assert cfg.center_crop, "Expecting center_crop=True because model was trained with image augmentations!"

    set_seed_everywhere(cfg.seed)
    assert cfg.unnorm_key, "cfg.unnorm_key must not be empty!"

    model = get_model(cfg)
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA norm_stats!"

    processor = get_processor(cfg) if cfg.model_family == "openvla" else None
    task_suite = get_task_suite(cfg.task_suite_name)
    selected_task_orders = get_selected_task_orders(cfg, task_suite)
    resize_size = get_image_resize_size(cfg)
    resume_state = load_resume_state(cfg)

    run_id = build_run_id(cfg)
    Path(cfg.local_log_dir).mkdir(parents=True, exist_ok=True)
    local_log_filepath = cfg.worker_log_path or os.path.join(cfg.local_log_dir, run_id + ".txt")
    # 如果当前 worker 是恢复运行，就以追加模式继续写原日志；
    # 这样旧日志里的已完成 episode 不会被覆盖，恢复后的新增内容也能自然接在后面。
    is_resume_mode = resume_state is not None and resume_state.get("resume_source") != "fresh"
    log_file = open(local_log_filepath, "a" if is_resume_mode and Path(local_log_filepath).exists() else "w")
    use_wandb = should_use_wandb(cfg)

    progress_payload = build_initial_progress_payload(
        cfg,
        run_id=run_id,
        selected_task_orders=selected_task_orders,
        local_log_filepath=local_log_filepath,
        resume_state=resume_state,
    )
    completed_task_ids = {int(task_id) for task_id in progress_payload.get("completed_task_ids", [])}
    next_episode_idx_by_task = {
        int(task_id): int(episode_idx)
        for task_id, episode_idx in progress_payload.get("next_episode_idx_by_task", {}).items()
    }
    task_stats_map = {
        int(task_id): dict(task_stats) for task_id, task_stats in progress_payload.get("task_stats", {}).items()
    }
    for task_order in selected_task_orders:
        next_episode_idx_by_task.setdefault(task_order, 0)

    total_episodes = int(progress_payload.get("total_episodes", 0))
    total_successes = int(progress_payload.get("total_successes", 0))
    total_rollout_videos = int(progress_payload.get("total_rollout_videos", 0))
    total_success_videos = int(progress_payload.get("total_success_videos", 0))
    total_failed_videos = int(progress_payload.get("total_failed_videos", 0))
    final_suite_success_rate = 0.0
    error_message = None

    print(f"Logging to local log file: {local_log_filepath}")
    print(f"Task suite: {cfg.task_suite_name}")
    print(f"Checkpoint: {cfg.pretrained_checkpoint}")
    print(f"Use force FiLM: {cfg.use_force_film}")
    if cfg.use_force_film:
        print(f"Force-FiLM checkpoint: {cfg.force_film_checkpoint}")
        print(f"Force-FiLM encoder checkpoint: {cfg.force_encoder_checkpoint_path}")
        print(f"Force-FiLM encoder embedding type: {cfg.force_encoder_embedding_type}")
        print(f"Force-FiLM history length: {cfg.force_history_length}")
        print(
            "Constant Force-FiLM input before first contact: "
            f"{cfg.constant_force_film_input_before_first_contact}"
        )
    print(f"Local video dir: {cfg.local_video_dir}")
    print(f"Selected task orders: {selected_task_orders}")
    if cfg.worker_id is not None:
        print(f"Worker ID: {cfg.worker_id}")
        print(f"Worker video subdir: {cfg.worker_video_subdir}")
    if is_resume_mode:
        print(f"Resume source: {progress_payload.get('resume_source')}")
        print(f"Completed tasks before resume: {sorted(completed_task_ids)}")
        print(f"Next episode map before resume: {next_episode_idx_by_task}")

    # fresh run 时写一份全新的头部信息；
    # resumed run 时，在原日志末尾加一个明显的恢复分隔块，方便后续人工排查。
    if is_resume_mode and Path(local_log_filepath).exists():
        log_file.write("\n================ RESUME ================\n")
        log_file.write(f"Resume source: {progress_payload.get('resume_source')}\n")
        log_file.write(f"Completed tasks before resume: {sorted(completed_task_ids)}\n")
        log_file.write(f"Next episode map before resume: {next_episode_idx_by_task}\n")
    else:
        log_file.write(f"Task suite: {cfg.task_suite_name}\n")
        log_file.write(f"Checkpoint: {cfg.pretrained_checkpoint}\n")
        log_file.write(f"Use force FiLM: {cfg.use_force_film}\n")
        if cfg.use_force_film:
            log_file.write(f"Force-FiLM checkpoint: {cfg.force_film_checkpoint}\n")
            log_file.write(f"Force-FiLM encoder checkpoint: {cfg.force_encoder_checkpoint_path}\n")
            log_file.write(f"Force-FiLM encoder embedding type: {cfg.force_encoder_embedding_type}\n")
            log_file.write(f"Force-FiLM history length: {cfg.force_history_length}\n")
            log_file.write(
                "Constant Force-FiLM input before first contact: "
                f"{cfg.constant_force_film_input_before_first_contact}\n"
            )
        log_file.write(f"Local video dir: {cfg.local_video_dir}\n")
        log_file.write(f"Selected task orders: {selected_task_orders}\n")
        if cfg.worker_id is not None:
            log_file.write(f"Worker ID: {cfg.worker_id}\n")
            log_file.write(f"Worker video subdir: {cfg.worker_video_subdir}\n")
    log_file.flush()

    # 启动时先把当前恢复计划落一份到 progress.json。
    # 这样即使程序刚启动不久就中断，launcher 也至少能知道这个 worker 准备从哪个 task / episode 接着跑。
    sync_progress_payload(
        progress_payload,
        completed_task_ids=completed_task_ids,
        current_task_id=None,
        next_episode_idx_by_task=next_episode_idx_by_task,
        task_stats_map=task_stats_map,
        total_episodes=total_episodes,
        total_successes=total_successes,
        total_rollout_videos=total_rollout_videos,
        total_success_videos=total_success_videos,
        total_failed_videos=total_failed_videos,
        completed=False,
        error=None,
    )
    write_progress_json(cfg, progress_payload)

    if use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    try:
        task_progress_desc = "Tasks" if cfg.worker_id is None else f"Tasks(worker_{cfg.worker_id:02d})"
        for task_order in tqdm.tqdm(selected_task_orders, desc=task_progress_desc):
            # 如果 launcher 已经确认这个 task 在历史运行里完整完成，就直接跳过，不重复评测。
            if task_order in completed_task_ids:
                continue

            task = task_suite.get_task(task_order)
            initial_states = task_suite.get_task_init_states(task_order)
            task_bddl_path = LIBERO_DIR / "libero" / "libero" / "bddl_files" / task.problem_folder / task.bddl_file
            if not task_bddl_path.exists():
                raise FileNotFoundError(f"Task BDDL file not found: {task_bddl_path}")

            policy_task_description = build_policy_task_description(task.language)
            env = None
            env_default = None
            rollout_video_records = []
            # 如果是恢复运行，这里先把这个 task 已经完成的 episode / success 预热回来，
            # 后面只对“剩余没跑的 episode”继续累加。
            existing_task_stats = task_stats_map.get(
                task_order,
                build_task_stats_entry(
                    task_id=task_order,
                    task_name=task.language,
                    episodes=0,
                    successes=0,
                    task_bddl_path=str(task_bddl_path),
                ),
            )
            task_episodes = int(existing_task_stats.get("episodes", 0))
            task_successes = int(existing_task_stats.get("successes", 0))
            start_episode_idx = int(next_episode_idx_by_task.get(task_order, task_episodes))

            try:
                env = create_single_task_env(cfg, task_bddl_path, cfg.gripper_type)
                env_default = create_single_task_env(cfg, task_bddl_path, "default")
                env_default.reset()

                episode_desc = f"Episodes(task_{task_order:02d})"
                if cfg.worker_id is not None:
                    episode_desc += f"-worker_{cfg.worker_id:02d}"

                # 如果这个 task 的 episode 已经全部完成，这里直接跳过 episode 主循环。
                if start_episode_idx >= cfg.num_trials:
                    completed_task_ids.add(task_order)
                    next_episode_idx_by_task[task_order] = cfg.num_trials
                    task_stats_map[task_order] = build_task_stats_entry(
                        task_id=task_order,
                        task_name=task.language,
                        episodes=task_episodes,
                        successes=task_successes,
                        task_bddl_path=str(task_bddl_path),
                    )
                    continue

                if start_episode_idx > 0:
                    print(f"Resuming task {task_order} from episode {start_episode_idx + 1}/{cfg.num_trials}")
                    log_file.write(f"Resuming task {task_order} from episode {start_episode_idx + 1}/{cfg.num_trials}\n")
                    log_file.flush()

                for episode_idx in tqdm.tqdm(range(start_episode_idx, cfg.num_trials), desc=episode_desc, leave=False):
                    # 在 episode 真正开始之前，先把“下一条要跑的 episode”写入 progress.json。
                    # 这样哪怕当前 episode 中途挂掉，恢复时也会从这个 episode 重新跑，而不是错误地跳到下一条。
                    next_episode_idx_by_task[task_order] = episode_idx
                    sync_progress_payload(
                        progress_payload,
                        completed_task_ids=completed_task_ids,
                        current_task_id=task_order,
                        next_episode_idx_by_task=next_episode_idx_by_task,
                        task_stats_map=task_stats_map,
                        total_episodes=total_episodes,
                        total_successes=total_successes,
                        total_rollout_videos=total_rollout_videos,
                        total_success_videos=total_success_videos,
                        total_failed_videos=total_failed_videos,
                        completed=False,
                        error=None,
                    )
                    write_progress_json(cfg, progress_payload)

                    init_state_idx = episode_idx % len(initial_states)
                    success = False
                    t = 0
                    last_info = None
                    # 只有显式要求保存视频时，当前 episode 才进入视频录制路径。
                    # 这样当 `save_all_rollout_videos=False` 时，我们既不缓存帧，也不构帧。
                    record_video = cfg.save_all_rollout_videos
                    replay_images = [] if record_video else None
                    video_writer = None
                    video_tmp_path = None

                    print(f"\nTask {task_order}: {task.language}")
                    print(f"Task BDDL: {task_bddl_path}")
                    print(f"输入到vla的指令: {policy_task_description}")
                    print(f"Starting episode {episode_idx + 1}/{cfg.num_trials} with init state {init_state_idx}...")
                    log_file.write(f"\nTask {task_order}: {task.language}\n")
                    log_file.write(f"Task BDDL: {task_bddl_path}\n")
                    log_file.write(f"输入到vla的指令: {policy_task_description}\n")
                    log_file.write(f"Starting episode {episode_idx + 1}/{cfg.num_trials} with init state {init_state_idx}...\n")

                    env.reset()
                    converted_state = convert_default_init_state_to_suction(
                        init_state_default=initial_states[init_state_idx],
                        env_default=env_default,
                        env_suction=env,
                    )
                    obs = env.set_init_state(converted_state)
                    # 每条 episode 重新创建左侧补零的 [6,T] 力历史窗口，避免上一条轨迹的信息串入当前轨迹。
                    suction_wrench_history = create_suction_wrench_history(cfg.force_history_length)
                    # action_history 和 suction_wrench_history 使用完全相同的滚动方式。
                    # 每一列代表同一个控制时刻：当前时刻读到的 wrench，以及该时刻马上要执行的 action。
                    action_history = create_action_history(cfg.force_history_length)
                    # 吸附失败触发后，回溯不是单步完成，而是拆成多步 action 队列。
                    # 队列非空时先执行回溯动作，等队列清空后再恢复 VLA 决策。
                    pending_rewind_actions: list[np.ndarray] = []
                    # 记录当前 episode 是否曾经出现过“吸附失败”。
                    # 这个标志一旦变成 True，就一直保持到 episode 结束，用来把视频分到四类目录里。
                    episode_has_suction_failure = False
                    # 记录当前 episode 内已经检测到多少次吸附失败。
                    # 后续确定性扰动会用这个次数选择方向并计算幅值，让多次失败时扰动逐步变明显。
                    suction_failure_count_in_episode = 0
                    # 记录 Force-FiLM 输入是否已经看到“第一次接触”。
                    # 这里按实验设定使用严格条件 Fz != 0.0；一旦触发，本条 episode 后续都恢复真实力历史输入。
                    force_film_first_contact_seen = False

                    if record_video:
                        # 流式写视频时，先写到临时目录，等 episode 结束后再按四分类结果归档。
                        video_parent_subdir = cfg.worker_video_subdir or run_id
                        video_writer, video_tmp_path = open_rollout_video_writer(
                            total_episodes + 1,
                            task.language,
                            log_file=log_file,
                            subdir=f"{video_parent_subdir}/tmp",
                            root_dir=cfg.local_video_dir,
                            fps=cfg.rollout_video_fps,
                        )

                    while t < cfg.max_steps + cfg.num_steps_wait:
                        try:
                            if t < cfg.num_steps_wait:
                                # 等待阶段也推进两个历史窗口。
                                # 注意：这里先读取当前时刻的 wrench，再构造同一时刻要执行的 dummy action，
                                # 然后一起写入历史窗口，最后才调用 env.step(dummy_action)。
                                # 这样 CSV 同一行里的 wrench/action 一定对应同一个控制时刻。
                                current_suction_wrench = extract_suction_wrench(obs, last_info)
                                if float(current_suction_wrench[2]) != 0.0:
                                    # 等待阶段理论上通常还没碰到物体；这里仍然更新状态，避免少数初始化接触场景污染实验。
                                    force_film_first_contact_seen = True
                                dummy_action = get_libero_dummy_action(cfg.model_family)
                                suction_wrench_history = update_suction_wrench_history(
                                    suction_wrench_history,
                                    current_suction_wrench,
                                )
                                action_history = update_action_history(action_history, dummy_action)
                                obs, _, done, last_info = env.step(dummy_action)
                                if record_video:
                                    # 恢复原始视频保存时序：先执行 dummy action，再保存 step 后的新画面。
                                    frame = create_combined_frame(
                                        obs["agentview_image"],
                                        extract_suction_wrench(obs, last_info),
                                        obs["robot0_eye_in_hand_image"],
                                        img_size=cfg.resolution,
                                    )
                                    # 流式模式下只保留当前帧，立即写盘，不再把整条 episode 攒在内存里。
                                    if video_writer is not None:
                                        video_writer.append_data(frame)
                                    elif replay_images is not None:
                                        replay_images.append(frame)
                                success = bool(done or env.check_success())
                                t += 1
                                if success:
                                    break
                                continue

                            img = get_libero_image(obs, resize_size)
                            observation = {"full_image": img}
                            # 这一帧的 suction_wrench 与当前 observation 对齐；更新后的 [6,T] 窗口用于生成本步 FiLM 参数。
                            current_suction_wrench = extract_suction_wrench(obs, last_info)
                            suction_wrench_history = update_suction_wrench_history(
                                suction_wrench_history,
                                current_suction_wrench,
                            )
                            if float(current_suction_wrench[2]) != 0.0:
                                # 第一次接触判定只看真实 Fz；触发后本控制时刻开始就给模型传真实历史窗口。
                                force_film_first_contact_seen = True
                            suction_failed = False
                            # 这个变量只用于视频显示，不参与控制。
                            # 0 表示当前执行的是 VLA action；大于 0 表示当前正在执行回溯 action。
                            rewind_actions_remaining_for_video = 0
                            if pending_rewind_actions:
                                # 回溯队列还没执行完时，本步不再调用 VLA，也不重新触发失败检测。
                                # 这样可以避免“回溯过程中再次检测失败，再套一层回溯”的循环。
                                action = pending_rewind_actions.pop(0)
                                # pop 之后列表里只剩未来步骤，所以这里额外加 1，把当前马上执行的回溯 action 算进去。
                                rewind_actions_remaining_for_video = 1 + len(pending_rewind_actions)
                                action_history = update_action_history(action_history, action)
                            else:
                                # 只在传给 VLA / Force-FiLM 时按开关替换六维力历史。
                                # suction_wrench_history 自身仍然保留真实值，后面的 CSV、吸附失败检测和视频显示都继续用真实力数据。
                                # 第一次真实 Fz != 0.0 后，立即恢复整段真实历史窗口，避免 20.0 在接触后继续残留。
                                force_film_suction_wrench = (
                                    build_constant_force_film_input(suction_wrench_history)
                                    if (
                                        cfg.constant_force_film_input_before_first_contact
                                        and not force_film_first_contact_seen
                                    )
                                    else suction_wrench_history
                                )
                                action = get_action(
                                    cfg,
                                    model,
                                    observation,
                                    policy_task_description,
                                    processor=processor,
                                    suction_wrench=force_film_suction_wrench,
                                )
                                action = normalize_gripper_action(action, binarize=True)
                                if cfg.model_family == "openvla":
                                    action = invert_gripper_action(action)
                                # 只对 VLA 直接生成的 action 应用这个规则：
                                # 当前 Fz>0 表示吸盘正在与物体正向接触，此时强制开启吸盘，避免模型在接触瞬间关吸盘。
                                # 注意：回溯队列 action 不走这个分支，因此不会被这个规则二次覆盖。
                                action = apply_contact_force_suction_override(
                                    action=action,
                                    suction_wrench=current_suction_wrench,
                                    enabled=cfg.enable_contact_force_suction_override,
                                )

                                # action_history 的最后一列与 suction_wrench_history 的最后一列对应同一个控制时刻。
                                # 两者都在 env.step(action) 之前写入，记录的是“当前 wrench + 马上执行的 action”。
                                action_history = update_action_history(action_history, action)
                                if cfg.enable_suction_failure_detection:
                                    suction_failed, _failure_debug = detect_suction_failure_from_history(
                                        suction_wrench_history=suction_wrench_history,
                                        action_history=action_history,
                                        contact_fz_threshold=cfg.suction_failure_contact_fz_threshold,
                                        free_fz_threshold=cfg.suction_failure_free_fz_threshold,
                                        lift_action_z_threshold=cfg.suction_failure_lift_action_z_threshold,
                                        contact_history_steps=cfg.suction_failure_contact_history_steps,
                                        min_contact_steps=cfg.suction_failure_min_contact_steps,
                                    )
                                    if suction_failed:
                                        # 只要当前 episode 中任意一个控制时刻触发过 suction_failed，
                                        # 最后保存视频时就归入“有吸附失败”的类别。
                                        episode_has_suction_failure = True
                                        # 这个计数只在非回溯队列分支里增加；回溯执行期间不会再次检测失败，
                                        # 因此不会出现同一段回溯过程里反复叠加扰动的情况。
                                        suction_failure_count_in_episode += 1
                                    if suction_failed and cfg.enable_suction_failure_rewind:
                                        rewind_actions = build_suction_failure_rewind_action_sequence(
                                            suction_wrench_history=suction_wrench_history,
                                            action_history=action_history,
                                            contact_fz_threshold=cfg.suction_failure_contact_fz_threshold,
                                            pre_contact_steps=cfg.suction_rewind_pre_contact_steps,
                                        )
                                        if rewind_actions:
                                            if cfg.enable_suction_failure_escape_perturb:
                                                # 扰动动作追加在回溯动作之后。
                                                # 它只改变 x/y/z 三个平移维度，用来把末端带到一个不同状态，
                                                # 然后再把控制权交还给 VLA，而不是靠人工规则直接完成吸附。
                                                escape_perturb_action = build_suction_failure_escape_perturb_action(
                                                    suction_failure_count=suction_failure_count_in_episode,
                                                    perturb_scale_step=cfg.suction_escape_perturb_scale_step,
                                                )
                                                rewind_actions.append(escape_perturb_action)
                                            # 多步回溯会从本控制时刻立刻开始执行。
                                            # 第一条 action 替换当前 VLA action，剩余 action 存入队列给后续控制时刻继续执行。
                                            action = rewind_actions.pop(0)
                                            pending_rewind_actions.extend(rewind_actions)
                                            # 当前帧已经开始执行第一步回溯，所以视频里显示完整的回溯剩余步数。
                                            rewind_actions_remaining_for_video = 1 + len(pending_rewind_actions)
                                            # 覆盖 action_history 最后一列，保证 CSV 记录的是环境真正执行的回溯动作。
                                            action_history[:, -1] = action
                            if record_video:
                                # 正式推理阶段的视频帧使用 env.step(action) 之前的当前 obs。
                                # 因此画面、左侧 Fz 数值、suction_failed 红字和即将执行的 action 是同一控制时刻。
                                frame = create_combined_frame(
                                    obs["agentview_image"],
                                    current_suction_wrench,
                                    obs["robot0_eye_in_hand_image"],
                                    img_size=cfg.resolution,
                                )
                                if cfg.show_suction_failure_overlay_in_video and suction_failed:
                                    # 红字直接画在当前控制时刻的画面上，不再滞后一帧。
                                    frame = draw_suction_failure_overlay(frame)
                                if rewind_actions_remaining_for_video > 0:
                                    # 回溯动作计数直接叠到视频里，方便区分当前执行的是回溯 action 还是 VLA action。
                                    frame = draw_suction_rewind_action_count_overlay(
                                        frame,
                                        rewind_actions_remaining_for_video,
                                    )
                                # 流式模式下只保留当前帧，立即写盘，不再把整条 episode 攒在内存里。
                                if video_writer is not None:
                                    video_writer.append_data(frame)
                                elif replay_images is not None:
                                    replay_images.append(frame)
                            obs, _, done, last_info = env.step(action.tolist())
                            success = bool(done or env.check_success())
                            if success:
                                break
                            t += 1
                        except Exception as exc:
                            # 推理 / 环境 step / 视频写入中的任何异常都说明当前评测已经不可信。
                            # 这里不能再把它当作普通失败 episode 继续统计，否则会掩盖真实 bug，
                            # 表现成“全失败 + 一帧视频”，很难区分是策略差还是代码崩了。
                            error_context = (
                                f"Fatal rollout exception at task={task_order}, "
                                f"episode={episode_idx + 1}/{cfg.num_trials}, step={max(0, t - cfg.num_steps_wait)}"
                            )
                            traceback_text = traceback.format_exc()
                            print(error_context)
                            print(traceback_text)
                            log_file.write(f"{error_context}\n")
                            log_file.write(traceback_text)
                            log_file.flush()

                            # 如果异常发生时正在流式写 tmp 视频，先尽力关闭 writer，
                            # 再删除未完成的 tmp mp4，避免留下误导性的一帧失败视频。
                            if video_writer is not None:
                                try:
                                    video_writer.close()
                                except Exception as close_exc:
                                    log_file.write(f"Failed to close unfinished video writer: {close_exc}\n")
                                video_writer = None
                            if video_tmp_path is not None and os.path.exists(video_tmp_path):
                                try:
                                    os.remove(video_tmp_path)
                                    log_file.write(f"Removed unfinished rollout MP4: {video_tmp_path}\n")
                                except Exception as remove_exc:
                                    log_file.write(f"Failed to remove unfinished rollout MP4 {video_tmp_path}: {remove_exc}\n")
                                video_tmp_path = None
                            log_file.flush()

                            raise RuntimeError(error_context) from exc

                    task_episodes += 1
                    total_episodes += 1
                    if success:
                        task_successes += 1
                        total_successes += 1

                    # 这里按新的严格布尔语义执行：
                    # - False：完全不保存视频
                    # - True：无论成功还是失败，都保存该条 rollout 视频
                    video_parent_subdir = cfg.worker_video_subdir or run_id
                    # 按“最终成功/失败”和“过程中是否出现过吸附失败”组合成四类输出目录。
                    # 视频和 CSV 共用这套目录分类；关闭视频时，CSV 仍然会保存到同样位置。
                    video_category = get_rollout_video_category(
                        success=success,
                        has_suction_failure=episode_has_suction_failure,
                    )
                    subdir = f"{video_parent_subdir}/{video_category}"
                    if subdir is not None:
                        output_dir = Path(cfg.local_video_dir) / DATE / subdir
                    else:
                        output_dir = Path(cfg.local_video_dir) / DATE
                    # 先独立构造“本来会保存的视频路径”，即使 record_video=False 也能得到同名 CSV 路径。
                    # 这样 SAVE_ALL_ROLLOUT_VIDEOS=False 只会关闭 MP4 写入，不会影响 action/wrench CSV。
                    expected_video_path = output_dir / _build_rollout_video_filename(
                        total_episodes,
                        success,
                        task.language,
                    )
                    video_path = str(expected_video_path)

                    if record_video:
                        if video_writer is not None and video_tmp_path is not None:
                            video_path = finalize_rollout_video(
                                video_writer,
                                video_tmp_path,
                                total_episodes,
                                success=success,
                                task_description=task.language,
                                log_file=log_file,
                                subdir=subdir,
                                root_dir=cfg.local_video_dir,
                            )
                            video_writer = None
                            video_tmp_path = None
                        else:
                            video_path = save_rollout_video(
                                replay_images or [],
                                total_episodes,
                                success=success,
                                task_description=task.language,
                                log_file=log_file,
                                subdir=subdir,
                                root_dir=cfg.local_video_dir,
                                fps=cfg.rollout_video_fps,
                            )
                        rollout_video_records.append((success, video_path))

                    if cfg.save_action_wrench_history_csv:
                        save_action_wrench_history_csv(
                            csv_path=Path(video_path).with_suffix(".csv"),
                            suction_wrench_history=suction_wrench_history,
                            action_history=action_history,
                            log_file=log_file,
                        )

                    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes else 0.0
                    suite_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0

                    print(f"Success: {success}")
                    print(f"Steps executed: {max(0, t - cfg.num_steps_wait)} / {cfg.max_steps}")
                    print(f"# 当前任务 episodes: {task_episodes}")
                    print(f"# 当前任务 successes: {task_successes} ({task_success_rate * 100:.1f}%)")
                    print(f"# 套件累计 episodes: {total_episodes}")
                    print(f"# 套件累计 successes: {total_successes} ({suite_success_rate * 100:.1f}%)")

                    log_file.write(f"Success: {success}\n")
                    log_file.write(f"Steps executed: {max(0, t - cfg.num_steps_wait)} / {cfg.max_steps}\n")
                    log_file.write(f"# 当前任务 episodes: {task_episodes}\n")
                    log_file.write(f"# 当前任务 successes: {task_successes} ({task_success_rate * 100:.1f}%)\n")
                    log_file.write(f"# 套件累计 episodes: {total_episodes}\n")
                    log_file.write(f"# 套件累计 successes: {total_successes} ({suite_success_rate * 100:.1f}%)\n")
                    log_file.flush()

                    # 一个 episode 顺利走到结尾之后，再把“下一个待跑 episode 索引”推进一格。
                    # 这样 progress.json 里记录的就是“下一次恢复时应该从哪一条 episode 开始”。
                    next_episode_idx_by_task[task_order] = episode_idx + 1
                    task_stats_map[task_order] = build_task_stats_entry(
                        task_id=task_order,
                        task_name=task.language,
                        episodes=task_episodes,
                        successes=task_successes,
                        task_bddl_path=str(task_bddl_path),
                    )
                    sync_progress_payload(
                        progress_payload,
                        completed_task_ids=completed_task_ids,
                        current_task_id=task_order,
                        next_episode_idx_by_task=next_episode_idx_by_task,
                        task_stats_map=task_stats_map,
                        total_episodes=total_episodes,
                        total_successes=total_successes,
                        total_rollout_videos=total_rollout_videos,
                        total_success_videos=total_success_videos,
                        total_failed_videos=total_failed_videos,
                        completed=False,
                        error=None,
                    )
                    write_progress_json(cfg, progress_payload)

                    if use_wandb and cfg.worker_id is None:
                        wandb.log({"success_rate/suite_running": suite_success_rate})

                task_success_rate = float(task_successes) / float(task_episodes) if task_episodes else 0.0
                suite_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
                print(f"Current task success rate: {task_success_rate}")
                print(f"Current suite success rate: {suite_success_rate}")
                log_file.write(f"Current task success rate: {task_success_rate}\n")
                log_file.write(f"Current suite success rate: {suite_success_rate}\n")
                log_file.flush()

                task_stats_map[task_order] = build_task_stats_entry(
                    task_id=task_order,
                    task_name=task.language,
                    episodes=task_episodes,
                    successes=task_successes,
                    task_bddl_path=str(task_bddl_path),
                )
                completed_task_ids.add(task_order)
                next_episode_idx_by_task[task_order] = cfg.num_trials
                sync_progress_payload(
                    progress_payload,
                    completed_task_ids=completed_task_ids,
                    current_task_id=None,
                    next_episode_idx_by_task=next_episode_idx_by_task,
                    task_stats_map=task_stats_map,
                    total_episodes=total_episodes,
                    total_successes=total_successes,
                    total_rollout_videos=total_rollout_videos,
                    total_success_videos=total_success_videos,
                    total_failed_videos=total_failed_videos,
                    completed=False,
                    error=None,
                )
                write_progress_json(cfg, progress_payload)

                if use_wandb and cfg.worker_id is None and task_episodes > 0:
                    wandb.log({f"success_rate/task_{task_order:02d}": task_success_rate})
            finally:
                success_video_count = sum(1 for was_success, _ in rollout_video_records if was_success)
                failed_video_count = sum(1 for was_success, _ in rollout_video_records if not was_success)
                total_rollout_videos += len(rollout_video_records)
                total_success_videos += success_video_count
                total_failed_videos += failed_video_count
                log_file.write(f"All rollout videos saved: {len(rollout_video_records)}\n")
                log_file.write(f"Successful rollout videos: {success_video_count}\n")
                log_file.write(f"Failed rollout videos: {failed_video_count}\n")
                for was_success, video_path in rollout_video_records:
                    status = "success" if was_success else "failure"
                    log_file.write(f"Rollout video ({status}): {video_path}\n")
                log_file.flush()

                if env is not None:
                    env.close()
                if env_default is not None:
                    env_default.close()

        final_suite_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
        print(f"Final suite success rate: {final_suite_success_rate}")
        log_file.write(f"Final suite success rate: {final_suite_success_rate}\n")
        log_file.write(f"Total rollout videos: {total_rollout_videos}\n")
        log_file.write(f"Total successful rollout videos: {total_success_videos}\n")
        log_file.write(f"Total failed rollout videos: {total_failed_videos}\n")
        log_file.flush()

        if use_wandb and cfg.worker_id is None and total_episodes > 0:
            wandb.log({"success_rate/suite_final": final_suite_success_rate})
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"Fatal evaluation error: {error_message}")
        log_file.write(f"Fatal evaluation error: {error_message}\n")
        log_file.flush()
        sync_progress_payload(
            progress_payload,
            completed_task_ids=completed_task_ids,
            current_task_id=progress_payload.get("current_task_id"),
            next_episode_idx_by_task=next_episode_idx_by_task,
            task_stats_map=task_stats_map,
            total_episodes=total_episodes,
            total_successes=total_successes,
            total_rollout_videos=total_rollout_videos,
            total_success_videos=total_success_videos,
            total_failed_videos=total_failed_videos,
            completed=False,
            error=error_message,
        )
        write_progress_json(cfg, progress_payload)
        raise
    finally:
        sync_progress_payload(
            progress_payload,
            completed_task_ids=completed_task_ids,
            current_task_id=None,
            next_episode_idx_by_task=next_episode_idx_by_task,
            task_stats_map=task_stats_map,
            total_episodes=total_episodes,
            total_successes=total_successes,
            total_rollout_videos=total_rollout_videos,
            total_success_videos=total_success_videos,
            total_failed_videos=total_failed_videos,
            completed=error_message is None,
            error=error_message,
        )
        write_progress_json(cfg, progress_payload)
        summary = {
            "run_id": run_id,
            "shared_run_id": cfg.shared_run_id,
            "worker_id": cfg.worker_id,
            "task_orders": selected_task_orders,
            "checkpoint": str(cfg.pretrained_checkpoint),
            "use_force_film": bool(cfg.use_force_film),
            "force_film_checkpoint": cfg.force_film_checkpoint,
            "force_encoder_checkpoint_path": cfg.force_encoder_checkpoint_path,
            "force_encoder_embedding_type": cfg.force_encoder_embedding_type,
            "force_history_length": cfg.force_history_length,
            "constant_force_film_input_before_first_contact": bool(
                cfg.constant_force_film_input_before_first_contact
            ),
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "suite_success_rate": final_suite_success_rate,
            "total_rollout_videos": total_rollout_videos,
            "total_success_videos": total_success_videos,
            "total_failed_videos": total_failed_videos,
            "task_results": build_completed_task_results(
                task_stats_map=task_stats_map,
                completed_task_ids=completed_task_ids,
                num_trials=cfg.num_trials,
            ),
            "local_log_filepath": local_log_filepath,
            "local_video_dir": cfg.local_video_dir,
            "worker_video_subdir": cfg.worker_video_subdir,
            "completed": error_message is None,
            "error": error_message,
        }
        write_summary_json(cfg, summary)
        if use_wandb:
            wandb.finish()
        log_file.close()


if __name__ == "__main__":
    eval_libero_suite(build_config_from_cli())
