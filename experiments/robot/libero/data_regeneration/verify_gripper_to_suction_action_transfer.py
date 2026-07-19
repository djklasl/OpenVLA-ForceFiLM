# 本脚本用于验证“夹爪式机械臂的动作空间”是否与“吸盘式机械臂的动作空间”相同。
# 做法是：把 libero_suction 任务套件里、原本为“夹爪(default gripper)”采集的 7 维动作
#         原封不动地喂给“吸盘(suction)”环境回放，然后统计任务成功率。
# 输入：libero_suction 任务套件（默认，直接跑即可），以及 --libero_raw_data_dir 下的原始夹爪 HDF5。
# 输出：控制台实时进度条 + 成功率统计；并在 --output_json 落盘一份结构化 JSON 结果；
#       默认会把每条 demo 的回放视频保存成 mp4（成功/失败分文件名区分），便于人工观察失败原因；
#       若不想存视频可传 --save_rollout_video false。

"""
verify_gripper_to_suction_action_transfer.py

【脚本作用】
纯离线、全自动脚本，用来验证：
    “夹爪式 Panda 机械臂的动作空间”和“吸盘式 Panda 机械臂的动作空间”到底一不一样。

【当前默认跑 libero_suction 这一个套件】
本脚本目前默认处理 libero_suction 任务套件（--libero_task_suite 默认值就是它，直接运行即可）。
关于“为什么 libero_suction 名字带 suction 却当夹爪源数据来用”的详细说明，
见下方 SUPPORTED_GRIPPER_TASK_SUITES 处的注释。

核心思路：
    1. libero_suction 套件里的 actions 是为“default gripper（夹爪）”采集的 7 维动作；
    2. 这 7 维分别是 dx, dy, dz, dax, day, daz（末端位姿增量）+ 最后一维夹爪开合[-1,+1]；
    3. 我们把这些动作【原封不动】地喂给一个“suction（吸盘）”环境；
    4. 然后看吸盘环境能不能完成任务（用吸盘环境自身的 check_success 判定）；
    5. 如果成功率很高，说明两者运动空间基本兼容；如果成功率很低，说明不兼容。

与现有 collect_libero_suction_handover_dataset.py 的区别：
    - 那个脚本是“半自动”：前段自动回放夹爪动作、后段人工接管；
    - 本脚本是“全自动”：从头到尾全部用原始夹爪动作回放，不做任何人工介入；
    - 那个脚本最后一维会被强制改成 -1.0（吸盘关闭）；
    - 本脚本【不修改任何一维】，包括最后一维，正是为了真正测出运动空间是否相同。

【运行示例】
先激活环境：
    conda activate openvla

小规模快速验证：
    python experiments/robot/libero/data_regeneration/verify_gripper_to_suction_action_transfer.py \\
        --task_limit 2 --demo_limit_per_task 2

全量跑（默认就会保存每条 demo 的回放视频）：
    python experiments/robot/libero/data_regeneration/verify_gripper_to_suction_action_transfer.py

如果不想存视频（省磁盘 / 省显存），显式传 false：
    python experiments/robot/libero/data_regeneration/verify_gripper_to_suction_action_transfer.py \\
        --save_rollout_video false
"""

from __future__ import annotations
# 开启更灵活的类型注解语法（例如 list[dict] 这种写法）。

# ==============================================================================
# 线程数限制 + 渲染后端设置
# ==============================================================================
# 这一段必须放在 import torch / mujoco / libero 之前才会生效。
#
# 为什么这么做：
# LIBERO 仿真器（robosuite + mujoco）在启动时，默认会按 CPU 核数拉起很多线程，
# 但底层 mujoco 物理仿真实际只需要 1~2 个线程就够用了。
# 如果不限制，大量线程会在 CPU 上互相抢占，反而拖慢评测速度。
# 实测下来：
#   - OMP_NUM_THREADS=2（OpenMP 用 2 个线程）
#   - MKL_NUM_THREADS=1（Intel MKL 只用 1 个线程）
# 这样一张 A100 上可以同时挂 4 个评测进程，GPU 利用率稳定在 60~70%，
# 同时 CPU 占用也不高，整体吞吐最大。
import os


def _apply_thread_limits_before_imports():
    """
    在脚本最早期把线程数环境变量写好。

    这里故意在 import torch / libero 之前就完成写入，
    因为这些库在 import 时就会读取 OMP_NUM_THREADS / MKL_NUM_THREADS，
    import 之后再改环境变量就晚了。
    """
    # 先把命令行参数里可能指定的线程数解析出来（如果有的话）。
    # 用一个轻量解析，避免在设置环境变量之前就 import argparse 之外的重库。
    omp_threads = "2"  # 默认值，和用户实测最优值一致。
    mkl_threads = "1"  # 默认值，和用户实测最优值一致。
    # 扫描 sys.argv，看用户有没有在命令行里覆盖这两个值。
    # 这样即便用 os.environ.setdefault，命令行参数也能优先生效。
    import sys as _sys

    argv = _sys.argv
    for i, token in enumerate(argv):
        if token == "--omp_threads" and i + 1 < len(argv):
            omp_threads = argv[i + 1]
        elif token.startswith("--omp_threads="):
            omp_threads = token.split("=", 1)[1]
        elif token == "--mkl_threads" and i + 1 < len(argv):
            mkl_threads = argv[i + 1]
        elif token.startswith("--mkl_threads="):
            mkl_threads = token.split("=", 1)[1]

    # setdefault 的含义是：如果环境变量已经被外层 shell export 了，就尊重外层设置；
    # 否则才用我们这里解析出的默认值或命令行参数值。
    os.environ.setdefault("OMP_NUM_THREADS", omp_threads)
    os.environ.setdefault("MKL_NUM_THREADS", mkl_threads)
    # MUJOCO_GL=egl 表示用 EGL 做离屏渲染，方便在无显示器的服务器上跑。
    os.environ.setdefault("MUJOCO_GL", "egl")


_apply_thread_limits_before_imports()


# ==============================================================================
# 标准库 import
# ==============================================================================
import argparse
# 命令行参数解析。
import json
# 读写 JSON（用于落盘评测结果）。
import sys
# 修改 Python 模块搜索路径。
import time
# 时间戳生成。
from datetime import datetime
# 获取当前时间，用于生成唯一的运行 ID。
from pathlib import Path
# 面向对象的路径处理。

import h5py
# 读写 HDF5（LIBERO 数据格式）。
import numpy as np
# 数值计算，处理动作、状态数组。
import tqdm
# 进度条库，本脚本的核心可视化手段。

# ==============================================================================
# 仓库路径常量
# ==============================================================================
# ROOT_DIR 指向 openvla 仓库根目录。
# 本文件位于 .../experiments/robot/libero/data_regeneration/ 下，回退 4 层就是仓库根。
ROOT_DIR = Path(__file__).resolve().parents[4]

# LIBERO_DIR 指向仓库内置的 LIBERO 子目录，保证后续能 import libero 包。
LIBERO_DIR = ROOT_DIR / "LIBERO"

# BDDL（任务场景描述文件）固定走当前仓库内置的 libero 目录，不依赖 ~/.libero/config.yaml。
LOCAL_BDDL_ROOT = LIBERO_DIR / "libero" / "libero" / "bddl_files"

# 把仓库根目录和 LIBERO 子目录都放进 sys.path，确保 import 成功。
# 这一步要放在 import libero 之前。
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(LIBERO_DIR))

# 现在环境变量、路径都准备好了，才 import 真正的重库。
from libero.libero import benchmark
# LIBERO 提供的任务套件字典。
from libero.libero.envs.env_wrapper import ControlEnv
# LIBERO 的环境封装，既支持夹爪也支持吸盘。

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    # 获取空动作，用于“重置后等待物理稳定”的空跑阶段。
    save_rollout_video,
    # 可选地把一条 demo 的回放图像序列保存成 mp4。
)


# ==============================================================================
# 默认参数常量
# ==============================================================================
DEFAULT_RESOLUTION = 256
# 默认渲染分辨率（256x256）。
DEFAULT_CONTROL_FREQ = 20
# 控制频率：环境每秒 step 多少次。
DEFAULT_SETTLE_STEPS = 10
# 重置到初始状态后先空跑几步，等待物体掉落稳定。
DEFAULT_OMP_THREADS = 2
# OpenMP 默认线程数（用户实测最优）。
DEFAULT_MKL_THREADS = 1
# Intel MKL 默认线程数（用户实测最优）。

# 本脚本当前固定只跑 libero_suction 这一个任务套件。
#
# 为什么 libero_suction 名字带 suction 却当“夹爪源数据”来用？
#   经实测，它的源 HDF5 数据是用夹爪采集的：
#     1) gripper_states 是 (T, 2)，对应夹爪两个指的 qpos（吸盘只有 1 维开关）；
#     2) gripper_states 取值是连续的 -0.04 ~ -0.03（夹爪指节连续位置），不是吸盘的离散 {-1,+1}；
#     3) 不含 suction_wrench（六维力）字段，而真正的吸盘采集数据必含该字段。
#   所以 libero_suction 正好是“夹爪动作 -> 吸盘环境”这个验证实验的目标场景。
#   判定标准始终是“源数据是否用夹爪采集”，而不是套件名字里有没有 suction。
SUPPORTED_GRIPPER_TASK_SUITES = ["libero_suction"]


# ==============================================================================
# 以下这一组函数（_mujoco_joint_sizes / _build_joint_index /
# _is_robot_arm_joint / _is_end_effector_related_joint /
# convert_default_init_state_to_suction / build_task_bddl_path /
# create_suction_env / create_default_env / prepare_env_at_demo_start）
# 全部【原样复制】自同目录的 collect_libero_suction_handover_dataset.py。
#
# 为什么是复制而不是 import：
# 那个采集脚本顶层会 import realtime_control_and_monitor（依赖 fastapi/uvicorn），
# 而本脚本是纯离线验证脚本，不应该引入 Web 相关依赖。
# 这些函数本身是“夹爪 init_state -> 吸盘 init_state”的纯数值/环境逻辑，
# 与 Web 层完全无关，复制过来使用是合理且零风险的。
# ==============================================================================


def _mujoco_joint_sizes(joint_type: int) -> tuple[int, int]:
    """根据 MuJoCo joint 类型返回该 joint 在 qpos / qvel 里的长度。"""
    if joint_type == 0:
        # 自由关节：7 个 qpos（3 位置 + 4 四元数），6 个 qvel（3 线速度 + 3 角速度）。
        return 7, 6
    if joint_type == 1:
        # 球关节：4 个 qpos（四元数），3 个 qvel（角速度）。
        return 4, 3
    if joint_type in (2, 3):
        # 滑动关节或铰链关节：各 1 个。
        return 1, 1
    raise ValueError(f"Unknown MuJoCo joint type: {joint_type}")


def _build_joint_index(model) -> dict[str, tuple[int, int, int]]:
    """把 joint name 映射成 (qpos 起点, qvel 起点, joint 类型)。"""
    mapping: dict[str, tuple[int, int, int]] = {}
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name
        # 每个 joint 用名字做键，便于两个模型按名字对齐。
        mapping[name] = (
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
            int(model.jnt_type[joint_id]),
        )
    return mapping


def _is_robot_arm_joint(joint_name: str) -> bool:
    """
    判断某个 joint 是否属于 Panda 机械臂本体的 7 个关节。

    只认 robot0_joint1 到 robot0_joint7 这 7 个真正的机械臂关节，
    不用“是否包含 robot0”这种宽松规则，因为 gripper 名字里也可能带 robot0。
    （本脚本当前未直接用到，但与源文件保持一致，方便后续扩展。）
    """
    return joint_name.startswith("robot0_joint") and joint_name[len("robot0_joint") :].isdigit()


def _is_end_effector_related_joint(joint_name: str) -> bool:
    """
    判断某个 joint 是否属于末端执行器相关状态。

    目标是：不迁移 default gripper 的末端状态，也不迁移 suction 自己的末端状态，
    自动播放时 suction 保持关闭，从“关闭 + 静止”起步。
    只要 joint 名里明显带 gripper / finger / suction 这些关键词，就统一视为末端相关，
    在状态转换时直接清零。
    """
    lowered = joint_name.lower()
    return ("gripper" in lowered) or ("finger" in lowered) or ("suction" in lowered)


def convert_default_init_state_to_suction(*, init_state_default, env_default: ControlEnv, env_suction: ControlEnv) -> np.ndarray:
    """
    把“默认夹爪环境”的 init_state 转成“吸盘环境”可接受的 init_state。

    核心思想（有选择地迁移）：
    1. Panda 机械臂本体：迁移 7 个机械臂关节的 qpos 和 qvel（起始位姿 + 速度对齐原始 demo）。
    2. 末端执行器：不迁移 default gripper 末端状态；suction joint 的 qpos/qvel 强制置 0；
       其他 suction-only 内部状态保留 suction reset 默认值。
    3. 环境物体：物体 joint 的 qpos/qvel 都尽量保留，最大程度复现原始场景布局。

    注意：这里不做 xyzRxRyRz 笛卡尔位姿迁移。init_state 本质是 MuJoCo 的 [time, qpos, qvel]。
    对同一条 Panda 机械臂，直接迁移 7 个机械臂关节 qpos，就已经等价于把末端位姿迁过去了。
    """
    if hasattr(init_state_default, "detach"):
        # 兼容 torch tensor 输入。
        init_state_default = init_state_default.detach().cpu().numpy()
    state_d = np.asarray(init_state_default, dtype=np.float64).reshape(-1)

    model_d = env_default.sim.model
    model_s = env_suction.sim.model
    expected_len_d = 1 + int(model_d.nq) + int(model_d.nv)
    expected_len_s = 1 + int(model_s.nq) + int(model_s.nv)

    if len(state_d) != expected_len_d:
        raise ValueError(f"Default init_state length mismatch: got {len(state_d)}, expected {expected_len_d}")

    time_value = float(state_d[0])
    qpos_d = state_d[1 : 1 + int(model_d.nq)]
    qvel_d = state_d[1 + int(model_d.nq) :]

    # suction 环境当前 qpos 作为“未被原始状态覆盖部分”的默认值来源。
    qpos_s = np.asarray(env_suction.sim.data.qpos, dtype=np.float64).copy()
    # qvel 同理从 suction reset 默认值开始，再按同名 joint 对齐规则覆盖。
    qvel_s = np.asarray(env_suction.sim.data.qvel, dtype=np.float64).copy()

    joints_d = _build_joint_index(model_d)

    for joint_id in range(model_s.njnt):
        joint_name = model_s.joint(joint_id).name
        qposadr_s = int(model_s.jnt_qposadr[joint_id])
        dofadr_s = int(model_s.jnt_dofadr[joint_id])
        joint_type_s = int(model_s.jnt_type[joint_id])
        qpos_sz_s, qvel_sz_s = _mujoco_joint_sizes(joint_type_s)

        # 末端执行器相关 joint 一律不迁移，直接清零，保证吸盘从“关闭+静止”起步。
        if _is_end_effector_related_joint(joint_name):
            qpos_s[qposadr_s : qposadr_s + qpos_sz_s] = 0.0
            qvel_s[dofadr_s : dofadr_s + qvel_sz_s] = 0.0
            continue

        # 只存在于 suction 模型、default 模型里没有的 joint，直接跳过，保留 suction 默认值。
        if joint_name not in joints_d:
            continue

        qposadr_d, dofadr_d, joint_type_d = joints_d[joint_name]
        qpos_sz_d, qvel_sz_d = _mujoco_joint_sizes(joint_type_d)

        if (qpos_sz_d, qvel_sz_d) != (qpos_sz_s, qvel_sz_s):
            raise ValueError(
                f"Joint size mismatch for {joint_name}: default {(qpos_sz_d, qvel_sz_d)} vs suction {(qpos_sz_s, qvel_sz_s)}"
            )

        # 同名 joint 直接迁移 qpos / qvel：
        # 这会保留机械臂 7 轴起始位姿/速度 + 场景物体起始位姿/速度。
        qpos_s[qposadr_s : qposadr_s + qpos_sz_s] = qpos_d[qposadr_d : qposadr_d + qpos_sz_d]
        qvel_s[dofadr_s : dofadr_s + qvel_sz_s] = qvel_d[dofadr_d : dofadr_d + qvel_sz_d]

    state_s = np.concatenate(([time_value], qpos_s, qvel_s), axis=0)
    if len(state_s) != expected_len_s:
        raise RuntimeError(f"Converted suction state length mismatch: got {len(state_s)}, expected {expected_len_s}")
    return state_s


def build_task_bddl_path(task) -> str:
    """根据 task 元数据解析出它的 BDDL 路径。"""
    return str(LOCAL_BDDL_ROOT / task.problem_folder / task.bddl_file)


def create_suction_env(task, resolution: int, control_freq: int) -> ControlEnv:
    """
    创建吸盘环境。本脚本真正用来回放夹爪动作、判定成功的就是它。

    ignore_done=True：忽略底层 robosuite 的 horizon done，由外层循环自己控制结束条件。
    （与采集脚本保持一致，避免“在已终止 episode 里 step”报错。）
    """
    return ControlEnv(
        bddl_file_name=build_task_bddl_path(task),
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="suction",
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=control_freq,
        ignore_done=True,
    )


def create_default_env(task, resolution: int, control_freq: int) -> ControlEnv:
    """
    创建默认夹爪环境。本脚本里它【只用于一件事】：
    提供 convert_default_init_state_to_suction 所需的 model_d（夹爪模型几何信息）。
    本身不参与任何动作回放。
    """
    return ControlEnv(
        bddl_file_name=build_task_bddl_path(task),
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=control_freq,
        ignore_done=True,
    )


def prepare_env_at_demo_start(env: ControlEnv, init_state, settle_steps: int) -> tuple[dict, dict]:
    """
    把某个环境重置到指定 demo 的起始状态，并等环境稳定几步。

    返回稳定后的 obs 和 info。
    """
    env.reset()
    obs = env.set_init_state(init_state)
    info: dict = {}
    for _ in range(settle_steps):
        # 用 dummy action 空跑几步，等物理（主要是物体掉落）稳定。
        obs, _, _, info = env.step(get_libero_dummy_action("openvla"))
    return obs, info


# ==============================================================================
# 以下是本脚本自己的逻辑（不复用采集脚本）
# ==============================================================================


def replay_single_demo_with_gripper_actions(
    *,
    env_suction: ControlEnv,
    orig_actions: np.ndarray,
    orig_states: np.ndarray,
    env_default: ControlEnv,
    settle_steps: int,
    save_video: bool,
) -> dict:
    """
    在吸盘环境里，把【原始夹爪动作】原样回放一遍，并返回本条 demo 的统计结果。

    【这是整个验证实验的核心】
    关键点：
    1. 原始动作是夹爪环境的 7 维 action，最后一维是夹爪开合[-1,+1]。
    2. 这里【不做任何修改】，包括最后一维也不动，直接喂给吸盘环境。
    3. 夹爪的“闭合(+1)”会被吸盘解释为“吸盘开启”，这正是我们要测的语义兼容性。
    4. 成功判定完全用吸盘环境自身的 check_success。

    返回字典字段：
    - steps_total：原始动作总步数
    - steps_executed：实际执行了多少步（可能因 done 提前结束）
    - success：吸盘环境是否判定成功
    - early_done：是否因为环境报 done 而提前结束
    - error：若中途抛异常，记录异常信息字符串；正常时为 None
    - replay_images：若 save_video=True，保存回放帧序列；否则为空列表
    """
    result = {
        "steps_total": int(len(orig_actions)),
        "steps_executed": 0,
        "success": False,
        "early_done": False,
        "error": None,
        "replay_images": [] if save_video else None,
    }

    try:
        # 第一步：把吸盘环境重置到与原始夹爪 demo 一致的起始状态。
        # 注意：初始状态做了“夹爪->吸盘”的有选择转换，只让“动作”成为唯一变量。
        converted_state = convert_default_init_state_to_suction(
            init_state_default=orig_states[0],
            env_default=env_default,
            env_suction=env_suction,
        )
        obs, _ = prepare_env_at_demo_start(env_suction, converted_state, settle_steps)

        # 第二步：逐帧把原始夹爪动作喂给吸盘环境。
        done = False
        for step_index, action in enumerate(orig_actions):
            # 【关键】action 原样直传，不修改任何一维。
            # 这与采集脚本里“action[-1] = -1.0”的半自动做法形成对照。
            action_arr = np.asarray(action, dtype=np.float64).copy()

            # 如果需要保存视频，就先把当前观测画面存下来（在 step 之前的画面）。
            # 注意：mujoco/robosuite 的 agentview_image 默认是上下颠倒、左右也镜像的，
            # 直接存成视频看起来会是“倒着的”。这里做一次 180° 旋转（行、列同时反转），
            # 处理方式与 run_libero_eval.py 的 get_libero_image 里 img[::-1, ::-1] 完全一致，
            # 保证保存出来的视频方向符合人的观察习惯。
            if save_video:
                raw_frame = np.asarray(obs["agentview_image"])
                result["replay_images"].append(raw_frame[::-1, ::-1].copy())

            obs, reward, done, info = env_suction.step(action_arr.tolist())
            # 真正把动作送入吸盘环境执行一步。
            result["steps_executed"] = step_index + 1

            if done:
                # 环境自己报 done（任务完成或其它终止条件），提前结束。
                result["early_done"] = True
                break

        # 第三步：用吸盘环境自身的成功条件判定。
        # done 为 True 通常意味着任务成功；但也有些情况是 horizon 结束，
        # 所以再补一个 check_success() 兜底。
        success = bool(done or env_suction.check_success())
        result["success"] = success

    except Exception as exc:
        # 单条 demo 抛异常时，记录为失败并继续，不中断整体统计。
        # 这样某条数据特别异常也不会让整个评测白跑。
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["success"] = False

    return result


def build_default_output_json_path(task_suite_name: str) -> Path:
    """构造默认的输出 JSON 路径：./experiments/logs/verify_g2s_{suite}_{时间戳}.json。"""
    log_dir = ROOT_DIR / "experiments" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"verify_g2s_{task_suite_name}_{timestamp}.json"


def save_results_json(json_path: Path, payload: dict):
    """把评测结果字典落盘成 JSON（中文不转义）。"""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        # ensure_ascii=False 让中文 task 描述直接以中文写入，而不是 \uXXXX。


def print_final_summary(payload: dict):
    """在控制台打印最终汇总表，方便一眼看到每个 task 的成功率。"""
    print("\n" + "=" * 80)
    print(f"验证结果汇总：夹爪动作 -> 吸盘环境  (任务套件: {payload['task_suite']})")
    print("=" * 80)
    # 表头。
    header = f"{'task_id':>7}  {'episodes':>8}  {'successes':>9}  {'success_rate':>12}  task_description"
    print(header)
    print("-" * len(header))
    # 每个 task 一行。
    for task_entry in payload["tasks"]:
        rate = task_entry["success_rate"]
        print(
            f"{task_entry['task_id']:>7}  "
            f"{task_entry['episodes']:>8}  "
            f"{task_entry['successes']:>9}  "
            f"{rate * 100:>11.1f}%  "
            f"{task_entry['task_description']}"
        )
    print("-" * len(header))
    summary = payload["summary"]
    print(
        f"{'TOTAL':>7}  "
        f"{summary['total_demos']:>8}  "
        f"{summary['total_successes']:>9}  "
        f"{summary['success_rate'] * 100:>11.1f}%"
    )
    print("=" * 80)


def run_verification(args) -> dict:
    """
    主流程：遍历选定套件的所有 task、所有 demo，把夹爪动作原样喂给吸盘环境。

    返回完整的结构化结果字典，同时也会在内部把进度条、控制台统计打出来。
    """
    # 1) 准备任务套件。
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.libero_task_suite not in benchmark_dict:
        raise ValueError(
            f"未知的 task suite: {args.libero_task_suite}，可选: {list(benchmark_dict.keys())}"
        )
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks
    if args.task_limit is not None:
        num_tasks_in_suite = min(num_tasks_in_suite, args.task_limit)
        # 只处理前 N 个 task（快速试跑用）。

    # 原始夹爪 HDF5 目录：如果用户没传，就用仓库内置的默认目录自动推导。
    raw_data_dir = Path(args.libero_raw_data_dir)

    print("=" * 80)
    print("开始验证：夹爪动作 -> 吸盘环境")
    print(f"任务套件      : {args.libero_task_suite}")
    print(f"原始数据目录  : {raw_data_dir}")
    print(f"task 数量     : {num_tasks_in_suite}（共 {task_suite.n_tasks}）")
    print(f"每 task demo  : {'全部' if args.demo_limit_per_task is None else args.demo_limit_per_task}")
    print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS')}")
    print(f"分辨率 / 控制频率 / settle步数 : {args.resolution} / {args.control_freq} / {args.settle_steps}")
    print(f"保存视频      : {args.save_rollout_video}")
    print("=" * 80)

    # 2) 准备结果容器。
    tasks_result: list[dict] = []
    total_episodes = 0
    total_successes = 0

    # 3) 外层进度条：遍历每个 task。
    task_pbar = tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks", position=0, dynamic_ncols=True)
    for task_id in task_pbar:
        task = task_suite.get_task(task_id)
        task_description = task.language
        orig_data_path = raw_data_dir / f"{task.name}_demo.hdf5"

        if not orig_data_path.exists():
            # 如果原始数据文件不存在，记录为空 task 并跳过，不让整个脚本崩。
            print(f"[警告] 找不到原始数据文件: {orig_data_path}，跳过该 task。")
            tasks_result.append(
                {
                    "task_id": task_id,
                    "task_name": task.name,
                    "task_description": task_description,
                    "demos": [],
                    "episodes": 0,
                    "successes": 0,
                    "success_rate": 0.0,
                    "error": f"原始数据文件不存在: {orig_data_path}",
                }
            )
            continue

        # 打开原始夹爪 HDF5（只读）。
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]
        raw_demo_count = len(orig_data.keys())
        num_demos_to_run = raw_demo_count if args.demo_limit_per_task is None else min(raw_demo_count, args.demo_limit_per_task)

        # 每个 task 创建独立的 suction/default 环境，避免状态泄漏。
        env_suction = create_suction_env(task, args.resolution, args.control_freq)
        env_default = create_default_env(task, args.resolution, args.control_freq)
        env_suction.seed(0)
        env_default.seed(0)

        task_episodes = 0
        task_successes = 0
        task_demos_result: list[dict] = []

        try:
            # 内层进度条：遍历该 task 的每条 demo。
            demo_pbar = tqdm.tqdm(
                range(num_demos_to_run),
                desc=f"  task{task_id}",
                position=1,
                leave=False,
                dynamic_ncols=True,
            )
            for demo_index in demo_pbar:
                demo_data = orig_data[f"demo_{demo_index}"]
                orig_actions = demo_data["actions"][()]
                orig_states = demo_data["states"][()]

                # 核心调用：把这条夹爪 demo 的动作原样喂给吸盘环境回放。
                replay_result = replay_single_demo_with_gripper_actions(
                    env_suction=env_suction,
                    orig_actions=orig_actions,
                    orig_states=orig_states,
                    env_default=env_default,
                    settle_steps=args.settle_steps,
                    save_video=args.save_rollout_video,
                )

                success = bool(replay_result["success"])
                task_episodes += 1
                total_episodes += 1
                if success:
                    task_successes += 1
                    total_successes += 1

                # 可选：保存本条 demo 的回放视频。
                if args.save_rollout_video and replay_result["replay_images"]:
                    # 视频按“全局 episode 编号 + success + task”命名，落到子目录。
                    save_rollout_video(
                        replay_result["replay_images"],
                        idx=total_episodes,
                        success=success,
                        task_description=task_description,
                        log_file=None,
                        subdir=f"gripper_to_suction/{args.libero_task_suite}",
                        root_dir=args.video_root_dir,
                    )

                # 进度条 postfix 实时显示本 task / 全局 成功率。
                task_rate = (task_successes / task_episodes) if task_episodes else 0.0
                global_rate = (total_successes / total_episodes) if total_episodes else 0.0
                demo_pbar.set_postfix(
                    demo=f"demo_{demo_index}",
                    ok=replay_result["success"],
                    steps=f"{replay_result['steps_executed']}/{replay_result['steps_total']}",
                    task_rate=f"{task_rate * 100:.0f}%",
                    global_rate=f"{global_rate * 100:.0f}%",
                )

                # 单条 demo 的结构化记录。
                task_demos_result.append(
                    {
                        "demo_index": int(demo_index),
                        "steps_total": int(replay_result["steps_total"]),
                        "steps_executed": int(replay_result["steps_executed"]),
                        "success": bool(replay_result["success"]),
                        "early_done": bool(replay_result["early_done"]),
                        "error": replay_result["error"],
                    }
                )

            demo_pbar.close()
        finally:
            # 无论正常还是异常，都把环境和文件句柄关掉，避免资源泄漏。
            env_default.close()
            env_suction.close()
            orig_data_file.close()

        task_rate = (task_successes / task_episodes) if task_episodes else 0.0
        tasks_result.append(
            {
                "task_id": int(task_id),
                "task_name": task.name,
                "task_description": task_description,
                "demos": task_demos_result,
                "episodes": int(task_episodes),
                "successes": int(task_successes),
                "success_rate": float(task_rate),
            }
        )

        # 外层进度条也带上当前全局成功率，方便边跑边看。
        global_rate = (total_successes / total_episodes) if total_episodes else 0.0
        task_pbar.set_postfix(
            task_rate=f"{task_rate * 100:.0f}%",
            global_rate=f"{global_rate * 100:.0f}%",
            done=f"{total_episodes}",
        )

    # 4) 汇总。
    overall_rate = (total_successes / total_episodes) if total_episodes else 0.0
    payload = {
        "task_suite": args.libero_task_suite,
        "config": {
            "libero_raw_data_dir": str(raw_data_dir),
            "task_limit": args.task_limit,
            "demo_limit_per_task": args.demo_limit_per_task,
            "resolution": args.resolution,
            "control_freq": args.control_freq,
            "settle_steps": args.settle_steps,
            "omp_threads": int(args.omp_threads),
            "mkl_threads": int(args.mkl_threads),
            "save_rollout_video": bool(args.save_rollout_video),
        },
        "summary": {
            "total_demos": int(total_episodes),
            "total_successes": int(total_successes),
            "success_rate": float(overall_rate),
        },
        "tasks": tasks_result,
    }
    return payload


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="验证：把原始夹爪动作原样喂给吸盘环境，统计成功率，测试两者运动空间是否相同。"
    )
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        default="libero_suction",
        choices=SUPPORTED_GRIPPER_TASK_SUITES,
        help="要测试的 LIBERO 任务套件名。默认就是 libero_suction，直接跑即可，无需手动指定。",
    )
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        default=None,
        help="原始夹爪 HDF5 数据集目录。不传则自动用 ./LIBERO/libero/datasets/libero_suction。",
    )
    parser.add_argument("--task_limit", type=int, default=None, help="只跑前 N 个 task（快速试跑）。不传则跑全部。")
    parser.add_argument(
        "--demo_limit_per_task",
        type=int,
        default=None,
        help="每个 task 只跑前 N 条 demo（快速试跑）。不传则跑全部。",
    )
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="渲染分辨率。")
    parser.add_argument("--control_freq", type=int, default=DEFAULT_CONTROL_FREQ, help="环境控制频率。")
    parser.add_argument("--settle_steps", type=int, default=DEFAULT_SETTLE_STEPS, help="初始化后等待物理稳定的步数。")
    parser.add_argument(
        "--omp_threads",
        type=int,
        default=DEFAULT_OMP_THREADS,
        help="OMP_NUM_THREADS。默认 2（用户实测：一张 A100 挂 4 个评测时最优）。",
    )
    parser.add_argument(
        "--mkl_threads",
        type=int,
        default=DEFAULT_MKL_THREADS,
        help="MKL_NUM_THREADS。默认 1（用户实测最优）。",
    )
    parser.add_argument(
        "--save_rollout_video",
        type=lambda s: str(s).lower() not in ("0", "false", "no", "off"),
        default=True,
        help="是否保存每条 demo 的回放视频（成功/失败分文件名区分）。默认开启；传 --save_rollout_video false 可关闭。",
    )
    parser.add_argument(
        "--video_root_dir",
        type=str,
        default="./rollouts/gripper_to_suction",
        help="回放视频保存根目录。",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="结果 JSON 输出路径。不传则自动生成到 ./experiments/logs/ 下。",
    )
    args = parser.parse_args()

    # 如果用户没传原始数据目录，就用仓库内置默认目录自动推导。
    if args.libero_raw_data_dir is None:
        args.libero_raw_data_dir = str(LIBERO_DIR / "libero" / "datasets" / args.libero_task_suite)

    # 如果用户没传输出 JSON 路径，就自动生成一个带时间戳的路径。
    if args.output_json is None:
        args.output_json = str(build_default_output_json_path(args.libero_task_suite))

    return args


def main():
    """脚本入口。"""
    args = parse_args()

    t_start = time.time()
    payload = run_verification(args)
    elapsed = time.time() - t_start

    payload["summary"]["elapsed_seconds"] = float(elapsed)

    # 落盘 JSON 结果。
    output_json_path = Path(args.output_json)
    save_results_json(output_json_path, payload)

    # 控制台打印最终汇总表。
    print_final_summary(payload)
    print(f"结果已保存到: {output_json_path}")
    print(f"总耗时: {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
