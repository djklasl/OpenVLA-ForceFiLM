"""
regenerate_suction_libero_dataset.py

【脚本定位】
这个脚本是“吸盘版 LIBERO 数据再生成流程”的第二阶段。
它读取第一阶段 collect_libero_suction_handover_dataset.py 产生的“原始采集 HDF5”，
然后像官方 regenerate_libero_dataset.py 一样，在环境里重新回放这些轨迹并重新采样观测，
同时执行两件关键清洗工作：

1. 删除 no-op 动作；
2. 在轨迹尾部显式补“释放 + 等待下落”阶段，然后删除最终失败的整条轨迹。

【和官方脚本的关系】
整体结构尽量贴近官方 regenerate_libero_dataset.py：
- 逐 task 处理
- 逐 demo 读取
- 在环境里 replay
- 重新采样 obs / states / images
- 最终只保留成功轨迹

不同点只在于：
- 官方脚本的输入是原始默认夹爪 LIBERO demo；
- 这里的输入是第一阶段已经在吸盘环境里真实执行过的原始采集 demo。

【字段收敛说明】
为了和第一阶段输出保持一致，这里也沿用同一套 schema：
- obs/gripper_states 表示 suction gripper 的 robot0_gripper_qpos，shape 为 (T, 1)；
- obs/suction_wrench 保留；
- obs/suction_on、obs/suction_latched、control_modes 都不再写入。

【输入输出约束】
- 这个脚本只读取第一阶段原始采集目录，不会对原始 HDF5 做任何修改；
- 清洗后的成功轨迹会写入 data_regeneration/processed_{task_suite}；
- metainfo 至少记录每条 demo 的 success 和 initial_state，并允许附加更多阶段 2 信息。

【运行方式】
在仓库根目录下可直接运行，例如：

1. 标准运行（处理整个 libero_suction，输出到默认 processed 目录）
   python experiments/robot/libero/data_regeneration/regenerate_suction_libero_dataset.py --libero_task_suite libero_suction --libero_raw_data_dir ./data_regeneration/libero_suction

2. 显式指定输出目录
   python experiments/robot/libero/data_regeneration/regenerate_suction_libero_dataset.py --libero_task_suite libero_suction --libero_raw_data_dir ./data_regeneration/libero_suction --libero_target_dir ./data_regeneration/processed_libero_suction

3. 调试运行（只先处理 1 个 task）
   python experiments/robot/libero/data_regeneration/regenerate_suction_libero_dataset.py --libero_task_suite libero_suction --libero_raw_data_dir ./data_regeneration/libero_suction --task_limit 1

说明：
- 若不显式传 --libero_target_dir，默认输出到 data_regeneration/processed_{task_suite}；
 - 原始采集目录（例如 data_regeneration/libero_suction）中的 HDF5 不会被修改。

【为什么这里不做 WebSocket】
第二阶段本质上是离线清洗脚本，不存在人工接管，也不追求交互式低延迟控制。
因此它故意不接入 WebSocket 或浏览器控制链路，只保留离线回放 + 清洗职责。
"""

# ==============================================================================
# 1. 标准库导入
# ==============================================================================
import argparse
import json
import os
import sys
from pathlib import Path

# ==============================================================================
# 2. 第三方库导入
# ==============================================================================
import h5py
import numpy as np
import tqdm

# ==============================================================================
# 3. 路径与环境变量
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parents[4]
LIBERO_DIR = ROOT_DIR / "LIBERO"
# 阶段 2 清洗脚本里的 BDDL 也固定走当前仓库内置的 LIBERO 目录，避免受到 ~/.libero/config.yaml 干扰。
LOCAL_BDDL_ROOT = LIBERO_DIR / "libero" / "libero" / "bddl_files"

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(LIBERO_DIR))

# ==============================================================================
# 4. 项目内部导入
# ==============================================================================
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from experiments.robot.libero.libero_utils import get_libero_dummy_action, quat2axisangle


# ==============================================================================
# 5. 常量
# ==============================================================================
DEFAULT_RESOLUTION = 256
DEFAULT_CONTROL_FREQ = 20
DEFAULT_DUMMY_STEPS = 10
DEFAULT_TAIL_DUMMY_STEPS = 20


# ==============================================================================
# 6. 核心辅助函数
# ==============================================================================
def is_noop(action, prev_action=None, threshold=1e-4):
    """
    判断当前动作是否属于 no-op。

    规则沿用官方 regenerate_libero_dataset.py：
    - 除最后一维之外，其余动作幅值几乎为 0；
    - 且最后一维（这里是 suction 开关语义）与上一步相同。

    这样可以避免把“机械臂静止但吸盘开关发生变化”的关键步骤误删掉。
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


def build_tail_dummy_action() -> np.ndarray:
    """构造轨迹尾部的等待动作：机械臂静止，吸盘保持关闭。"""
    tail_action = np.zeros(7, dtype=np.float64)
    tail_action[-1] = -1.0
    return tail_action


def build_replay_actions(raw_actions: np.ndarray, tail_dummy_steps: int) -> tuple[list[np.ndarray], int]:
    """
    基于原始动作构造阶段 2 真正要 replay 的动作序列。

    流程：
    1. 先按现有规则删除 no-op；
    2. 强制最后一个保留动作的吸盘维度为 -1；
    3. 在末尾追加若干步静止且吸盘关闭的 dummy action，
       给物体留出从吸盘脱离并落入篮子的物理时间。

    返回：
    - replay_actions: 最终用于 replay 的动作序列
    - num_noops_filtered: 被过滤掉的 no-op 数量
    """
    filtered_actions: list[np.ndarray] = []
    num_noops_filtered = 0

    for action in raw_actions:
        action_arr = np.asarray(action, dtype=np.float64).copy()
        prev_action = filtered_actions[-1] if filtered_actions else None
        if is_noop(action_arr, prev_action):
            num_noops_filtered += 1
            continue
        filtered_actions.append(action_arr)

    tail_release = build_tail_dummy_action()
    if filtered_actions:
        filtered_actions[-1] = filtered_actions[-1].copy()
        filtered_actions[-1][-1] = -1.0
    else:
        filtered_actions.append(tail_release.copy())

    replay_actions = [action.copy() for action in filtered_actions]
    for _ in range(tail_dummy_steps):
        replay_actions.append(tail_release.copy())

    return replay_actions, num_noops_filtered


def build_task_bddl_path(task) -> str:
    """根据 benchmark task 解析出真实 BDDL 文件路径。"""
    return str(LOCAL_BDDL_ROOT / task.problem_folder / task.bddl_file)


def create_suction_env(task, resolution: int, control_freq: int) -> ControlEnv:
    """创建阶段 2 回放用的吸盘环境。"""
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
        # 阶段 2 这里也和阶段 1 保持一致，显式忽略底层 robosuite 的 horizon done。
        # 原因是当前 replay / 清洗流程已经由外层脚本自己决定什么时候结束一条 demo：
        # 1. replay 动作放完；
        # 2. 任务成功；
        # 3. 尾部 dummy release 段执行完。
        # 如果这里保持 False，当轨迹步数接近默认 horizon 时，底层环境可能先被判定为 episode 结束，
        # 下一次 step 就会抛出 “executing action in terminated episode”。
        ignore_done=True,
    )


def extract_suction_wrench(obs: dict, info: dict | None = None) -> np.ndarray:
    """统一提取 suction_wrench；如果缺失则退回 info 或全 0。"""
    if "suction_wrench" in obs and obs["suction_wrench"] is not None:
        return np.asarray(obs["suction_wrench"], dtype=np.float64)
    if info is not None and "suction_force_torque" in info and info["suction_force_torque"] is not None:
        return np.asarray(info["suction_force_torque"], dtype=np.float64)
    return np.zeros(6, dtype=np.float64)


def extract_robot_state(obs: dict) -> np.ndarray:
    """提取紧凑版 robot_state。"""
    return np.concatenate([np.atleast_1d(obs["robot0_gripper_qpos"]), obs["robot0_eef_pos"], obs["robot0_eef_quat"]], axis=0)


def default_processed_dir_for_suite(task_suite_name: str) -> Path:
    """阶段 2 默认输出目录：data_regeneration/processed_{task_suite}。"""
    return ROOT_DIR / "data_regeneration" / f"processed_{task_suite_name}"


def resolve_checked_target_dir(raw_data_dir: Path, target_dir: Path | None, task_suite_name: str) -> Path:
    """
    解析并校验阶段 2 输出目录。

    约束：
    - 若未显式指定，则默认写到 data_regeneration/processed_{task_suite}；
    - 目标目录不能是任何原始采集目录本身；
    - 目标目录不能位于任何原始采集目录内部；
    - 目标目录必须位于 data_regeneration 根目录下。
    """
    raw_dir = raw_data_dir.resolve()
    data_regen_root = (ROOT_DIR / "data_regeneration").resolve()
    default_target = default_processed_dir_for_suite(task_suite_name)
    resolved_target = (target_dir if target_dir is not None else default_target).resolve()

    raw_collection_dirs = {raw_dir}
    for metainfo_path in data_regen_root.glob('*/*_raw_collection_metainfo.json'):
        raw_collection_dirs.add(metainfo_path.parent.resolve())

    for candidate_raw_dir in raw_collection_dirs:
        if resolved_target == candidate_raw_dir:
            raise ValueError(f"目标目录不能与原始采集目录相同：{resolved_target}")
        if candidate_raw_dir in resolved_target.parents:
            raise ValueError(f"目标目录不能位于原始采集目录内部：{resolved_target}")

    if data_regen_root not in resolved_target.parents and resolved_target != data_regen_root:
        raise ValueError(f"阶段 2 目标目录必须位于 {data_regen_root} 下方，当前给定为：{resolved_target}")

    return resolved_target


def confirm_overwrite_nonempty_dir(target_dir: Path):
    """若目标目录已存在且非空，则提示用户确认覆盖。"""
    if not target_dir.exists() or not any(target_dir.iterdir()):
        return

    user_input = input(
        f"Target directory already exists and is not empty: {target_dir}\n"
        "Enter 'y' to overwrite the directory contents, or anything else to exit: "
    )
    if user_input != "y":
        raise SystemExit(0)


def init_episode_buffers() -> dict:
    """初始化一条清洗后 demo 的时间序列缓存。"""
    return {
        "states": [],
        "actions": [],
        "robot_states": [],
        "gripper_states": [],
        "joint_states": [],
        "ee_states": [],
        "agentview_images": [],
        "eye_in_hand_images": [],
        "suction_wrench": [],
    }


def record_observation(buffers: dict, obs: dict, info: dict | None = None):
    """
    把当前时刻的观测写进缓存。

    注意：
    - obs/gripper_states 在阶段 2 中继续表示 suction gripper 的 robot0_gripper_qpos；
    - suction_on 不写入最终 HDF5，只保留为运行时信息。
    """
    buffers["gripper_states"].append(np.atleast_1d(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)).copy())
    buffers["joint_states"].append(np.asarray(obs["robot0_joint_pos"], dtype=np.float64).copy())
    buffers["ee_states"].append(
        np.hstack((np.asarray(obs["robot0_eef_pos"], dtype=np.float64), quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))))
    )
    buffers["agentview_images"].append(np.asarray(obs["agentview_image"]).copy())
    buffers["eye_in_hand_images"].append(np.asarray(obs["robot0_eye_in_hand_image"]).copy())
    buffers["suction_wrench"].append(extract_suction_wrench(obs, info).copy())


def save_episode_to_hdf5(grp, demo_name: str, buffers: dict):
    """把一条清洗后的成功 demo 写入最终 HDF5。"""
    ep_data_grp = grp.create_group(demo_name)
    obs_grp = ep_data_grp.create_group("obs")

    obs_grp.create_dataset("gripper_states", data=np.stack(buffers["gripper_states"], axis=0))
    obs_grp.create_dataset("joint_states", data=np.stack(buffers["joint_states"], axis=0))
    obs_grp.create_dataset("ee_states", data=np.stack(buffers["ee_states"], axis=0))
    obs_grp.create_dataset("ee_pos", data=np.stack(buffers["ee_states"], axis=0)[:, :3])
    obs_grp.create_dataset("ee_ori", data=np.stack(buffers["ee_states"], axis=0)[:, 3:])
    obs_grp.create_dataset("agentview_rgb", data=np.stack(buffers["agentview_images"], axis=0))
    obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(buffers["eye_in_hand_images"], axis=0))
    obs_grp.create_dataset("suction_wrench", data=np.stack(buffers["suction_wrench"], axis=0))

    dones = np.zeros(len(buffers["actions"]), dtype=np.uint8)
    rewards = np.zeros(len(buffers["actions"]), dtype=np.uint8)
    dones[-1] = 1
    rewards[-1] = 1

    ep_data_grp.create_dataset("actions", data=np.stack(buffers["actions"], axis=0))
    ep_data_grp.create_dataset("states", data=np.stack(buffers["states"], axis=0))
    ep_data_grp.create_dataset("robot_states", data=np.stack(buffers["robot_states"], axis=0))
    ep_data_grp.create_dataset("rewards", data=rewards)
    ep_data_grp.create_dataset("dones", data=dones)


# ==============================================================================
# 7. 主流程
# ==============================================================================
def main(args):
    """第二阶段主流程：回放阶段 1 原始 HDF5，删除 no-op，并且只保留成功轨迹。"""
    raw_data_dir = Path(args.libero_raw_data_dir).resolve()
    target_dir = resolve_checked_target_dir(
        raw_data_dir,
        Path(args.libero_target_dir) if args.libero_target_dir is not None else None,
        args.libero_task_suite,
    )
    confirm_overwrite_nonempty_dir(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    metainfo_json_out_path = os.path.join(target_dir, f"{args.libero_task_suite}_metainfo.json")
    metainfo_json_dict = {}

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    if args.task_limit is not None:
        num_tasks_in_suite = min(num_tasks_in_suite, args.task_limit)

    num_replays = 0
    num_success = 0
    num_noops = 0

    task_pbar = tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks", dynamic_ncols=True)
    for task_id in task_pbar:
        task = task_suite.get_task(task_id)
        task_description = task.language
        task_pbar.set_description(f"Tasks [{task_id + 1}/{num_tasks_in_suite}] {task.name}")

        raw_data_path = os.path.join(raw_data_dir, f"{task.name}_demo.hdf5")
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Cannot find stage-1 raw data file: {raw_data_path}")

        new_data_path = os.path.join(target_dir, f"{task.name}_demo.hdf5")

        raw_data_file = h5py.File(raw_data_path, "r")
        raw_data = raw_data_file["data"]
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        env = create_suction_env(task, args.resolution, args.control_freq)
        env.seed(0)
        env.reset()

        task_replays = 0
        task_success = 0
        task_noops = 0
        num_demos = len(raw_data.keys())
        with tqdm.tqdm(total=num_demos, desc=f"Demos ({task.name})", leave=False, dynamic_ncols=True) as demo_pbar:
            for demo_index in range(num_demos):
                demo_data = raw_data[f"demo_{demo_index}"]
                raw_actions = demo_data["actions"][()]
                raw_states = demo_data["states"][()]
                raw_robot_states = demo_data["robot_states"][()]

                env.reset()
                obs = env.set_init_state(raw_states[0])
                info = {}
                dummy_success = True
                for _ in range(DEFAULT_DUMMY_STEPS):
                    obs, _, done, info = env.step(get_libero_dummy_action("openvla"))
                    if done:
                        dummy_success = False
                        break

                buffers = init_episode_buffers()
                done = False
                replay_actions, demo_noops = build_replay_actions(raw_actions, args.tail_dummy_steps)

                if dummy_success:
                    num_noops += demo_noops
                    task_noops += demo_noops
                    for action in replay_actions:
                        if len(buffers["states"]) == 0:
                            buffers["states"].append(raw_states[0].copy())
                            buffers["robot_states"].append(raw_robot_states[0].copy())
                        else:
                            buffers["states"].append(env.sim.get_state().flatten().copy())
                            buffers["robot_states"].append(extract_robot_state(obs))

                        record_observation(buffers, obs, info)
                        buffers["actions"].append(np.asarray(action, dtype=np.float64).copy())

                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            break

                success = bool(dummy_success and (done or env.check_success()))

                if success and len(buffers["actions"]) > 0:
                    save_episode_to_hdf5(grp, f"demo_{demo_index}", buffers)
                    num_success += 1
                    task_success += 1

                num_replays += 1
                task_replays += 1

                task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{demo_index}"
                metainfo_json_dict.setdefault(task_key, {})
                metainfo_json_dict[task_key][episode_key] = {
                    "success": bool(success),
                    "initial_state": raw_states[0].tolist(),
                    "source_demo_index": int(demo_index),
                    "raw_steps": int(len(raw_actions)),
                    "kept_steps": int(max(len(replay_actions) - args.tail_dummy_steps, 0)),
                    "final_replay_steps": int(len(replay_actions)),
                    "num_noops_filtered": int(demo_noops),
                    "forced_release_applied": True,
                    "tail_dummy_steps_appended": int(args.tail_dummy_steps),
                }

                with open(metainfo_json_out_path, "w", encoding="utf-8") as f:
                    json.dump(metainfo_json_dict, f, indent=2, ensure_ascii=False)

                demo_pbar.update(1)
                demo_pbar.set_postfix(demo=demo_index, success=task_success, kept=task_success, noops=task_noops)
                task_pbar.set_postfix(global_success=num_success, global_noops=num_noops)

        env.close()
        raw_data_file.close()
        new_data_file.close()
        tqdm.tqdm.write(
            f"Task done: {task.name}, replays={task_replays}, kept={task_success}, noops={task_noops}, out={new_data_path}"
        )

    tqdm.tqdm.write(f"Dataset regeneration complete! Saved regenerated dataset at: {target_dir}")
    tqdm.tqdm.write(f"Saved metainfo JSON at: {metainfo_json_out_path}")
    tqdm.tqdm.write(f"Total # episodes replayed: {num_replays}")
    tqdm.tqdm.write(f"Total # successful replays kept: {num_success}")
    tqdm.tqdm.write(f"Total # no-op actions filtered out: {num_noops}")


# ==============================================================================
# 8. 命令行入口
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90", "libero_suction"],
        required=True,
        help="要处理的 LIBERO task suite 名称。",
    )
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        required=True,
        help="阶段 1 原始采集 HDF5 目录。",
    )
    parser.add_argument(
        "--libero_target_dir",
        type=str,
        default=None,
        help="阶段 2 清洗后 HDF5 目录；默认写到 data_regeneration/processed_{task_suite}。",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="相机图像分辨率。",
    )
    parser.add_argument(
        "--control_freq",
        type=int,
        default=DEFAULT_CONTROL_FREQ,
        help="环境控制频率。",
    )
    parser.add_argument(
        "--task_limit",
        type=int,
        default=None,
        help="只处理前多少个 task；调试时很有用。",
    )
    parser.add_argument(
        "--tail_dummy_steps",
        type=int,
        default=DEFAULT_TAIL_DUMMY_STEPS,
        help="no-op 过滤后，尾部额外追加多少步吸盘关闭的 dummy action，用于等待物体落入篮子。",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
