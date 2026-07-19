"""
collect_libero_suction_handover_dataset.py

【脚本作用】
这是吸盘版 LIBERO 数据再生成流程的第一阶段采集脚本。
它不会删除 no-op，也不会筛掉失败轨迹；它只负责把“浏览器看到的原始任务预览”
和“真实执行的自动前段 + 人工后段”完整采下来，保存为后续第二阶段清洗脚本可直接读取的原始 HDF5。

【这份脚本现在的页面流程】
1. 进入某个未完成 task 后，浏览器先显示两个按钮：
   - 播放原始LIBERO轨迹
   - 退出
2. 用户点击“播放原始LIBERO轨迹”后，脚本会自动寻找该 task 第一条成功的源 demo，并在浏览器里完整播放一遍。
3. 预览结束后，浏览器显示三个按钮：
   - 数据采集
   - 重播
   - 退出
4. 点击“数据采集”后，脚本开始采当前 task 中下一个未完成 demo。
5. 采集时：
   - 自动模式沿用原始轨迹前段；
   - 用户按 H 切到人工模式；
   - 用户按 R 放弃当前 demo，并回到“数据采集 / 重播 / 退出”页面；
   - 用户按 ESC 直接退出整个脚本。
6. 一个 demo 保存完成后，如果当前 task 还没采完，就再次回到
   “数据采集 / 重播 / 退出”页面，方便继续下一个 demo 或重新看预览。
7. 当前 task 全部完成后，脚本会自动进入下一个未完成 task，并重新显示
   “播放原始LIBERO轨迹 / 退出”。

【断点续采规则】
- 输出目录固定为：/data1/ps/openvla/data/{任务套件名}
- 每个 task 对应一个 {task.name}_demo.hdf5
- 再次运行脚本时，会先扫描这些 HDF5：
  - 如果某个 task 已经包含了和源数据相同数量的 demo，就认定为“已完成”，直接跳过；
  - 如果只采到一部分，就从下一个缺失 demo 继续。
- 若同时存在“半完成 task”和“全新未开始 task”，脚本会优先进入半完成 task，避免浪费已经采到的一半工作量。

【运行示例】
先激活环境：
    conda activate openvla

然后运行，例如：
    python experiments/robot/libero/data_regeneration/collect_libero_suction_handover_dataset.py \
        --libero_task_suite libero_object \
        --libero_raw_data_dir ./LIBERO/libero/datasets/libero_object

【HDF5 字段删改说明】
这份脚本在尽量保持原始 LIBERO HDF5 主结构不变的前提下，针对吸盘语义做了以下收敛：

1. 保留原始主结构：
   - data/demo_x/actions
   - data/demo_x/states
   - data/demo_x/rewards
   - data/demo_x/dones
   - data/demo_x/robot_states
   - data/demo_x/obs/*

2. 修改字段语义：
   - obs/gripper_states
     原始 LIBERO 里它表示 default gripper 状态；
     这里把它改成“吸盘开关状态”，shape 固定为 (T, 1)。

3. 保留吸盘特有字段：
   - obs/suction_wrench
     这是吸盘六维力信息，原始 LIBERO 没有，但这里必须保留。

4. 明确删除的冗余字段：
   - obs/suction_on
   - obs/suction_latched
   - control_modes
   删除原因：
   - suction_on 已经并入 obs/gripper_states；
   - suction_latched 只作为运行时调试信息，不进入训练数据；
   - 自动/人工切换只保留 demo attr 里的 handover_step，不再保存逐步控制模式序列。

【最重要的数据来源原则】
原始 LIBERO 数据绝不能被整包照搬进第一阶段新 HDF5。

1. 原始数据只负责：
   - 提供初始场景参考（states[0]）
   - 提供自动播放前段的参考动作（orig_actions）

2. 新 HDF5 里真正落盘的数据，一律以本地 suction 场景真实执行结果为准：
   - states 来自本地 env_suction.sim.get_state()
   - robot_states 来自本地 suction 环境当前观测
   - obs/* 来自本地 suction 环境当前观测
   - rewards / dones 来自本地 suction 环境真实 step 结果

3. 自动阶段：
   - 只借用原始动作的机械臂运动部分
   - 吸盘动作强制保持关闭

4. 人工阶段：
   - action 完全来自浏览器遥操作
   - 与原始 LIBERO 后半段动作不再有直接继承关系
"""

from __future__ import annotations
# 开启更灵活的类型注解语法（例如在类型里直接写 list[dict]）。

import argparse
# 命令行参数解析：读取 --libero_task_suite 这类启动参数。
import json
# 读写 JSON（用于保存采集元信息 metainfo）。
import os
# 访问环境变量、路径拼接等操作系统能力。
import sys
# 修改 Python 模块搜索路径（sys.path）。
import time
# sleep 等时间控制函数。
from pathlib import Path
# 更现代的路径对象，替代纯字符串路径。

import h5py
# 读写 HDF5 文件（本脚本核心数据格式）。
import numpy as np
# 数值计算库，处理动作、状态、图像数组。
from PIL import Image, ImageDraw, ImageFont
# Pillow：用于绘图、字体渲染、图像缩放与拼接。

# ------------------------------------------------------------------------------
# 路径与环境初始化
# ------------------------------------------------------------------------------
# ROOT_DIR 指向 openvla 仓库根目录。后面固定输出目录会基于它拼接。
ROOT_DIR = Path(__file__).resolve().parents[4]

# LIBERO_DIR 指向仓库里内置的 LIBERO 子目录。这样后面的 import 能直接找到 libero 包。
LIBERO_DIR = ROOT_DIR / "LIBERO"
# 采集脚本里的 BDDL 一律显式走当前仓库内置的 LIBERO 目录，不依赖 ~/.libero/config.yaml。
LOCAL_BDDL_ROOT = LIBERO_DIR / "libero" / "libero" / "bddl_files"

# 第一阶段数据采集的输出根目录固定写在 /data1/ps/openvla/data_regeneration 下。
DATA_OUTPUT_ROOT = ROOT_DIR / "data_regeneration"

# 使用 EGL 做离屏渲染，方便在服务器无头环境下运行。
os.environ.setdefault("MUJOCO_GL", "egl")

# 把仓库根目录和 LIBERO 子目录都放进 sys.path，确保 import 成功。
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(LIBERO_DIR))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from experiments.robot.libero.data_regeneration.realtime_control_and_monitor import (
    BrowserKeyboardController,
    RealtimeMonitorState,
    start_monitor_server,
)
from experiments.robot.libero.libero_utils import get_libero_dummy_action, quat2axisangle


# ------------------------------------------------------------------------------
# 默认参数与阶段常量
# ------------------------------------------------------------------------------
DEFAULT_RESOLUTION = 256
# 画面默认分辨率（256x256）。
DEFAULT_CONTROL_FREQ = 20
# 控制频率：每秒 step 多少次。
DEFAULT_SETTLE_STEPS = 10
# 重置到初始状态后先空跑几步，等待物理稳定。
DEFAULT_MANUAL_MAX_STEPS = 1200
# 人工接管后最多采集多少步，防止误操作导致无限采集。
DEFAULT_MENU_MONITOR_HZ = 2.0
# 菜单阶段只需要静态展示，低频推送即可，避免占满远程带宽。
DEFAULT_PREVIEW_MONITOR_HZ = 8.0
# 预览阶段需要连续播放，但不必和控制频率一样高。
DEFAULT_COLLECT_MONITOR_HZ = 12.0
# 真实采集阶段仍需较顺滑监看，但优先保证按钮/键盘响应。
DEFAULT_AUTO_REPLAY_SPEED_SCALE = 0.5
# 自动采集阶段默认按 0.5x 播放原始动作。
# 也就是说，如果环境 control_freq 是 20Hz，那么自动段默认按 10Hz 回放。
# 这样用户在浏览器里看到机械臂接近目标物体时，通常来得及按 H 切人工。

# 这两个字符串是“机械臂当前控制模式”，会显示在浏览器页面上。
AUTO_MODE = "AUTO_REPLAY"
MANUAL_MODE = "MANUAL_BROWSER"

# 下面这些字符串表示“当前页面处于哪一个阶段”。
STAGE_TASK_MENU = "TASK_MENU"
STAGE_PREVIEW = "PREVIEW_DEMO"
STAGE_POST_PREVIEW_MENU = "POST_PREVIEW_MENU"
STAGE_COLLECTING = "COLLECTING"
STAGE_ALL_DONE = "ALL_TASKS_DONE"

# 浏览器按钮的内部 ID。前端点击按钮后会把这些 ID 发回服务器。
BUTTON_PREVIEW = "preview"
BUTTON_EXIT = "exit"
BUTTON_COLLECT = "collect"
BUTTON_REPLAY = "replay"


def build_action_button(button_id: str, label: str, description: str) -> dict:
    """构造一个按钮描述字典，供页面直接渲染。"""
    return {
        "button_id": str(button_id),
        # 前后端约定的按钮唯一 ID，点击后会回传这个字段。
        "label": str(label),
        # 页面上显示的按钮文字。
        "description": str(description),
        # 对按钮作用的解释文字。
    }


# 这两个按钮只用于“刚进入某个 task 时”的入口页面。
TASK_MENU_BUTTONS = [
    build_action_button(
        BUTTON_PREVIEW,
        "播放原始LIBERO轨迹",
        "先完整播放一次原始成功轨迹，帮助你确认目标物体是谁、应该放到哪里，再决定是否开始采集。",
    ),
    build_action_button(
        BUTTON_EXIT,
        "退出",
        "安全退出当前脚本。如果你现在不想继续采集，可以直接点这个按钮结束程序。",
    ),
]

# 这三个按钮用于“预览完成之后、以及每个 demo 保存之后”的决策页面。
POST_PREVIEW_BUTTONS = [
    build_action_button(
        BUTTON_COLLECT,
        "数据采集",
        "进入当前 task 的真实采集阶段，开始采当前下一个未完成 demo。",
    ),
    build_action_button(
        BUTTON_REPLAY,
        "重播",
        "再完整看一遍原始成功轨迹，重新确认目标物体和目标位置。",
    ),
    build_action_button(
        BUTTON_EXIT,
        "退出",
        "结束本次采集，安全退出脚本。已经保存到磁盘的数据不会丢失。",
    ),
]


def _load_font(size: int):
    """尽量加载常见中文字体，方便在左侧力面板里画中文标题。"""
    font_candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
            # 只要某个字体可用，就立即返回。
        except Exception:
            continue
            # 当前字体不可用则尝试下一个。
    return ImageFont.load_default()
    # 所有候选字体都失败时，退回 PIL 默认字体，保证不崩溃。


def create_force_display(force_data: np.ndarray, title: str, width: int = 256, height: int = 256) -> np.ndarray:
    """把 6 维力/力矩数据画成左侧面板。"""
    img = Image.new("RGB", (width, height), color=(40, 40, 50))
    # 先创建一个深色背景画布。
    draw = ImageDraw.Draw(img)
    # 获取绘图上下文，后续用 draw.text 画文字。
    font_title = _load_font(22)
    font_label = _load_font(16)
    font_value = _load_font(20)

    draw.text((width // 2, 12), title, fill=(255, 255, 255), font=font_title, anchor="mm")
    # 顶部居中绘制标题。

    y_start = 45
    # Force 区域起始纵坐标。
    draw.text((80, y_start), "Force (N)", fill=(100, 200, 255), font=font_label)
    draw.text((80, y_start + 25), f"fx: {force_data[0]:8.3f}", fill=(255, 255, 255), font=font_value)
    draw.text((80, y_start + 48), f"fy: {force_data[1]:8.3f}", fill=(255, 255, 255), font=font_value)
    draw.text((80, y_start + 71), f"fz: {force_data[2]:8.3f}", fill=(255, 255, 255), font=font_value)

    y_start = 135
    # Torque 区域起始纵坐标。
    draw.text((80, y_start), "Torque (Nm)", fill=(255, 180, 100), font=font_label)
    draw.text((80, y_start + 25), f"tx: {force_data[3]:8.3f}", fill=(255, 255, 255), font=font_value)
    draw.text((80, y_start + 48), f"ty: {force_data[4]:8.3f}", fill=(255, 255, 255), font=font_value)
    draw.text((80, y_start + 71), f"tz: {force_data[5]:8.3f}", fill=(255, 255, 255), font=font_value)

    return np.array(img)
    # 把 PIL 图像转回 numpy，方便后续和相机图拼接。


def create_combined_frame(agentview_image: np.ndarray, suction_wrench: np.ndarray, eye_in_hand_image: np.ndarray, img_size: int = 256) -> np.ndarray:
    """拼接三联监看画面：左侧力面板 + 中间主视角 + 右侧手眼图。"""
    if agentview_image.dtype != np.uint8:
        agentview_image = np.clip(agentview_image * 255.0, 0, 255).astype(np.uint8)
        # 若输入是 0~1 浮点图，转换成 0~255 的 uint8 图像。
    center_img = np.array(Image.fromarray(agentview_image).resize((img_size, img_size)).rotate(180, expand=False))
    # 主视角统一缩放到固定尺寸，并旋转 180 度以匹配观察习惯。

    if eye_in_hand_image.dtype != np.uint8:
        eye_in_hand_image = np.clip(eye_in_hand_image * 255.0, 0, 255).astype(np.uint8)
    if eye_in_hand_image.shape[:2] != (img_size, img_size):
        eye_in_hand_image = np.array(Image.fromarray(eye_in_hand_image).resize((img_size, img_size)))
        # 手眼图尺寸不一致时，缩放到统一尺寸。

    left_img = create_force_display(suction_wrench, "接触外力(世界系)", img_size, img_size)
    return np.concatenate([left_img, center_img, eye_in_hand_image], axis=1)
    # 横向拼成 [力面板 | 主视角 | 手眼图]。


def _mujoco_joint_sizes(joint_type: int) -> tuple[int, int]:
    """根据 MuJoCo joint 类型返回该 joint 在 qpos / qvel 里的长度。"""
    if joint_type == 0:
        return 7, 6
    if joint_type == 1:
        return 4, 3
    if joint_type in (2, 3):
        return 1, 1
    raise ValueError(f"Unknown MuJoCo joint type: {joint_type}")


def _build_joint_index(model) -> dict[str, tuple[int, int, int]]:
    """把 joint name 映射成 qpos 起点、qvel 起点、joint 类型。"""
    mapping: dict[str, tuple[int, int, int]] = {}
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name
        # 每个 joint 用“名字”做键，便于两个模型按名字对齐。
        mapping[name] = (
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
            int(model.jnt_type[joint_id]),
        )
    return mapping


def _is_robot_arm_joint(joint_name: str) -> bool:
    """
    判断某个 joint 是否属于 Panda 机械臂本体的 7 个关节。

    这里故意不用“是否包含 robot0”这种宽松规则，因为 gripper 相关名字里也可能带 robot0。
    我们只认 `robot0_joint1` 到 `robot0_joint7` 这 7 个真正的机械臂关节。
    """
    return joint_name.startswith("robot0_joint") and joint_name[len("robot0_joint") :].isdigit()


def _is_end_effector_related_joint(joint_name: str) -> bool:
    """
    判断某个 joint 是否属于末端执行器相关状态。

    你的目标是：
    - 不迁移 default gripper 的末端状态；
    - 也不迁移 suction 自己的末端状态；
    - 自动播放时 suction 保持关闭，人工阶段才允许开启。

    所以只要 joint 名里明显带有 gripper / finger / suction 这些关键词，
    就统一视为“末端执行器相关 joint”，在转换时不从 default 拷贝，而是直接清零。
    """
    lowered = joint_name.lower()
    return ("gripper" in lowered) or ("finger" in lowered) or ("suction" in lowered)


def convert_default_init_state_to_suction(*, init_state_default, env_default: ControlEnv, env_suction: ControlEnv) -> np.ndarray:
    """
    把默认夹爪环境里的 init_state 转成吸盘环境可接受的 init_state。

    这里的核心思想不是“把 default init_state 原样照搬过去”，而是做一次有选择的迁移：

    1. Panda 机械臂本体：
       - 迁移 7 个机械臂关节的 qpos
       - 也迁移 7 个机械臂关节的 qvel
       这样做的含义是：机械臂的起始姿态和起始速度都尽量与原始 demo 对齐。
       这与 suction 单任务评估脚本里的转换策略保持一致。

    2. 末端执行器：
       - 不迁移 default gripper 的末端状态
       - suction 自己的 joint 不从原始状态继承
       - suction joint 的 qpos / qvel 强制置 0
       - 其他 suction-only 内部状态保留 suction reset 默认值
       这是因为 default gripper 和 suction 的物理语义本来就对不上。

    3. 环境物体：
       - 物体 joint 的 qpos / qvel 都尽量保留
       这样可以最大程度复现原始 demo 的场景布局和物体初始动态。

    4. joint 对齐规则：
       - 只要 default 与 suction 两个模型里存在同名 joint，就直接迁移它的 qpos / qvel
       - 唯一例外是 gripper0_suction_joint，它固定清零
       - suction-only / default-only joint 都不强行映射

    注意：
    - 这里不做 `xyzRxRyRz` 笛卡尔位姿迁移。
    - 原因是 `init_state` 本质是 MuJoCo 的 `[time, qpos, qvel]`，不是控制器命令。
    - 如果真按笛卡尔位姿迁移，还要额外做 IK 才能回到关节空间，复杂且不稳。
    - 对同一条 Panda 机械臂来说，直接迁移 7 个机械臂关节 qpos，就已经等价于把末端位姿迁过去了。
    """
    if hasattr(init_state_default, "detach"):
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

    # suction 环境自己的当前 qpos 可作为“未被原始状态覆盖部分”的默认值来源。
    # 例如某些 suction 模型专属 joint 在 default 环境里根本不存在，就保留 suction 默认模型值。
    qpos_s = np.asarray(env_suction.sim.data.qpos, dtype=np.float64).copy()
    # qvel 从 suction reset 默认值开始，再按“同名 joint 对齐”规则覆盖。
    # 这和吸盘评估脚本保持一致：机械臂与物体的已有速度尽量保留，只有 suction joint 强制清零。
    qvel_s = np.asarray(env_suction.sim.data.qvel, dtype=np.float64).copy()

    joints_d = _build_joint_index(model_d)

    for joint_id in range(model_s.njnt):
        joint_name = model_s.joint(joint_id).name
        qposadr_s = int(model_s.jnt_qposadr[joint_id])
        dofadr_s = int(model_s.jnt_dofadr[joint_id])
        joint_type_s = int(model_s.jnt_type[joint_id])
        qpos_sz_s, qvel_sz_s = _mujoco_joint_sizes(joint_type_s)

        # 末端执行器相关 joint 一律不迁移。
        # 这样可以确保：
        # - suction 不会继承 default gripper 的末端状态；
        # - 自动播放开始时，吸盘从“关闭 + 静止”的安全状态起步。
        if _is_end_effector_related_joint(joint_name):
            qpos_s[qposadr_s : qposadr_s + qpos_sz_s] = 0.0
            qvel_s[dofadr_s : dofadr_s + qvel_sz_s] = 0.0
            continue

        # 某些 joint 只存在于 suction 模型中，不存在于 default 模型里；这些直接跳过。
        if joint_name not in joints_d:
            continue

        qposadr_d, dofadr_d, joint_type_d = joints_d[joint_name]
        qpos_sz_d, qvel_sz_d = _mujoco_joint_sizes(joint_type_d)

        if (qpos_sz_d, qvel_sz_d) != (qpos_sz_s, qvel_sz_s):
            raise ValueError(
                f"Joint size mismatch for {joint_name}: default {(qpos_sz_d, qvel_sz_d)} vs suction {(qpos_sz_s, qvel_sz_s)}"
            )

        # 现在的策略与 suction 单任务评估脚本一致：
        # 只要 joint 在两个模型里同名，就直接迁移 qpos / qvel。
        # 这会保留：
        # - 机械臂 7 轴的起始位姿和速度
        # - 场景中物体的起始位姿和速度
        # 同时不会把 default gripper 指关节错误映射到 suction，因为那些 joint 在 suction 模型里根本不存在。
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
    """创建吸盘环境。第一阶段真正采集数据时使用它。"""
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
        # 这里故意忽略底层 robosuite 的 horizon done。
        # 原因是当前采集流程已经由外层采集循环自己控制结束条件：
        # 1. 任务成功就结束；
        # 2. 自动段动作放完就结束；
        # 3. 人工段达到 manual_max_steps 就结束；
        # 4. 用户按 R / 退出时也会提前结束。
        # 如果这里保持 False，底层 episode 可能会先因为 horizon 到期而终止，
        # 外层采集循环却还没来得及停下，下一次 step 就会报
        # “executing action in terminated episode”。
        ignore_done=True,
    )


def create_default_env(task, resolution: int, control_freq: int) -> ControlEnv:
    """创建默认夹爪环境。它只用于两件事：状态转换和原始轨迹预览。"""
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
        # 预览环境和状态转换环境也保持同样设置，避免 default 环境与 suction 环境
        # 在 episode 终止策略上不一致，导致预览或状态转换时出现额外分叉行为。
        ignore_done=True,
    )


def extract_suction_wrench(obs: dict, info: dict | None = None) -> np.ndarray:
    """统一提取当前时刻六维力；没有时返回全 0。"""
    if "suction_wrench" in obs and obs["suction_wrench"] is not None:
        return np.asarray(obs["suction_wrench"], dtype=np.float64)
    if info is not None and "suction_force_torque" in info and info["suction_force_torque"] is not None:
        return np.asarray(info["suction_force_torque"], dtype=np.float64)
    return np.zeros(6, dtype=np.float64)


def extract_robot_state(obs: dict) -> np.ndarray:
    """
    提取一个紧凑版 robot_state。

    注意：
    - 这里返回的是“当前本地 suction 场景真实观测”拼出的 robot_state；
    - 不是把原始 LIBERO HDF5 里的 robot_states 直接复制过来。
    - 之所以仍然沿用 robot_states 这个键名，是为了尽量保持与原始 HDF5 主结构同构。
    """
    return np.concatenate([np.atleast_1d(obs["robot0_gripper_qpos"]), obs["robot0_eef_pos"], obs["robot0_eef_quat"]], axis=0)


def init_episode_buffers() -> dict:
    """
    初始化一条待保存 demo 的内存缓冲区。

    这里显式不再保存下面这些冗余字段：
    - obs/suction_on
    - obs/suction_latched
    - control_modes

    原因：
    - obs/gripper_states 仍沿用原始 LIBERO 的语义，表示末端执行器内部状态；
    - suction_on / suction_latched 只作为运行时调试信息；
    - 自动/人工切换只保留 handover_step 这个 attr。
    """
    return {
        "states": [],
        # MuJoCo 全状态（qpos/qvel 等）。
        "actions": [],
        # 每一步执行的动作向量。
        "robot_states": [],
        # 紧凑机器人状态（和历史格式兼容）。
        "rewards": [],
        "dones": [],
        "gripper_states": [],
        # 这里仍沿用原始 LIBERO 的语义：保存末端执行器内部状态。
        # 对 suction gripper 来说，它来自 robot0_gripper_qpos，shape 为 (T, 1)。
        "joint_states": [],
        "ee_states": [],
        "agentview_images": [],
        "eye_in_hand_images": [],
        "suction_wrench": [],
    }


def record_observation(buffers: dict, obs: dict, info: dict | None = None):
    """
    把当前时刻观测写进缓冲区，供稍后一次性落盘。

    这里强调两个关键点：
    1. 写进 buffers 的都是“当前本地 suction 环境真实观测”；
    2. 不是把原始 LIBERO HDF5 的 obs 数据直接搬过来。

    此外：
    - obs/gripper_states 仍表示末端执行器内部状态（这里是 suction 的 robot0_gripper_qpos）；
    - 吸盘开关状态 suction_on / suction_latched 不再进入 HDF5，只保留为运行时调试信息。
    """
    buffers["gripper_states"].append(np.atleast_1d(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)).copy())
    buffers["joint_states"].append(np.asarray(obs["robot0_joint_pos"], dtype=np.float64).copy())
    buffers["ee_states"].append(
        np.hstack((np.asarray(obs["robot0_eef_pos"], dtype=np.float64), quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))))
    )
    buffers["agentview_images"].append(np.asarray(obs["agentview_image"]).copy())
    buffers["eye_in_hand_images"].append(np.asarray(obs["robot0_eye_in_hand_image"]).copy())
    buffers["suction_wrench"].append(extract_suction_wrench(obs, info).copy())


def save_episode_to_hdf5(grp, demo_name: str, buffers: dict, success: bool, handover_step: int, source_demo_index: int):
    """
    把一条已经采完的 demo 正式写入 HDF5。

    这个函数只负责“真正保存成功采完的一条 demo”。
    如果中途用户按了 R 重启当前 demo，那么根本不会调用这个函数。

    这里要特别说明 HDF5 schema 的取舍：
    - 保留与原始 LIBERO 同名的主键，尽量保持结构同构；
    - obs/gripper_states 继续表示末端执行器内部状态；
    - obs/suction_wrench 作为吸盘特有模态保留；
    - obs/suction_on / obs/suction_latched / control_modes 明确不再写入。

    同时，本函数写入的 states / robot_states / obs/* 都来自第一阶段真实执行后的本地 suction 场景，
    不是把原始 LIBERO HDF5 的这些字段直接复制过来。
    """
    if demo_name in grp:
        raise ValueError(f"目标 HDF5 中已经存在 {demo_name}，这说明续采索引计算有问题。")

    ep_data_grp = grp.create_group(demo_name)
    ep_data_grp.attrs["success"] = bool(success)
    ep_data_grp.attrs["handover_step"] = int(handover_step)
    ep_data_grp.attrs["source_demo_index"] = int(source_demo_index)

    obs_grp = ep_data_grp.create_group("obs")
    obs_grp.create_dataset("gripper_states", data=np.stack(buffers["gripper_states"], axis=0))
    obs_grp.create_dataset("joint_states", data=np.stack(buffers["joint_states"], axis=0))
    obs_grp.create_dataset("ee_states", data=np.stack(buffers["ee_states"], axis=0))
    obs_grp.create_dataset("ee_pos", data=np.stack(buffers["ee_states"], axis=0)[:, :3])
    obs_grp.create_dataset("ee_ori", data=np.stack(buffers["ee_states"], axis=0)[:, 3:])
    obs_grp.create_dataset("agentview_rgb", data=np.stack(buffers["agentview_images"], axis=0))
    obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(buffers["eye_in_hand_images"], axis=0))
    obs_grp.create_dataset("suction_wrench", data=np.stack(buffers["suction_wrench"], axis=0))
    # 注意：
    # - gripper_states 这里保存的是 suction gripper 的 robot0_gripper_qpos，shape 为 (T, 1)；
    # - 不再额外写 obs/suction_on 和 obs/suction_latched，它们只保留在运行时监控里。

    ep_data_grp.create_dataset("actions", data=np.stack(buffers["actions"], axis=0))
    ep_data_grp.create_dataset("states", data=np.stack(buffers["states"], axis=0))
    ep_data_grp.create_dataset("robot_states", data=np.stack(buffers["robot_states"], axis=0))
    ep_data_grp.create_dataset("rewards", data=np.asarray(buffers["rewards"], dtype=np.uint8))
    ep_data_grp.create_dataset("dones", data=np.asarray(buffers["dones"], dtype=np.uint8))
    # control_modes 不再保存为逐步 dataset；
    # 自动转人工的位置只通过 handover_step attr 表达。


def task_output_dir(task_suite_name: str) -> Path:
    """根据任务套件名计算固定输出目录。"""
    return DATA_OUTPUT_ROOT / task_suite_name


def metainfo_path_for_suite(task_suite_name: str) -> Path:
    """根据任务套件名计算 metainfo JSON 路径。"""
    return task_output_dir(task_suite_name) / f"{task_suite_name}_raw_collection_metainfo.json"


def parse_demo_index(demo_name: str) -> int | None:
    """把 demo_3 这种名字解析成 3；不符合格式时返回 None。"""
    if not demo_name.startswith("demo_"):
        return None
    suffix = demo_name.split("_", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def list_completed_demo_indices(target_data_path: Path) -> list[int]:
    """
    读取某个 task 的输出 HDF5，并返回已经落盘成功的 demo 索引列表。

    这里只认 group 名符合 demo_x 的条目，其他异常 group 一律忽略。
    """
    if not target_data_path.exists():
        return []

    completed: list[int] = []
    with h5py.File(target_data_path, "r") as h5_file:
        if "data" not in h5_file:
            return []
        for demo_name in h5_file["data"].keys():
            demo_index = parse_demo_index(demo_name)
            if demo_index is not None:
                completed.append(demo_index)
    return sorted(set(completed))


def next_missing_demo_index(raw_demo_count: int, completed_demo_indices: list[int]) -> int | None:
    """找到当前 task 中下一个还没采到的 demo 索引。"""
    completed_set = set(completed_demo_indices)
    for demo_index in range(raw_demo_count):
        if demo_index not in completed_set:
            return demo_index
    return None


def load_metainfo_json(json_path: Path) -> dict:
    """如果 metainfo JSON 已存在，就先读回来，避免覆盖以前的记录。"""
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metainfo_json(json_path: Path, metainfo_json_dict: dict):
    """把 metainfo 字典落盘成 JSON。"""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metainfo_json_dict, f, indent=2, ensure_ascii=False)
        # ensure_ascii=False 可以直接写中文，不会变成 \uXXXX。


def clear_transient_browser_events(controller: BrowserKeyboardController):
    """
    清掉一次性事件，避免旧的点击/按键残留到新阶段。

    这里故意不调用 consume_shutdown_request()，因为退出请求应该保留给上层主循环处理。
    """
    controller.clear_pressed_keys()
    # 清除“持续按下”的键状态。
    controller.consume_button_action()
    # 清除“上一次按钮点击”的一次性事件。
    controller.consume_handover_request()
    # 清除“切人工”一次性事件。
    controller.consume_restart_request()
    # 清除“重启当前 demo”一次性事件。


def get_env_action_dim(env: ControlEnv) -> int:
    """从底层 robosuite action_spec 里安全取动作维度。"""
    low, _ = env.env.action_spec
    return int(np.asarray(low).shape[0])


def build_frame_from_obs(obs: dict, info: dict | None, resolution: int) -> np.ndarray:
    """根据当前 obs / info 生成浏览器要看的三联画面。"""
    return create_combined_frame(
        obs["agentview_image"],
        extract_suction_wrench(obs, info),
        obs["robot0_eye_in_hand_image"],
        img_size=resolution,
    )


def push_monitor_state(
    *,
    monitor_state: RealtimeMonitorState,
    controller: BrowserKeyboardController,
    frame_rgb: np.ndarray,
    mode_label: str,
    stage_title: str,
    stage_description: str,
    action_buttons: list[dict],
    obs: dict | None = None,
    info: dict | None = None,
    extra_lines: list[str] | None = None,
    update_hz: float | None = None,
):
    """统一把当前阶段的页面状态推给浏览器，避免各处手写 update 参数。"""
    wrench = extract_suction_wrench(obs or {}, info)
    monitor_state.update(
        frame_rgb=frame_rgb,
        mode_label=mode_label,
        suction_on=bool((info or {}).get("suction_on", False)),
        suction_latched=bool((info or {}).get("suction_latched", False)),
        suction_desired=controller.get_desired_suction_state(),
        suction_wrench=wrench,
        suction_wrench_total_load=np.asarray(
            (info or {}).get("suction_wrench_total_load_world", np.zeros(6, dtype=np.float64)),
            dtype=np.float64,
        ),
        stage_title=stage_title,
        stage_description=stage_description,
        action_buttons=action_buttons,
        extra_lines=extra_lines or [],
        update_hz=update_hz,
    )


def prepare_env_at_demo_start(env: ControlEnv, init_state, settle_steps: int) -> tuple[dict, dict]:
    """
    把某个环境重置到指定 demo 的起始状态，并等环境稳定几步。

    返回值是稳定后的 obs 和 info，后续页面菜单和预览都基于这个状态开始显示。
    """
    env.reset()
    obs = env.set_init_state(init_state)
    info: dict = {}
    for _ in range(settle_steps):
        obs, _, _, info = env.step(get_libero_dummy_action("openvla"))
    return obs, info


def find_first_successful_preview_demo_index(orig_data, env_default: ControlEnv, settle_steps: int) -> tuple[int, bool]:
    """
    从 demo_0 开始回放源轨迹，找到第一条成功 demo。

    返回：
    - preview_demo_index：用于预览的 demo 索引
    - found_success：是否真的找到了成功 demo

    如果没找到任何成功 demo，就回退到 demo_0，并把 found_success 设为 False。
    """
    raw_demo_count = len(orig_data.keys())
    for demo_index in range(raw_demo_count):
        demo_data = orig_data[f"demo_{demo_index}"]
        orig_actions = demo_data["actions"][()]
        orig_states = demo_data["states"][()]

        _, _ = prepare_env_at_demo_start(env_default, orig_states[0], settle_steps)

        done = False
        for action in orig_actions:
            _, _, done, _ = env_default.step(np.asarray(action, dtype=np.float64).tolist())
            if done:
                break

        if done or env_default.check_success():
            return demo_index, True

    return 0, False


def wait_for_button_choice(
    controller: BrowserKeyboardController,
    *,
    allowed_button_ids: set[str],
    poll_interval: float = 0.05,
) -> str:
    """阻塞等待浏览器点击某个允许的按钮；若收到退出请求则抛出 KeyboardInterrupt。"""
    while True:
        if controller.consume_shutdown_request():
            raise KeyboardInterrupt("收到浏览器退出请求。")
            # 把退出请求抛给上层，让上层统一清理资源。

        button_id = controller.consume_button_action()
        if button_id in allowed_button_ids:
            return button_id
            # 只有命中允许按钮才返回，其他点击一律忽略。

        time.sleep(poll_interval)
        # 降低空轮询 CPU 占用。


def play_preview_demo(
    *,
    env_default: ControlEnv,
    controller: BrowserKeyboardController,
    monitor_state: RealtimeMonitorState,
    preview_demo_data,
    task_description: str,
    preview_demo_index: int,
    resolution: int,
    control_freq: int,
):
    """
    在浏览器里完整播放一遍原始成功轨迹。

    这个阶段不采数据，也不接收“数据采集”之类的按钮，只负责让用户看懂：
    - 要操作的是哪个物体；
    - 最后应该放到哪里。
    """
    orig_actions = preview_demo_data["actions"][()]
    orig_states = preview_demo_data["states"][()]

    obs, info = prepare_env_at_demo_start(env_default, orig_states[0], settle_steps=0)

    for step_index, action in enumerate(orig_actions):
        if controller.consume_shutdown_request():
            raise KeyboardInterrupt("收到浏览器退出请求。")

        frame_rgb = build_frame_from_obs(obs, info, resolution)
        push_monitor_state(
            monitor_state=monitor_state,
            controller=controller,
            frame_rgb=frame_rgb,
            mode_label=STAGE_PREVIEW,
            stage_title="原始轨迹预览",
            stage_description="当前正在自动播放原始 LIBERO 成功轨迹。这个阶段不采集数据，只是帮助你先看懂目标物体和目标位置。",
            action_buttons=[],
            obs=obs,
            info=info,
            extra_lines=[
                f"当前任务：{task_description}",
                f"当前预览示范：demo_{preview_demo_index}",
                f"当前预览进度：第 {step_index + 1} / {len(orig_actions)} 步",
                "请先观察目标物体是谁、最终要放到哪里。",
            ],
            update_hz=DEFAULT_PREVIEW_MONITOR_HZ,
        )

        obs, _, _, info = env_default.step(np.asarray(action, dtype=np.float64).tolist())
        time.sleep(1.0 / max(control_freq, 1))

    # 预览结束后再推一次最终帧，防止页面停在倒数第二帧。
    final_frame = build_frame_from_obs(obs, info, resolution)
    push_monitor_state(
        monitor_state=monitor_state,
        controller=controller,
        frame_rgb=final_frame,
        mode_label=STAGE_PREVIEW,
        stage_title="原始轨迹预览已结束",
        stage_description="你已经看完一遍原始成功轨迹。现在可以开始采集、重新播放，或者直接退出。",
        action_buttons=[],
        obs=obs,
        info=info,
        extra_lines=[
            f"当前任务：{task_description}",
            f"刚刚播放的预览示范：demo_{preview_demo_index}",
            "原始轨迹预览已经播放完成。",
        ],
        update_hz=DEFAULT_MENU_MONITOR_HZ,
    )

    return obs, info, final_frame


def collect_one_demo(
    *,
    task,
    demo_index: int,
    orig_data,
    env_suction: ControlEnv,
    env_default: ControlEnv,
    controller: BrowserKeyboardController,
    monitor_state: RealtimeMonitorState,
    args,
) -> dict:
    """
    执行一条真实 demo 的采集。

    返回值里的 status 只有两种：
    - restart：用户按了 R，本条 demo 作废，回到“数据采集 / 重播 / 退出”页面
    - saved：这条 demo 正常结束，可写入 HDF5
    """
    # 进入新 demo 之前先把浏览器残留状态清干净，避免旧按键或旧按钮影响新回合。
    controller.reset_episode_state()
    clear_transient_browser_events(controller)

    if controller.consume_shutdown_request():
        raise KeyboardInterrupt("收到浏览器退出请求。")

    demo_data = orig_data[f"demo_{demo_index}"]
    orig_actions = demo_data["actions"][()]
    orig_states = demo_data["states"][()]

    # 这里只把原始 demo 的起始状态当成“场景初始化参考”，
    # 真实进入采集后的所有落盘 states / robot_states / obs 都必须来自本地 suction 场景。
    converted_state = convert_default_init_state_to_suction(
        init_state_default=orig_states[0],
        env_default=env_default,
        env_suction=env_suction,
    )
    obs, info = prepare_env_at_demo_start(env_suction, converted_state, args.settle_steps)

    buffers = init_episode_buffers()
    control_mode = AUTO_MODE
    handover_step = -1
    auto_action_index = 0
    manual_step_count = 0
    done = False

    print(f"  开始采集当前 task 的 demo_{demo_index}")
    auto_replay_hz = max(float(args.control_freq) * float(args.auto_replay_speed_scale), 1e-3)
    auto_replay_sleep = 1.0 / auto_replay_hz
    # 自动采集阶段单独使用一个“降速后的目标频率”。
    # 这里故意不影响原始轨迹预览速度，只影响“点击数据采集后”的 AUTO_REPLAY。
    # 默认 0.5x 时：20Hz -> 10Hz，更容易在接近目标时按 H 切人工。

    # 用户刚点击“数据采集”时，先立即推送一次页面状态，明确显示正在采哪条 demo。
    init_frame = build_frame_from_obs(obs, info, args.resolution)
    push_monitor_state(
        monitor_state=monitor_state,
        controller=controller,
        frame_rgb=init_frame,
        mode_label=control_mode,
        stage_title=f"数据采集中（demo_{demo_index}）",
        stage_description="已进入真实采集阶段。",
        action_buttons=[],
        obs=obs,
        info=info,
        extra_lines=[
            f"当前正在采集：demo_{demo_index}",
            "当前采集模式：自动回放",
        ],
        update_hz=DEFAULT_COLLECT_MONITOR_HZ,
    )

    while True:
        if controller.consume_shutdown_request():
            raise KeyboardInterrupt("收到浏览器退出请求。")

        if controller.consume_restart_request():
            print(f"  当前 demo_{demo_index} 收到重启请求，丢弃本次缓存并返回上一级菜单。")
            return {"status": "restart"}
            # 直接返回 restart，调用方不会保存 buffers，当前 demo 数据自然清空。

        if control_mode == AUTO_MODE and controller.consume_handover_request():
            control_mode = MANUAL_MODE
            handover_step = len(buffers["actions"])
            print(f"  demo_{demo_index} 在 step={handover_step} 触发 handover，切到人工模式。")
            # 记录 handover_step，后续可分析“哪一步开始人工接管”。

        if control_mode == AUTO_MODE and auto_action_index >= len(orig_actions):
            break

        if control_mode == MANUAL_MODE and manual_step_count >= args.manual_max_steps:
            print(f"  demo_{demo_index} 的人工控制步数达到上限 {args.manual_max_steps}，结束本条采集。")
            break

        if control_mode == AUTO_MODE:
            # 自动阶段只借用原始动作的“机械臂运动部分”作为参考。
            # 这里绝不是把原始 demo 的整条数据直接搬进新 HDF5。
            # 真正写盘的 states / robot_states / obs / rewards / dones，
            # 都必须来自本地 suction 环境真实执行后的结果。
            # 同时最后一维强制保持 suction 关闭，避免把原始 default gripper 的抓取语义错误映射到吸盘上。
            action = np.asarray(orig_actions[auto_action_index], dtype=np.float64).copy()
            action[-1] = -1.0
            # 最后一维约定为吸盘开关，这里强制为 -1（关闭）。
            auto_action_index += 1
        else:
            # 人工阶段完全读取浏览器当前键盘状态。
            # 从这一刻开始，action 和原始 LIBERO 后半段再也没有直接关系。
            action = controller.get_action(get_env_action_dim(env_suction))
            # 按当前键盘状态实时生成动作。
            manual_step_count += 1

        # 先记录“执行这个 action 之前”的环境状态和观测，再推进一步。
        # 这里记录的一律是本地 suction 场景真实数据，不能直接复用原始 demo 中同名字段。
        buffers["states"].append(env_suction.sim.get_state().flatten().copy())
        buffers["robot_states"].append(extract_robot_state(obs))
        record_observation(buffers, obs, info)
        buffers["actions"].append(np.asarray(action, dtype=np.float64).copy())

        obs, reward, done, info = env_suction.step(action.tolist())
        # 真正把动作送入环境执行一步。
        buffers["rewards"].append(np.uint8(reward))
        buffers["dones"].append(np.uint8(done))

        frame_rgb = build_frame_from_obs(obs, info, args.resolution)
        push_monitor_state(
            monitor_state=monitor_state,
            controller=controller,
            frame_rgb=frame_rgb,
            mode_label=control_mode,
            stage_title=f"数据采集中（demo_{demo_index}）",
            stage_description="当前正在采集真实吸盘轨迹。",
            action_buttons=[],
            obs=obs,
            info=info,
            extra_lines=[
                f"当前正在采集：demo_{demo_index}",
                f"当前采集模式：{'人工控制' if control_mode == MANUAL_MODE else '自动回放'}",
            ],
            update_hz=DEFAULT_COLLECT_MONITOR_HZ,
        )

        if done:
            break

        if control_mode == AUTO_MODE:
            # 自动采集阶段显式降速，避免重播太快导致用户来不及切人工。
            # 注意：
            # 1. 这里只影响自动采集阶段；
            # 2. 原始轨迹预览阶段仍保持原来的预览播放速度；
            # 3. 人工阶段不 sleep，因为人工控制更需要即时响应。
            time.sleep(auto_replay_sleep)

    success = bool(done or env_suction.check_success())
    print(
        f"  demo_{demo_index} 采集完成：steps={len(buffers['actions'])}, "
        f"success={success}, handover_step={handover_step}"
    )
    return {
        "status": "saved",
        "buffers": buffers,
        "success": success,
        "handover_step": handover_step,
        "source_demo_index": demo_index,
    }


def main(args):
    """
    第一阶段主流程。

    这条主流程严格按照用户给的流程图组织：
    1. 启动时先扫描输出目录，看哪些 task 已完成、哪些没完成；
    2. 进入某个未完成 task 后，先展示“播放原始LIBERO轨迹 / 退出”；
    3. 预览完成后，展示“数据采集 / 重播 / 退出”；
    4. 数据采集一次只采一个 demo；
    5. 保存后若当前 task 未完成，则回到“数据采集 / 重播 / 退出”；
    6. 当前 task 全部完成后，进入下一个未完成 task。
    """
    # 目标输出目录固定为 /data1/ps/openvla/data/{suite}
    target_dir = task_output_dir(args.libero_task_suite)
    target_dir.mkdir(parents=True, exist_ok=True)

    # metainfo 也统一放在这个目录里，方便一起查看。
    metainfo_json_out_path = metainfo_path_for_suite(args.libero_task_suite)
    metainfo_json_dict = load_metainfo_json(metainfo_json_out_path)

    # 浏览器控制器负责保存当前按键状态、按钮点击事件、一次性 handover/restart/quit 事件。
    controller = BrowserKeyboardController(
        movement_speed=args.movement_speed,
        enable_handover=True,
        reset_suction_on_reset=True,
    )

    # 监看状态对象会把最新画面、阶段说明、按钮说明、六维力等缓存起来，交给 WebSocket 服务持续推送。
    monitor_state = RealtimeMonitorState(title="LIBERO 吸盘原始采集监看", jpeg_quality=70, update_hz=DEFAULT_MENU_MONITOR_HZ)

    # 启动浏览器监看与控制服务。页面现在既负责显示，也负责把按钮点击和键盘事件发回来。
    _, server = start_monitor_server(
        monitor_state,
        host=args.web_host,
        port=args.web_port,
        title="LIBERO 吸盘数据采集监看",
        subtitle="浏览器页面会先带你看原始成功轨迹，再让你决定是否开始采集；页面中会显示当前阶段、按钮作用和按键作用。",
        keyboard_lines=[
            "先点击页面，使浏览器获得键盘焦点",
            "任务入口页和预览完成页：请优先使用页面按钮",
            "数据采集时：H 切换到人工模式",
            "数据采集时：W / S 控制 X 方向前后",
            "数据采集时：A / D 控制 Y 方向左右",
            "数据采集时：↑ / ↓ 控制 Z 方向上下",
            "数据采集时：SPACE 切换吸盘开 / 关",
            "数据采集时：R 重启当前 demo，返回菜单重新来",
            "任意阶段：ESC 退出整个脚本",
        ],
        control_state=controller,
        show_wrench_panels=False,
        keyboard_help_position="below_video",
        require_monitor_for_actions=True,
    )

    print("=" * 80)
    print("LIBERO 吸盘原始采集服务已启动")
    print(f"浏览器监看页面: http://{args.web_host if args.web_host != '0.0.0.0' else '127.0.0.1'}:{args.web_port}")
    print(f"固定输出目录: {target_dir}")
    print("浏览器页面会显示阶段说明、按钮作用、以及当前可用按键说明。")
    print("=" * 80)

    benchmark_dict = benchmark.get_benchmark_dict()
    # 读取 LIBERO 提供的所有任务套件构造器。
    task_suite = benchmark_dict[args.libero_task_suite]()
    # 根据用户参数实例化选定套件。
    num_tasks_in_suite = task_suite.n_tasks
    if args.task_limit is not None:
        num_tasks_in_suite = min(num_tasks_in_suite, args.task_limit)
        # 若设置 task_limit，只处理前 N 个任务。

    # 先扫描所有 task 的完成度，用于：
    # 1. 判断是不是已经全部采完；
    # 2. 把“半完成 task”排到前面优先继续。
    task_progress_list: list[dict] = []
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        orig_data_path = Path(args.libero_raw_data_dir) / f"{task.name}_demo.hdf5"
        if not orig_data_path.exists():
            raise FileNotFoundError(f"Cannot find raw data file: {orig_data_path}")

        with h5py.File(orig_data_path, "r") as orig_file:
            raw_demo_count = len(orig_file["data"].keys())

        target_data_path = target_dir / f"{task.name}_demo.hdf5"
        completed_demo_indices = list_completed_demo_indices(target_data_path)
        completed_demo_count = len(completed_demo_indices)
        is_completed = completed_demo_count >= raw_demo_count

        task_progress_list.append(
            {
                "task_id": task_id,
                "task": task,
                "orig_data_path": orig_data_path,
                "target_data_path": target_data_path,
                "raw_demo_count": raw_demo_count,
                "completed_demo_indices": completed_demo_indices,
                "completed_demo_count": completed_demo_count,
                "is_completed": is_completed,
            }
        )

    # 先把“已经做了一部分但没做完”的 task 放前面，再放“完全没开始”的 task。
    pending_task_list = [info for info in task_progress_list if not info["is_completed"]]
    pending_task_list.sort(
        key=lambda info: (
            0 if info["completed_demo_count"] > 0 else 1,
            -int(info["completed_demo_count"]),
            int(info["task_id"]),
        )
    )
    # 排序规则：
    # 1) 半完成 task（completed>0）优先；
    # 2) 已完成数量更多的优先（减少中断成本）；
    # 3) 再按 task_id 稳定排序。

    if not pending_task_list:
        push_monitor_state(
            monitor_state=monitor_state,
            controller=controller,
            frame_rgb=np.zeros((args.resolution, args.resolution * 3, 3), dtype=np.uint8),
            mode_label=STAGE_ALL_DONE,
            stage_title="所有 task 已完成",
            stage_description="当前任务套件里所有 task 都已经采完了。你现在可以直接关闭页面，或者按 Q / ESC 退出脚本。",
            action_buttons=[],
            obs=None,
            info=None,
            extra_lines=[
                f"当前任务套件：{args.libero_task_suite}",
                f"输出目录：{target_dir}",
                "当前没有待采集的任务。",
            ],
            update_hz=DEFAULT_MENU_MONITOR_HZ,
        )
        print(f"所有 task 已完成，输出目录：{target_dir}")
        monitor_state.shutdown_event.set()
        server.should_exit = True
        return

    try:
        # 逐个处理所有未完成 task。
        for task_info in pending_task_list:
            task = task_info["task"]
            task_description = task.language
            orig_data_path = task_info["orig_data_path"]
            target_data_path = task_info["target_data_path"]

            orig_data_file = h5py.File(orig_data_path, "r")
            orig_data = orig_data_file["data"]
            # orig_data 是源示教数据，每个 demo_x 里有 actions/states。

            # 这里用 a 模式打开目标文件，是为了支持断点续采。
            new_data_file = h5py.File(target_data_path, "a")
            grp = new_data_file["data"] if "data" in new_data_file else new_data_file.create_group("data")
            # 用追加模式打开目标文件；没有 data 组就创建。

            env_suction = create_suction_env(task, args.resolution, args.control_freq)
            env_default = create_default_env(task, args.resolution, args.control_freq)
            env_suction.seed(0)
            env_default.seed(0)

            try:
                # 先找这个 task 用来做预览的原始成功 demo。
                preview_demo_index, found_success = find_first_successful_preview_demo_index(
                    orig_data,
                    env_default,
                    args.settle_steps,
                )
                preview_demo_data = orig_data[f"demo_{preview_demo_index}"]
                preview_initial_obs, preview_initial_info = prepare_env_at_demo_start(
                    env_default,
                    preview_demo_data["states"][0],
                    args.settle_steps,
                )
                preview_menu_frame = build_frame_from_obs(preview_initial_obs, preview_initial_info, args.resolution)

                # first_task_menu=True 表示刚进入这个 task，要先看“播放原始LIBERO轨迹 / 退出”入口页。
                first_task_menu = True

                while True:
                    # 每次进入循环时，先重新读取当前 task 已完成 demo，支持中途断点续采继续推进。
                    completed_demo_indices = list_completed_demo_indices(target_data_path)
                    next_demo_index = next_missing_demo_index(len(orig_data.keys()), completed_demo_indices)

                    # 当前 task 已全部完成，退出内层循环，进入下一个 task。
                    if next_demo_index is None:
                        print(f"task '{task_description}' 已全部采集完成。")
                        break
                        # 当前 task 无缺失 demo，进入下一个 task。

                    clear_transient_browser_events(controller)

                    if first_task_menu:
                        # 这是“刚进入某个 task”时的专属页面，只显示两个按钮。
                        stage_description = (
                            "你现在进入了一个新的 task。先点击“播放原始LIBERO轨迹”，完整看一遍成功示范，"
                            "确认目标物体和目标位置；如果暂时不想继续，可以直接点击“退出”。"
                        )
                        if not found_success:
                            stage_description += " 当前没有找到成功 demo，预览时会回退播放 demo_0。"

                        push_monitor_state(
                            monitor_state=monitor_state,
                            controller=controller,
                            frame_rgb=preview_menu_frame,
                            mode_label=STAGE_TASK_MENU,
                            stage_title="任务入口页",
                            stage_description=stage_description,
                            action_buttons=TASK_MENU_BUTTONS,
                            obs=preview_initial_obs,
                            info=preview_initial_info,
                            extra_lines=[
                                f"当前任务：{task_description}",
                                f"下一条待采集示范：demo_{next_demo_index}",
                                f"当前将播放的预览示范：demo_{preview_demo_index}",
                                f"当前任务完成进度：{len(completed_demo_indices)} / {len(orig_data.keys())}",
                            ],
                            update_hz=DEFAULT_MENU_MONITOR_HZ,
                        )

                        button_choice = wait_for_button_choice(controller, allowed_button_ids={BUTTON_PREVIEW, BUTTON_EXIT})
                        if button_choice == BUTTON_EXIT:
                            raise KeyboardInterrupt("用户在任务入口页点击退出。")
                            # 抛到外层 finally，统一做资源清理。

                        # 进入原始轨迹预览。
                        _, _, preview_menu_frame = play_preview_demo(
                            env_default=env_default,
                            controller=controller,
                            monitor_state=monitor_state,
                            preview_demo_data=preview_demo_data,
                            task_description=task_description,
                            preview_demo_index=preview_demo_index,
                            resolution=args.resolution,
                            control_freq=args.control_freq,
                        )
                        first_task_menu = False

                    # 到了这里，说明要么是预览刚播放完，要么是当前 task 里某条 demo 刚采完/刚重启。
                    # 根据流程图，这里都应该显示“数据采集 / 重播 / 退出”。
                    while True:
                        clear_transient_browser_events(controller)

                        push_monitor_state(
                            monitor_state=monitor_state,
                            controller=controller,
                            frame_rgb=preview_menu_frame,
                            mode_label=STAGE_POST_PREVIEW_MENU,
                            stage_title="预览完成页",
                            stage_description=(
                                "你已经看完当前 task 的原始成功轨迹。"
                                "如果已经确认清楚目标物体和目标位置，就点击“数据采集”；"
                                "如果还想再看一遍，就点“重播”；如果不想继续，就点“退出”。"
                            ),
                            action_buttons=POST_PREVIEW_BUTTONS,
                            obs=preview_initial_obs,
                            info=preview_initial_info,
                            extra_lines=[
                                f"当前任务：{task_description}",
                                f"下一条待采集示范：demo_{next_demo_index}",
                                f"当前预览示范：demo_{preview_demo_index}",
                                f"当前任务完成进度：{len(completed_demo_indices)} / {len(orig_data.keys())}",
                            ],
                            update_hz=DEFAULT_MENU_MONITOR_HZ,
                        )

                        button_choice = wait_for_button_choice(
                            controller,
                            allowed_button_ids={BUTTON_COLLECT, BUTTON_REPLAY, BUTTON_EXIT},
                        )
                        if button_choice == BUTTON_EXIT:
                            raise KeyboardInterrupt("用户在预览完成页点击退出。")

                        if button_choice == BUTTON_REPLAY:
                            _, _, preview_menu_frame = play_preview_demo(
                                env_default=env_default,
                                controller=controller,
                                monitor_state=monitor_state,
                                preview_demo_data=preview_demo_data,
                                task_description=task_description,
                                preview_demo_index=preview_demo_index,
                                resolution=args.resolution,
                                control_freq=args.control_freq,
                            )
                            continue
                            # 重播完成后仍停留在本菜单循环，等待下一次选择。

                        # 只有点到“数据采集”才跳出这个菜单循环，去真正采 demo。
                        break

                    collect_result = collect_one_demo(
                        task=task,
                        demo_index=next_demo_index,
                        orig_data=orig_data,
                        env_suction=env_suction,
                        env_default=env_default,
                        controller=controller,
                        monitor_state=monitor_state,
                        args=args,
                    )

                    # 用户按了 R，则直接回到“数据采集 / 重播 / 退出”页面，不保存任何数据。
                    if collect_result["status"] == "restart":
                        continue
                        # 不写盘，直接回到菜单页，符合“重启即清空当前 demo”。

                    save_episode_to_hdf5(
                        grp,
                        demo_name=f"demo_{next_demo_index}",
                        buffers=collect_result["buffers"],
                        success=collect_result["success"],
                        handover_step=collect_result["handover_step"],
                        source_demo_index=collect_result["source_demo_index"],
                    )
                    new_data_file.flush()
                    # 立刻刷盘，减少中断时的数据丢失风险。

                    metainfo_json_dict.setdefault(task.name, {})
                    metainfo_json_dict[task.name][f"demo_{next_demo_index}"] = {
                        "success": bool(collect_result["success"]),
                        "handover_step": int(collect_result["handover_step"]),
                        "source_demo_index": int(collect_result["source_demo_index"]),
                        "demo_source": "auto_then_manual",
                        "teleop_device": "browser_websocket",
                    }
                    save_metainfo_json(metainfo_json_out_path, metainfo_json_dict)

                    print(
                        f"  已保存 {task.name} / demo_{next_demo_index}："
                        f"success={collect_result['success']}, "
                        f"handover_step={collect_result['handover_step']}"
                    )
            finally:
                env_default.close()
                env_suction.close()
                orig_data_file.close()
                new_data_file.close()
                # 无论正常结束还是异常退出，都确保环境和文件句柄被关闭。
                print(f"task '{task_description}' 当前输出文件：{target_data_path}")
    finally:
        monitor_state.shutdown_event.set()
        server.should_exit = True
        print(f"原始采集流程结束。输出目录：{target_dir}")
        print(f"metainfo JSON：{metainfo_json_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 下面定义脚本支持的所有命令行参数。
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
        help="原始 LIBERO HDF5 数据集目录。",
    )
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, help="相机图像分辨率。")
    parser.add_argument("--control_freq", type=int, default=DEFAULT_CONTROL_FREQ, help="环境控制频率。")
    parser.add_argument("--settle_steps", type=int, default=DEFAULT_SETTLE_STEPS, help="初始化后等待环境稳定的步数。")
    parser.add_argument("--manual_max_steps", type=int, default=DEFAULT_MANUAL_MAX_STEPS, help="handover 后允许的最大人工控制步数。")
    parser.add_argument("--task_limit", type=int, default=None, help="只处理前多少个 task。")
    parser.add_argument("--web_host", type=str, default="0.0.0.0", help="浏览器监看页面监听地址。")
    parser.add_argument("--web_port", type=int, default=8080, help="浏览器监看页面监听端口。")
    parser.add_argument("--movement_speed", type=float, default=0.3, help="人工模式下浏览器键盘平移速度。")
    parser.add_argument(
        "--auto_replay_speed_scale",
        type=float,
        default=DEFAULT_AUTO_REPLAY_SPEED_SCALE,
        help="点击“数据采集”后自动重播阶段的速度倍率。默认 0.5 表示自动段按 0.5x 速度回放，更容易及时按 H 切人工。",
    )
    main(parser.parse_args())
    # 解析参数并进入主流程。
