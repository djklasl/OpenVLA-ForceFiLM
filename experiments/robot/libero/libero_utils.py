"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    img = resize_image(img, resize_size)
    return img


def _load_font(size: int):
    """尽量加载中文字体，方便直接在图像上叠加六维力信息。"""
    font_candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def create_force_display(force_data, title, width=256, height=256):
    """生成一张“六维力/力矩可视化”图。"""
    force_data = np.asarray(force_data, dtype=np.float64).reshape(-1)
    if force_data.size != 6:
        force_data = np.zeros(6, dtype=np.float64)

    img = Image.new("RGB", (width, height), color=(40, 40, 50))
    draw = ImageDraw.Draw(img)

    font_title = _load_font(22)
    font_label = _load_font(16)
    font_value = _load_font(20)

    draw.text((width // 2, 12), title, fill=(255, 255, 255), font=font_title, anchor="mm")

    y_start = 45
    draw.text((80, y_start), "Force (N)", fill=(100, 200, 255), font=font_label)
    draw.text((80, y_start + 25), f"fx: {force_data[0]:8.3f}", fill=(255, 255, 255), font=font_value)
    draw.text((80, y_start + 48), f"fy: {force_data[1]:8.3f}", fill=(255, 255, 255), font=font_value)
    draw.text((80, y_start + 71), f"fz: {force_data[2]:8.3f}", fill=(255, 255, 255), font=font_value)

    y_start = 135
    draw.text((80, y_start), "Torque (Nm)", fill=(255, 180, 100), font=font_label)
    draw.text((80, y_start + 25), f"tx: {force_data[3]:8.3f}", fill=(255, 255, 255), font=font_value)
    draw.text((80, y_start + 48), f"ty: {force_data[4]:8.3f}", fill=(255, 255, 255), font=font_value)
    draw.text((80, y_start + 71), f"tz: {force_data[5]:8.3f}", fill=(255, 255, 255), font=font_value)

    return np.array(img)


def create_combined_frame(center_img, suction_wrench, right_img, img_size=256):
    """把主视角图、六维力图、手眼图拼成一张监看图。"""
    if center_img.dtype != np.uint8:
        center_img = np.clip(center_img * 255, 0, 255).astype(np.uint8)

    center_pil = Image.fromarray(center_img)
    if center_pil.size[0] != img_size or center_pil.size[1] != img_size:
        center_pil = center_pil.resize((img_size, img_size))
    center_pil = center_pil.rotate(180, expand=False)
    center_img = np.array(center_pil)

    left_img = create_force_display(suction_wrench, "接触外力(世界系)", img_size, img_size)

    if right_img.dtype != np.uint8:
        right_img = np.clip(right_img * 255, 0, 255).astype(np.uint8)
    if right_img.shape[0] != img_size or right_img.shape[1] != img_size:
        right_img = np.array(Image.fromarray(right_img).resize((img_size, img_size)))

    return np.concatenate([left_img, center_img, right_img], axis=1)


def extract_suction_wrench(obs, info):
    """优先从 obs 里读取 suction_wrench；没有时再退回 step info。"""
    if "suction_wrench" in obs and obs["suction_wrench"] is not None:
        return np.asarray(obs["suction_wrench"], dtype=np.float64)
    if info is not None and "suction_force_torque" in info and info["suction_force_torque"] is not None:
        return np.asarray(info["suction_force_torque"], dtype=np.float64)
    return np.zeros(6, dtype=np.float64)


def create_suction_wrench_history(history_length=250):
    """创建左侧补零的 suction wrench 历史窗口，形状是 [6, history_length]。"""
    return np.zeros((6, int(history_length)), dtype=np.float32)


def update_suction_wrench_history(history, suction_wrench):
    """把当前六维力/力矩追加到历史窗口最后一列，并返回更新后的 [6, T] 窗口。"""
    history = np.asarray(history, dtype=np.float32)
    suction_wrench = np.asarray(suction_wrench, dtype=np.float32).reshape(-1)
    if history.ndim != 2 or history.shape[0] != 6:
        raise ValueError(f"`history` must have shape [6, T], got {history.shape}")
    if suction_wrench.shape[0] != 6:
        raise ValueError(f"`suction_wrench` must have 6 values, got {suction_wrench.shape}")
    history = np.roll(history, shift=-1, axis=1)
    history[:, -1] = suction_wrench
    return history


def _build_rollout_video_filename(idx, success, task_description):
    """统一生成 rollout 视频文件名，避免不同写入路径各自拼字符串。"""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    return f"{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"


def open_rollout_video_writer(idx, task_description, log_file=None, subdir=None, root_dir="./rollouts", fps=30):
    """
    打开一个“边评测边写入”的视频 writer。

    这里先写到 tmp 目录，而不是直接写到 success / failure 目录。
    原因是 episode 结束前我们还不知道本条 rollout 最终是否成功。
    """
    if subdir is not None:
        rollout_dir = f"{root_dir}/{DATE}/{subdir}"
    else:
        rollout_dir = f"{root_dir}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)

    # 临时文件名里不写 success，等 episode 结束后再根据最终结果改名。
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--task=recording.mp4"
    # fps 只控制保存后视频的播放速度，不改变评测时环境 step 的控制频率。
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    if log_file is not None:
        log_file.write(f"Opened rollout MP4 writer at path {mp4_path}\n")
    return video_writer, mp4_path


def finalize_rollout_video(video_writer, tmp_path, idx, success, task_description, log_file=None, subdir=None, root_dir="./rollouts"):
    """
    关闭流式 writer，并把临时文件移动到最终 success / failure 路径。

    这样可以保证：
    1. 写视频时不需要把整条 episode 的帧全部缓存到内存里。
    2. 结束后仍然能保持原来的 success / failure 目录结构。
    """
    video_writer.close()
    if subdir is not None:
        rollout_dir = f"{root_dir}/{DATE}/{subdir}"
    else:
        rollout_dir = f"{root_dir}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    final_path = f"{rollout_dir}/{_build_rollout_video_filename(idx, success, task_description)}"
    os.replace(tmp_path, final_path)
    print(f"Saved rollout MP4 at path {final_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {final_path}\n")
    return final_path


def save_rollout_video(
    rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    subdir=None,
    root_dir="./rollouts",
    fps=30,
):
    """
    Saves an MP4 replay of an episode.

    Args:
        rollout_images: 图像序列
        idx: 回合编号
        success: 是否成功
        task_description: 任务描述
        log_file: 日志文件（可选）
        subdir: 自定义子目录名（可选）
                如果传入，视频保存到 {root_dir}/{DATE}/{subdir}/
                如果不传，视频保存到 {root_dir}/{DATE}/
        root_dir: 视频根目录，默认是 ./rollouts
    """
    if subdir is not None:
        rollout_dir = f"{root_dir}/{DATE}/{subdir}"
    else:
        rollout_dir = f"{root_dir}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = f"{rollout_dir}/{_build_rollout_video_filename(idx, success, task_description)}"
    # fps 只控制保存后视频的播放速度；帧数量仍由调用方写入多少张图决定。
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
