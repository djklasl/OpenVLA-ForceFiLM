"""并发启动 4 个 suction eval worker，并在 launcher 主进程统一汇总日志与 W&B。"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# =========================
# 评测进程 CPU 线程默认限制
# =========================
# setdefault 的意思是：如果外部启动命令已经设置过这个环境变量，就尊重外部设置；
# 如果外部没有设置，才使用这里给出的默认值。这样既能给日常评测一个稳妥默认值，
# 又保留以后手动调参对比的入口。
#
# OMP_NUM_THREADS 控制 OpenMP 相关底层库的线程数。LIBERO / MuJoCo 单个仿真进程通常不需要很多线程，
# 这里默认给 2 个线程，适配“一张 GPU 挂两个 worker”的评测方式。
os.environ.setdefault("OMP_NUM_THREADS", "2")
# MKL_NUM_THREADS 控制 Intel MKL 的 CPU 线性代数线程数。评测时主要瓶颈通常不是 CPU 大矩阵计算，
# 设为 1 可以避免每个 worker 额外开很多 MKL 线程互相抢 CPU。
os.environ.setdefault("MKL_NUM_THREADS", "1")
# OPENBLAS_NUM_THREADS 控制 OpenBLAS 的线程数。不同环境里的 NumPy 可能链接 MKL 或 OpenBLAS，
# 两者都设置可以让 CPU 线程行为更可控。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# NUMEXPR_NUM_THREADS 控制 NumExpr 的线程数。部分数组处理库可能间接使用它，
# 这里同样限制为 1，避免评测 worker 额外开线程。
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR))

import wandb

from experiments.robot.robot_utils import DATE_TIME

# =========================
# 模型与动作反归一化配置
# =========================
# 当前评测使用的模型家族；suction 版 OpenVLA 这里保持为 openvla。
MODEL_FAMILY = "openvla"
# 要评测的 merged checkpoint 路径；后续换权重时只需要改这里。
PRETRAINED_CHECKPOINT = "runs_libero_suction_wrench_history_250_primus_trainable_rank256_film_lora/primus_trainable_rank256_film_lora+110000"
# 是否用 8bit 加载模型；通常评测保持 False。
LOAD_IN_8BIT = False
# 是否用 4bit 加载模型；通常评测保持 False。
LOAD_IN_4BIT = False
# 输入图像是否做中心裁剪；应与训练权重的预处理习惯保持一致。
CENTER_CROP = True
# 是否启用训练好的 ForceFiLMGenerator，把 suction_wrench 作为连续条件调制视觉特征。
USE_FORCE_FILM = True
# LoRA+ForceFiLM 联合训练的 merged 目录；目录内已经保存 force_film_generator.pt。
# 评测加载完整模型后，会再从这个目录读取 ForceFiLMGenerator 的 110000 step 参数。
FORCE_FILM_CHECKPOINT = PRETRAINED_CHECKPOINT
FORCE_ENCODER_EMBEDDING_TYPE = "mmcl"
FORCE_HISTORY_LENGTH = 250
# 是否在第一次接触物体之前，把输入给 VLA / Force-FiLM 的六维力历史强制替换成固定常数。
# - False：模型看到真实的 suction_wrench_history，保持正常 Force-FiLM 推理。
# - True：在第一次 Fz != 0.0 之前，模型看到全 20.0 的六维力历史；第一次接触后立刻恢复真实力历史。
# 注意：这个开关只影响“传给模型的力输入”，不会改 CSV、吸附失败检测、视频显示和环境里的真实六维力。
CONSTANT_FORCE_FILM_INPUT_BEFORE_FIRST_CONTACT = True
# 动作反归一化时使用的统计量 key；必须和 joint training 使用的 RLDS 数据集名严格一致。
UNNORM_KEY = "processed_libero_suction_wrench_history_250"

# =========================
# LIBERO 任务与环境配置
# =========================
# 要评测的 LIBERO 任务套件名称。
# baseline 现在针对的是自定义的 suction 任务集合，而不是原始 libero_object。
TASK_SUITE_NAME = "libero_suction"
# 当前环境使用 suction 吸盘末端执行器。
GRIPPER_TYPE = "suction"
# 每条 episode 开始前先执行多少步 dummy action，让环境先稳定下来。
NUM_STEPS_WAIT = 10
# 每条 episode 正式推理允许执行的最大动作步数。
MAX_STEPS = 400
# 每个 task 要评测多少条 episode；LIBERO 常用 50。
NUM_TRIALS = 50
# 相机图像分辨率；主视角、手眼图和力图拼接都按这个尺寸渲染。
RESOLUTION = 256
# 控制频率；需要与当前 suction 仿真设置保持一致。
CONTROL_FREQ = 20

# =========================
# 输出与运行标识配置
# =========================
# 是否保存所有 rollout 视频。
# - False：完全不保存任何 rollout 视频
# - True：保存所有 rollout 视频（成功和失败都会保存）
SAVE_ALL_ROLLOUT_VIDEOS = True
# rollout 视频播放帧率。这里设置为 1，表示保存后的视频每秒播放 1 个控制时刻画面。
# 注意：它只影响视频播放速度，不改变 CONTROL_FREQ，也不改变仿真控制频率。
ROLLOUT_VIDEO_FPS = 30
# 是否在保存 rollout 视频的同时，保存同名 CSV。
# CSV 里导出 episode 结束时的 250 长度 suction wrench 窗口和同步 action 窗口，
# 用于把六维力/力矩与同一控制时刻执行的动作放在一起分析。
SAVE_ACTION_WRENCH_HISTORY_CSV = True
# 是否启用“吸盘接触过物体、但当前空吸上移”的“吸附失败”状态检测。
ENABLE_SUCTION_FAILURE_DETECTION = True
# Fz 大于这个阈值时，认为吸盘在当前控制时刻正向挤压/接触过物体。
SUCTION_FAILURE_CONTACT_FZ_THRESHOLD = 0.0
# 当前 Fz 的绝对值小于等于这个阈值时，认为当前没有接触、也没有吸附负载。
SUCTION_FAILURE_FREE_FZ_THRESHOLD = 1e-6
# 当前 action 第三维 z 方向大于这个阈值时，认为策略正在让吸盘向上离开。
SUCTION_FAILURE_LIFT_ACTION_Z_THRESHOLD = 0.0
# 只在最近多少个控制时刻内统计正向接触，避免很久以前的接触影响当前判断。
SUCTION_FAILURE_CONTACT_HISTORY_STEPS = 2
# 最近历史里至少出现多少个正向接触点，才认为真的发生过吸附尝试，而不是单点噪声。
SUCTION_FAILURE_MIN_CONTACT_STEPS = 1
# 是否在保存的视频中把检测到的吸附失败用红色中文标出来。
SHOW_SUCTION_FAILURE_OVERLAY_IN_VIDEO = True
# - True：正式推理阶段里，只要当前 Fz>0，就强制 VLA action 的最后一维 action_suction=1；
# - False：完全尊重 VLA 原始输出，不额外修改吸盘维度。
# 注意：这个规则只作用于 VLA 新生成的 action，不作用于等待阶段 dummy action，也不作用于回溯队列 action。
ENABLE_CONTACT_FORCE_SUCTION_OVERRIDE = False
# 检测到吸附失败后，是否用历史 action 求和取反生成单步回溯 action。
ENABLE_SUCTION_FAILURE_REWIND = False
# 找到最近接触段起点后，再往前回退多少个控制时刻作为回溯点。
SUCTION_REWIND_PRE_CONTACT_STEPS = 20
# 是否在吸附失败回溯动作结束后，追加一个由失败次数驱动的确定性 x/y/z 扰动。
# 这个扰动不负责吸附，只负责让末端离开之前的失败轨迹，再把控制权交还给 VLA。
ENABLE_SUCTION_FAILURE_ESCAPE_PERTURB = False
# 扰动幅值步长；实际幅值 = 当前 episode 内吸附失败次数 * 这个值。
SUCTION_ESCAPE_PERTURB_SCALE_STEP = 0.1
# 可选的 run 备注；这里标记本轮是“更严格吸盘吸附规则”的重测。
# 这个备注会拼进 shared_run_id，让本地日志目录、视频目录和 W&B run name 都能区分旧评测与新评测。
RUN_ID_NOTE = "strict_suction_rule_取消掉吸盘吸附对相对速度要求和吸盘滑动关节因开关而移动"
# 本地文本日志目录；launcher 总日志和 worker 日志都写到这里。
LOCAL_LOG_DIR = "./experiments/logs"
# 本地视频根目录；worker 会继续在这里按“成功/失败 + 是否出现吸附失败”四分类保存视频。
LOCAL_VIDEO_DIR = "./infer_六维力_第一次接触前输入置20，接触后恢复正常力-吸附失败检测"
# 随机种子；用于保持评测过程可复现。
SEED = 7
# 是否从一次历史并发评测中恢复。
# - None：表示新开一轮全新的并发评测
# - 非 None：表示显式指定某个 shared_run_id，从它留下的 worker 日志 / progress 继续跑
RESUME_SHARED_RUN_ID = None
# 当历史运行还没有结构化 progress.json 时，是否回退解析旧 worker 文本日志。
# 为了兼容你之前已经跑到一半的评测，这里默认打开。
ALLOW_TEXT_LOG_RESUME = True

# =========================
# W&B 汇总配置
# =========================
# 是否由 launcher 在最后统一上传汇总结果到 W&B。
USE_WANDB = False
# W&B project 名称；这里只上传一张 task 成功率表和一个 suite 平均成功率。
WANDB_PROJECT = "runs_libero_suction_wrench_history_250_base_pretrained_ckpt_original_openvla--110000"
# W&B entity 名称。
WANDB_ENTITY = "speng0442-huazhong-university-of-science-and-technology"
# W&B 结果 key 前缀；不改 project，只把本轮重测结果写到新的 key 空间，避免覆盖旧的 success_rate/* 图表和表格。
WANDB_RESULT_PREFIX = "strict_suction_rule_取消掉吸盘吸附对相对速度要求"
# worker 是否各自上传 W&B；并发评测建议保持 False，只让 launcher 汇总上传。
WORKER_USE_WANDB = False

# =========================
# 并发 worker / GPU / task 分配
# =========================
# 固定 8 个 worker，并按你的要求绑定到 GPU 2、3、4、5。
# 这里不是简单按 task 顺序平均切分，而是根据 experiments/logs 里最近一次完整评测记录做负载均衡：
# - 每个 task 都跑了 50 条 episode，所以把这 50 条 episode 的 `Steps executed` 求和，当作该 task 的耗时权重；
# - 先让单个 worker 的总 step 尽量接近理论平均值，减少所有 worker 里最慢那个 worker 的尾部等待；
# - 再把两个 worker 组成一组放到同一张 GPU 上，让每张 GPU 的总 step 也尽量接近，避免某一张 GPU 明显拖慢总结束时间。
# 当前这组分配对应的 worker step 估计大致为：
# worker00=39671, worker01=42081, worker02=41859, worker03=42073,
# worker04=41945, worker05=42036, worker06=41967, worker07=42033。
GPU_TASK_LAYOUT = [
    {"worker_id": 0, "gpu_id": "2", "task_orders": [0, 1]},
    {"worker_id": 1, "gpu_id": "2", "task_orders": [11, 17, 19]},
    {"worker_id": 2, "gpu_id": "3", "task_orders": [6, 7, 12]},
    {"worker_id": 3, "gpu_id": "3", "task_orders": [2, 20, 22]},
    {"worker_id": 4, "gpu_id": "4", "task_orders": [4, 13, 15]},
    {"worker_id": 5, "gpu_id": "4", "task_orders": [8, 9, 14]},
    {"worker_id": 6, "gpu_id": "5", "task_orders": [10, 16, 18]},
    {"worker_id": 7, "gpu_id": "5", "task_orders": [3, 5, 21]},
]


def build_launcher_config():
    return SimpleNamespace(
        model_family=MODEL_FAMILY,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        load_in_8bit=LOAD_IN_8BIT,
        load_in_4bit=LOAD_IN_4BIT,
        center_crop=CENTER_CROP,
        use_force_film=USE_FORCE_FILM,
        force_film_checkpoint=FORCE_FILM_CHECKPOINT,
        force_encoder_embedding_type=FORCE_ENCODER_EMBEDDING_TYPE,
        force_history_length=FORCE_HISTORY_LENGTH,
        constant_force_film_input_before_first_contact=CONSTANT_FORCE_FILM_INPUT_BEFORE_FIRST_CONTACT,
        unnorm_key=UNNORM_KEY,
        task_suite_name=TASK_SUITE_NAME,
        gripper_type=GRIPPER_TYPE,
        num_steps_wait=NUM_STEPS_WAIT,
        max_steps=MAX_STEPS,
        num_trials=NUM_TRIALS,
        resolution=RESOLUTION,
        control_freq=CONTROL_FREQ,
        save_all_rollout_videos=SAVE_ALL_ROLLOUT_VIDEOS,
        rollout_video_fps=ROLLOUT_VIDEO_FPS,
        save_action_wrench_history_csv=SAVE_ACTION_WRENCH_HISTORY_CSV,
        enable_suction_failure_detection=ENABLE_SUCTION_FAILURE_DETECTION,
        suction_failure_contact_fz_threshold=SUCTION_FAILURE_CONTACT_FZ_THRESHOLD,
        suction_failure_free_fz_threshold=SUCTION_FAILURE_FREE_FZ_THRESHOLD,
        suction_failure_lift_action_z_threshold=SUCTION_FAILURE_LIFT_ACTION_Z_THRESHOLD,
        suction_failure_contact_history_steps=SUCTION_FAILURE_CONTACT_HISTORY_STEPS,
        suction_failure_min_contact_steps=SUCTION_FAILURE_MIN_CONTACT_STEPS,
        show_suction_failure_overlay_in_video=SHOW_SUCTION_FAILURE_OVERLAY_IN_VIDEO,
        enable_contact_force_suction_override=ENABLE_CONTACT_FORCE_SUCTION_OVERRIDE,
        enable_suction_failure_rewind=ENABLE_SUCTION_FAILURE_REWIND,
        suction_rewind_pre_contact_steps=SUCTION_REWIND_PRE_CONTACT_STEPS,
        enable_suction_failure_escape_perturb=ENABLE_SUCTION_FAILURE_ESCAPE_PERTURB,
        suction_escape_perturb_scale_step=SUCTION_ESCAPE_PERTURB_SCALE_STEP,
        run_id_note=RUN_ID_NOTE,
        local_log_dir=LOCAL_LOG_DIR,
        local_video_dir=LOCAL_VIDEO_DIR,
        seed=SEED,
        resume_shared_run_id=RESUME_SHARED_RUN_ID,
        allow_text_log_resume=ALLOW_TEXT_LOG_RESUME,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        wandb_result_prefix=WANDB_RESULT_PREFIX,
        worker_use_wandb=WORKER_USE_WANDB,
        worker_specs=GPU_TASK_LAYOUT,
    )


def bool_to_cli(value: bool) -> str:
    return "true" if value else "false"


def build_shared_run_id(cfg) -> str:
    run_id = f"EVAL-suite-{cfg.task_suite_name}-{cfg.model_family}-{cfg.gripper_type}-parallel-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


def write_total_log(log_path: Path, lines: list[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n")


def read_json_file(json_path: Path) -> dict:
    """读取一个 JSON 文件。"""
    return json.loads(json_path.read_text())


def build_worker_artifact_paths(shared_run_id: str, worker_id: int, log_dir: Path) -> dict[str, Path | str]:
    """统一生成某个 worker 在当前 shared_run_id 下的所有文件路径。"""
    worker_name = f"worker{worker_id:02d}"
    return {
        "summary_json_path": log_dir / f"{shared_run_id}--{worker_name}.summary.json",
        "worker_log_path": log_dir / f"{shared_run_id}--{worker_name}.txt",
        "progress_json_path": log_dir / f"{shared_run_id}--{worker_name}.progress.json",
        "resume_state_json_path": log_dir / f"{shared_run_id}--{worker_name}.resume_state.json",
        "worker_video_subdir": f"{shared_run_id}/{worker_name}",
    }


def build_task_stats_entry(
    *,
    task_id: int,
    task_name: str,
    episodes: int,
    successes: int,
    task_bddl_path: str,
) -> dict:
    """统一构造 task 统计字典，保证 launcher / worker / summary 形状一致。"""
    success_rate = float(successes) / float(episodes) if episodes else 0.0
    return {
        "task_id": task_id,
        "task_name": task_name,
        "episodes": int(episodes),
        "successes": int(successes),
        "success_rate": success_rate,
        "task_bddl_path": task_bddl_path,
    }


def infer_completed_task_ids(cfg, spec: dict, resume_state: dict) -> set[int]:
    """统一根据恢复状态推断哪些 task 已经完整完成。

    这里不再只信任 `completed_task_ids`：
    - 老日志 / progress.json 可能先把 `episodes` 刷到了 `num_trials`
    - 但程序还没来得及把 task 写进 `completed_task_ids` 就中断
    这种情况下，`episodes >= num_trials` 也必须被视为已完成，否则最终汇总会漏 task。
    """
    completed_task_ids = {int(task_id) for task_id in resume_state.get("completed_task_ids", [])}
    task_stats = resume_state.get("task_stats", {})
    next_episode_idx_by_task = resume_state.get("next_episode_idx_by_task", {})

    for task_id in spec["task_orders"]:
        task_id_str = str(task_id)
        task_stat = task_stats.get(task_id_str, {})
        if int(task_stat.get("episodes", 0)) >= cfg.num_trials:
            completed_task_ids.add(task_id)
            continue
        if int(next_episode_idx_by_task.get(task_id_str, 0)) >= cfg.num_trials:
            completed_task_ids.add(task_id)

    return completed_task_ids


def build_empty_resume_state(cfg, spec: dict, artifact_paths: dict[str, Path | str], *, resume_source: str) -> dict:
    """构造一个“从头开始”的标准化 worker 恢复状态。"""
    return {
        "shared_run_id": cfg.resume_shared_run_id,
        "worker_id": spec["worker_id"],
        "task_orders": list(spec["task_orders"]),
        "checkpoint": str(cfg.pretrained_checkpoint),
        "use_force_film": bool(cfg.use_force_film),
        "force_film_checkpoint": cfg.force_film_checkpoint,
        "force_encoder_embedding_type": cfg.force_encoder_embedding_type,
        "force_history_length": cfg.force_history_length,
        "constant_force_film_input_before_first_contact": bool(
            cfg.constant_force_film_input_before_first_contact
        ),
        "resume_source": resume_source,
        "completed": False,
        "completed_task_ids": [],
        "current_task_id": None,
        "next_episode_idx_by_task": {str(task_id): 0 for task_id in spec["task_orders"]},
        "task_stats": {},
        "total_episodes": 0,
        "total_successes": 0,
        "total_rollout_videos": 0,
        "total_success_videos": 0,
        "total_failed_videos": 0,
        "local_log_filepath": str(artifact_paths["worker_log_path"]),
        "local_video_dir": cfg.local_video_dir,
        "worker_video_subdir": artifact_paths["worker_video_subdir"],
        "error": None,
    }


def build_resume_state_from_summary(
    cfg,
    spec: dict,
    artifact_paths: dict[str, Path | str],
    summary_payload: dict,
    *,
    resume_source: str,
) -> dict:
    """把 summary.json 转成统一的 resume_state 结构。"""
    task_stats = {
        str(task_result["task_id"]): dict(task_result) for task_result in summary_payload.get("task_results", [])
    }
    completed_task_ids = [int(task_id) for task_id in task_stats]
    next_episode_idx_by_task = {str(task_id): 0 for task_id in spec["task_orders"]}
    for task_id in completed_task_ids:
        next_episode_idx_by_task[str(task_id)] = cfg.num_trials

    return {
        "shared_run_id": summary_payload.get("shared_run_id", cfg.resume_shared_run_id),
        "worker_id": summary_payload.get("worker_id", spec["worker_id"]),
        "task_orders": summary_payload.get("task_orders", list(spec["task_orders"])),
        "checkpoint": summary_payload.get("checkpoint", str(cfg.pretrained_checkpoint)),
        "use_force_film": bool(summary_payload.get("use_force_film", False)),
        "force_film_checkpoint": summary_payload.get("force_film_checkpoint"),
        "force_encoder_embedding_type": summary_payload.get("force_encoder_embedding_type", cfg.force_encoder_embedding_type),
        "force_history_length": int(summary_payload.get("force_history_length", cfg.force_history_length)),
        "constant_force_film_input_before_first_contact": bool(
            summary_payload.get(
                "constant_force_film_input_before_first_contact",
                cfg.constant_force_film_input_before_first_contact,
            )
        ),
        "resume_source": resume_source,
        "completed": bool(summary_payload.get("completed", False)),
        "completed_task_ids": completed_task_ids,
        "current_task_id": None,
        "next_episode_idx_by_task": next_episode_idx_by_task,
        "task_stats": task_stats,
        "total_episodes": int(summary_payload.get("total_episodes", 0)),
        "total_successes": int(summary_payload.get("total_successes", 0)),
        "total_rollout_videos": int(summary_payload.get("total_rollout_videos", 0)),
        "total_success_videos": int(summary_payload.get("total_success_videos", 0)),
        "total_failed_videos": int(summary_payload.get("total_failed_videos", 0)),
        "local_log_filepath": summary_payload.get("local_log_filepath", str(artifact_paths["worker_log_path"])),
        "local_video_dir": summary_payload.get("local_video_dir", cfg.local_video_dir),
        "worker_video_subdir": summary_payload.get("worker_video_subdir", artifact_paths["worker_video_subdir"]),
        "error": summary_payload.get("error"),
    }


def parse_worker_log_for_resume(
    cfg,
    spec: dict,
    artifact_paths: dict[str, Path | str],
) -> dict:
    """从旧 worker 文本日志尽量恢复出“做到哪个 task / episode”。

    恢复策略：
    - 已完整结束的 task：直接标记 completed；
    - 最后一个进行中的 task：从“最后一个可能未完成的 episode”重新跑；
    - 不尝试恢复 episode 中间 step。
    """
    log_path = Path(artifact_paths["worker_log_path"])
    resume_state = build_empty_resume_state(cfg, spec, artifact_paths, resume_source="worker_txt")
    if not log_path.exists():
        return resume_state

    selected_task_orders: list[int] | None = None
    current_task_id: int | None = None
    current_task_name: str | None = None
    current_task_bddl_path: str = ""
    task_stats: dict[str, dict] = {}
    completed_task_ids: set[int] = set()

    selected_task_orders_pattern = re.compile(r"^Selected task orders:\s*(.+)$")
    task_line_pattern = re.compile(r"^Task\s+(\d+):\s*(.+)$")
    start_episode_pattern = re.compile(r"^Starting episode\s+(\d+)/(\d+)\s+with init state\s+\d+\.\.\.$")
    task_episodes_pattern = re.compile(r"^# 当前任务 episodes:\s*(\d+)$")
    task_successes_pattern = re.compile(r"^# 当前任务 successes:\s*(\d+)")
    total_episodes_pattern = re.compile(r"^# 套件累计 episodes:\s*(\d+)$")
    total_successes_pattern = re.compile(r"^# 套件累计 successes:\s*(\d+)")
    checkpoint_pattern = re.compile(r"^Checkpoint:\s*(.+)$")
    use_force_film_pattern = re.compile(r"^Use force FiLM:\s*(.+)$")
    force_checkpoint_pattern = re.compile(r"^Force-FiLM checkpoint:\s*(.+)$")
    force_encoder_embedding_pattern = re.compile(r"^Force-FiLM encoder embedding type:\s*(.+)$")
    force_history_length_pattern = re.compile(r"^Force-FiLM history length:\s*(.+)$")
    constant_force_film_input_before_first_contact_pattern = re.compile(
        r"^Constant Force-FiLM input before first contact:\s*(.+)$"
    )
    bddl_pattern = re.compile(r"^Task BDDL:\s*(.+)$")

    for raw_line in log_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        selected_match = selected_task_orders_pattern.match(line)
        if selected_match:
            try:
                selected_task_orders = [int(task_id) for task_id in ast.literal_eval(selected_match.group(1))]
            except Exception:
                selected_task_orders = None
            continue

        checkpoint_match = checkpoint_pattern.match(line)
        if checkpoint_match:
            resume_state["checkpoint"] = checkpoint_match.group(1).strip()
            continue

        use_force_film_match = use_force_film_pattern.match(line)
        if use_force_film_match:
            resume_state["use_force_film"] = use_force_film_match.group(1).strip().lower() == "true"
            continue

        force_checkpoint_match = force_checkpoint_pattern.match(line)
        if force_checkpoint_match:
            resume_state["force_film_checkpoint"] = force_checkpoint_match.group(1).strip()
            continue

        force_encoder_embedding_match = force_encoder_embedding_pattern.match(line)
        if force_encoder_embedding_match:
            resume_state["force_encoder_embedding_type"] = force_encoder_embedding_match.group(1).strip()
            continue

        force_history_length_match = force_history_length_pattern.match(line)
        if force_history_length_match:
            resume_state["force_history_length"] = int(force_history_length_match.group(1).strip())
            continue

        constant_force_film_input_before_first_contact_match = (
            constant_force_film_input_before_first_contact_pattern.match(line)
        )
        if constant_force_film_input_before_first_contact_match:
            resume_state["constant_force_film_input_before_first_contact"] = (
                constant_force_film_input_before_first_contact_match.group(1).strip().lower() == "true"
            )
            continue

        task_match = task_line_pattern.match(line)
        if task_match:
            current_task_id = int(task_match.group(1))
            current_task_name = task_match.group(2).strip()
            task_stats.setdefault(
                str(current_task_id),
                build_task_stats_entry(
                    task_id=current_task_id,
                    task_name=current_task_name,
                    episodes=0,
                    successes=0,
                    task_bddl_path="",
                ),
            )
            continue

        bddl_match = bddl_pattern.match(line)
        if bddl_match and current_task_id is not None:
            current_task_bddl_path = bddl_match.group(1).strip()
            task_stats[str(current_task_id)]["task_bddl_path"] = current_task_bddl_path
            continue

        start_episode_match = start_episode_pattern.match(line)
        if start_episode_match and current_task_id is not None:
            # 这里记录“当前开始跑了哪个 episode”。
            # 如果后面没有看到该 episode 完整结束的统计更新，就说明这条 episode 最多只跑到一半；
            # 恢复时就应该从这个 episode 重新跑。
            resume_state["current_task_id"] = current_task_id
            resume_state["next_episode_idx_by_task"][str(current_task_id)] = int(start_episode_match.group(1)) - 1
            continue

        task_episodes_match = task_episodes_pattern.match(line)
        if task_episodes_match and current_task_id is not None:
            episodes = int(task_episodes_match.group(1))
            task_stats.setdefault(
                str(current_task_id),
                build_task_stats_entry(
                    task_id=current_task_id,
                    task_name=current_task_name or f"task_{current_task_id:02d}",
                    episodes=0,
                    successes=0,
                    task_bddl_path=current_task_bddl_path,
                ),
            )
            task_stats[str(current_task_id)]["episodes"] = max(int(task_stats[str(current_task_id)]["episodes"]), episodes)
            task_stats[str(current_task_id)]["success_rate"] = (
                float(task_stats[str(current_task_id)]["successes"]) / float(episodes) if episodes else 0.0
            )
            # 一旦看到了“当前任务 episodes = K”，就说明前 K 条 episode 都已经完整结束；
            # 所以下次应从第 K+1 条开始，也就是 0-based 的 episode_idx=K。
            resume_state["next_episode_idx_by_task"][str(current_task_id)] = episodes
            if episodes >= cfg.num_trials:
                completed_task_ids.add(current_task_id)
            continue

        task_successes_match = task_successes_pattern.match(line)
        if task_successes_match and current_task_id is not None:
            successes = int(task_successes_match.group(1))
            task_stats.setdefault(
                str(current_task_id),
                build_task_stats_entry(
                    task_id=current_task_id,
                    task_name=current_task_name or f"task_{current_task_id:02d}",
                    episodes=0,
                    successes=0,
                    task_bddl_path=current_task_bddl_path,
                ),
            )
            task_stats[str(current_task_id)]["successes"] = max(int(task_stats[str(current_task_id)]["successes"]), successes)
            episodes = int(task_stats[str(current_task_id)]["episodes"])
            task_stats[str(current_task_id)]["success_rate"] = float(successes) / float(episodes) if episodes else 0.0
            continue

        total_episodes_match = total_episodes_pattern.match(line)
        if total_episodes_match:
            resume_state["total_episodes"] = max(int(resume_state["total_episodes"]), int(total_episodes_match.group(1)))
            continue

        total_successes_match = total_successes_pattern.match(line)
        if total_successes_match:
            resume_state["total_successes"] = max(
                int(resume_state["total_successes"]),
                int(total_successes_match.group(1)),
            )
            continue

        if line.startswith("Current task success rate:") and current_task_id is not None:
            completed_task_ids.add(current_task_id)
            resume_state["next_episode_idx_by_task"][str(current_task_id)] = max(
                int(resume_state["next_episode_idx_by_task"][str(current_task_id)]),
                int(task_stats.get(str(current_task_id), {}).get("episodes", 0)),
            )
            if int(task_stats.get(str(current_task_id), {}).get("episodes", 0)) >= cfg.num_trials:
                resume_state["next_episode_idx_by_task"][str(current_task_id)] = cfg.num_trials
            continue

        if line.startswith("Final suite success rate:"):
            resume_state["completed"] = True

    if selected_task_orders is not None:
        resume_state["task_orders"] = selected_task_orders
    resume_state["task_stats"] = task_stats
    resume_state["completed_task_ids"] = sorted(completed_task_ids)

    # 如果最后一个 task 在日志里已经把 episode 数刷到了 `num_trials`，即使没来得及写 summary，
    # 这里也把它视作已完成，避免无意义重复。
    for task_id_str, task_stat in task_stats.items():
        if int(task_stat.get("episodes", 0)) >= cfg.num_trials:
            completed_task_ids.add(int(task_id_str))
            resume_state["next_episode_idx_by_task"][task_id_str] = cfg.num_trials

    resume_state["completed_task_ids"] = sorted(completed_task_ids)
    if all(int(resume_state["next_episode_idx_by_task"].get(str(task_id), 0)) >= cfg.num_trials for task_id in spec["task_orders"]):
        resume_state["completed"] = True
        resume_state["current_task_id"] = None

    return resume_state


def load_existing_worker_resume_state(cfg, spec: dict, artifact_paths: dict[str, Path | str]) -> dict:
    """为某个 worker 构造统一的恢复状态。

    优先级：
    1. progress.json：最可靠，包含结构化实时进度；
    2. 已完成的 summary.json：说明该 worker 无需继续跑；
    3. 旧 worker txt：兼容历史运行；
    4. fresh：表示没有可恢复状态，从头跑该 worker 分配的 task。
    """
    progress_json_path = Path(artifact_paths["progress_json_path"])
    summary_json_path = Path(artifact_paths["summary_json_path"])

    if progress_json_path.exists():
        progress_payload = read_json_file(progress_json_path)
        progress_payload["resume_source"] = "progress_json"
        return progress_payload

    if summary_json_path.exists():
        summary_payload = read_json_file(summary_json_path)
        if bool(summary_payload.get("completed", False)):
            return build_resume_state_from_summary(
                cfg,
                spec,
                artifact_paths,
                summary_payload,
                resume_source="summary_json",
            )

    if cfg.allow_text_log_resume:
        text_resume_state = parse_worker_log_for_resume(cfg, spec, artifact_paths)
        if text_resume_state["task_stats"] or Path(artifact_paths["worker_log_path"]).exists():
            return text_resume_state

    if summary_json_path.exists():
        summary_payload = read_json_file(summary_json_path)
        return build_resume_state_from_summary(
            cfg,
            spec,
            artifact_paths,
            summary_payload,
            resume_source="summary_json",
        )

    return build_empty_resume_state(cfg, spec, artifact_paths, resume_source="fresh")


def validate_resume_state_checkpoint(cfg, resume_state: dict, *, shared_run_id: str, worker_id: int) -> None:
    """检查恢复状态里的 base / force checkpoint 是否和本次 launcher 配置一致。"""
    existing_checkpoint = resume_state.get("checkpoint")
    if existing_checkpoint is not None and str(existing_checkpoint) != str(cfg.pretrained_checkpoint):
        raise ValueError(
            f"Checkpoint mismatch for {shared_run_id} worker{worker_id:02d}: "
            f"resume checkpoint = {existing_checkpoint}, current checkpoint = {cfg.pretrained_checkpoint}"
        )
    existing_use_force_film = bool(resume_state.get("use_force_film", False))
    if existing_use_force_film != bool(cfg.use_force_film):
        raise ValueError(
            f"Force-FiLM enable mismatch for {shared_run_id} worker{worker_id:02d}: "
            f"resume use_force_film = {existing_use_force_film}, current use_force_film = {cfg.use_force_film}"
        )
    if cfg.use_force_film:
        existing_force_checkpoint = resume_state.get("force_film_checkpoint")
        if str(existing_force_checkpoint) != str(cfg.force_film_checkpoint):
            raise ValueError(
                f"Force-FiLM checkpoint mismatch for {shared_run_id} worker{worker_id:02d}: "
                f"resume force checkpoint = {existing_force_checkpoint}, "
                f"current force checkpoint = {cfg.force_film_checkpoint}"
            )
        expected_force_params = {
            "force_encoder_embedding_type": cfg.force_encoder_embedding_type,
            "force_history_length": cfg.force_history_length,
            "constant_force_film_input_before_first_contact": bool(
                cfg.constant_force_film_input_before_first_contact
            ),
        }
        for field_name, expected_value in expected_force_params.items():
            existing_value = resume_state.get(field_name, expected_value)
            if str(existing_value) != str(expected_value):
                raise ValueError(
                    f"Force-FiLM {field_name} mismatch for {shared_run_id} worker{worker_id:02d}: "
                    f"resume = {existing_value}, current = {expected_value}"
                )


def worker_has_remaining_work(cfg, spec: dict, resume_state: dict) -> bool:
    """判断这个 worker 是否还有未完成的 task / episode。"""
    completed_task_ids = infer_completed_task_ids(cfg, spec, resume_state)
    next_episode_idx_by_task = {
        int(task_id): int(episode_idx)
        for task_id, episode_idx in resume_state.get("next_episode_idx_by_task", {}).items()
    }
    for task_id in spec["task_orders"]:
        if task_id in completed_task_ids:
            continue
        if next_episode_idx_by_task.get(task_id, 0) < cfg.num_trials:
            return True
    return False


def build_summary_from_resume_state(cfg, spec: dict, resume_state: dict, artifact_paths: dict[str, Path | str]) -> dict:
    """当某个 worker 不需要重新拉起时，用恢复状态直接合成 summary 参与总汇总。"""
    task_orders = [int(task_id) for task_id in resume_state.get("task_orders", spec["task_orders"])]
    task_stats = {
        int(task_id): dict(task_stat) for task_id, task_stat in resume_state.get("task_stats", {}).items()
    }
    completed_task_ids = infer_completed_task_ids(cfg, {"task_orders": task_orders}, resume_state)
    total_episodes = int(resume_state.get("total_episodes", 0))
    total_successes = int(resume_state.get("total_successes", 0))
    suite_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0

    return {
        "run_id": f"{resume_state.get('shared_run_id')}--worker{int(resume_state.get('worker_id', 0)):02d}",
        "shared_run_id": resume_state.get("shared_run_id"),
        "worker_id": resume_state.get("worker_id"),
        "task_orders": task_orders,
        "checkpoint": resume_state.get("checkpoint"),
        "use_force_film": bool(resume_state.get("use_force_film", False)),
        "force_film_checkpoint": resume_state.get("force_film_checkpoint"),
        "force_encoder_embedding_type": resume_state.get("force_encoder_embedding_type", cfg.force_encoder_embedding_type),
        "force_history_length": int(resume_state.get("force_history_length", cfg.force_history_length)),
        "constant_force_film_input_before_first_contact": bool(
            resume_state.get(
                "constant_force_film_input_before_first_contact",
                cfg.constant_force_film_input_before_first_contact,
            )
        ),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "suite_success_rate": suite_success_rate,
        "total_rollout_videos": int(resume_state.get("total_rollout_videos", 0)),
        "total_success_videos": int(resume_state.get("total_success_videos", 0)),
        "total_failed_videos": int(resume_state.get("total_failed_videos", 0)),
        "task_results": [dict(task_stats[task_id]) for task_id in sorted(completed_task_ids) if task_id in task_stats],
        "local_log_filepath": resume_state.get("local_log_filepath", str(artifact_paths["worker_log_path"])),
        "local_video_dir": resume_state.get("local_video_dir"),
        "worker_video_subdir": resume_state.get("worker_video_subdir"),
        "completed": bool(resume_state.get("completed", False)),
        "error": resume_state.get("error"),
    }


def launch_parallel_eval() -> None:
    cfg = build_launcher_config()
    if cfg.use_force_film:
        force_ckpt_path = Path(cfg.force_film_checkpoint)
        force_ckpt_file = force_ckpt_path / "force_film_generator.pt" if force_ckpt_path.is_dir() else force_ckpt_path
        if not force_ckpt_file.is_file():
            raise FileNotFoundError(f"Force-FiLM checkpoint not found: {force_ckpt_file}")

    is_resume_mode = cfg.resume_shared_run_id is not None
    shared_run_id = cfg.resume_shared_run_id if is_resume_mode else build_shared_run_id(cfg)
    log_dir = Path(cfg.local_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    total_log_path = log_dir / f"{shared_run_id}.txt"
    worker_script = ROOT_DIR / "experiments" / "robot" / "libero" / "suction_infer" / "run_libero_single_task_suction_eval.py"

    log_lines = [
        f"Task suite: {cfg.task_suite_name}",
        f"Checkpoint: {cfg.pretrained_checkpoint}",
        f"Use force FiLM: {cfg.use_force_film}",
        f"Force-FiLM checkpoint: {cfg.force_film_checkpoint}",
        f"Force-FiLM encoder embedding type: {cfg.force_encoder_embedding_type}",
        f"Force-FiLM history length: {cfg.force_history_length}",
        f"Constant Force-FiLM input before first contact: {cfg.constant_force_film_input_before_first_contact}",
        f"Unnorm key: {cfg.unnorm_key}",
        f"Local video dir: {cfg.local_video_dir}",
        f"Shared run id: {shared_run_id}",
        f"Run mode: {'resume' if is_resume_mode else 'fresh'}",
        "",
        "Worker layout:",
    ]
    for spec in cfg.worker_specs:
        log_lines.append(f"- worker{spec['worker_id']:02d}: gpu={spec['gpu_id']}, tasks={spec['task_orders']}")
    write_total_log(total_log_path, log_lines)

    launched = []
    for spec in cfg.worker_specs:
        worker_id = spec["worker_id"]
        artifact_paths = build_worker_artifact_paths(shared_run_id, worker_id, log_dir)
        if is_resume_mode:
            resume_state = load_existing_worker_resume_state(cfg, spec, artifact_paths)
        else:
            resume_state = build_empty_resume_state(cfg, spec, artifact_paths, resume_source="fresh")

        validate_resume_state_checkpoint(cfg, resume_state, shared_run_id=shared_run_id, worker_id=worker_id)

        # 统一把标准化恢复计划写到本地文件。
        # fresh run 也会写一份，这样 worker 端始终只需要处理一种输入格式。
        Path(artifact_paths["resume_state_json_path"]).write_text(json.dumps(resume_state, ensure_ascii=False, indent=2))

        launched.append(
            {
                "spec": spec,
                "resume_state": resume_state,
                "artifact_paths": artifact_paths,
                "process": None,
                "return_code": None,
                "skipped": False,
            }
        )

        if not worker_has_remaining_work(cfg, spec, resume_state):
            launched[-1]["skipped"] = True
            print(
                f"Skipping worker{worker_id:02d}: all assigned tasks are already completed "
                f"(source={resume_state.get('resume_source')})."
            )
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = spec["gpu_id"]
        # 每个 worker 都是独立 Python 子进程。这里把 launcher 侧的默认线程限制也写进子进程环境，
        # 确保 worker 在导入 NumPy / MuJoCo / LIBERO 等库之前就能拿到同样的线程配置。
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

        cmd = [
            sys.executable,
            str(worker_script),
            "--model-family",
            cfg.model_family,
            "--pretrained-checkpoint",
            str(cfg.pretrained_checkpoint),
            "--load-in-8bit",
            bool_to_cli(cfg.load_in_8bit),
            "--load-in-4bit",
            bool_to_cli(cfg.load_in_4bit),
            "--center-crop",
            bool_to_cli(cfg.center_crop),
            "--use-force-film",
            bool_to_cli(cfg.use_force_film),
            "--force-encoder-embedding-type",
            str(cfg.force_encoder_embedding_type),
            "--force-history-length",
            str(cfg.force_history_length),
            "--constant-force-film-input-before-first-contact",
            bool_to_cli(cfg.constant_force_film_input_before_first_contact),
            "--task-suite-name",
            cfg.task_suite_name,
            "--gripper-type",
            cfg.gripper_type,
            "--unnorm-key",
            cfg.unnorm_key,
            "--num-steps-wait",
            str(cfg.num_steps_wait),
            "--max-steps",
            str(cfg.max_steps),
            "--num-trials",
            str(cfg.num_trials),
            "--resolution",
            str(cfg.resolution),
            "--control-freq",
            str(cfg.control_freq),
            "--save-all-rollout-videos",
            bool_to_cli(cfg.save_all_rollout_videos),
            "--rollout-video-fps",
            str(cfg.rollout_video_fps),
            "--save-action-wrench-history-csv",
            bool_to_cli(cfg.save_action_wrench_history_csv),
            "--enable-suction-failure-detection",
            bool_to_cli(cfg.enable_suction_failure_detection),
            "--suction-failure-contact-fz-threshold",
            str(cfg.suction_failure_contact_fz_threshold),
            "--suction-failure-free-fz-threshold",
            str(cfg.suction_failure_free_fz_threshold),
            "--suction-failure-lift-action-z-threshold",
            str(cfg.suction_failure_lift_action_z_threshold),
            "--suction-failure-contact-history-steps",
            str(cfg.suction_failure_contact_history_steps),
            "--suction-failure-min-contact-steps",
            str(cfg.suction_failure_min_contact_steps),
            "--show-suction-failure-overlay-in-video",
            bool_to_cli(cfg.show_suction_failure_overlay_in_video),
            "--enable-contact-force-suction-override",
            bool_to_cli(cfg.enable_contact_force_suction_override),
            "--enable-suction-failure-rewind",
            bool_to_cli(cfg.enable_suction_failure_rewind),
            "--suction-rewind-pre-contact-steps",
            str(cfg.suction_rewind_pre_contact_steps),
            "--enable-suction-failure-escape-perturb",
            bool_to_cli(cfg.enable_suction_failure_escape_perturb),
            "--suction-escape-perturb-scale-step",
            str(cfg.suction_escape_perturb_scale_step),
            "--local-log-dir",
            cfg.local_log_dir,
            "--local-video-dir",
            cfg.local_video_dir,
            "--use-wandb",
            bool_to_cli(cfg.use_wandb),
            "--wandb-project",
            cfg.wandb_project,
            "--wandb-entity",
            cfg.wandb_entity,
            "--seed",
            str(cfg.seed),
            "--worker-id",
            str(worker_id),
            "--task-orders",
            ",".join(str(task_id) for task_id in spec["task_orders"]),
            "--shared-run-id",
            shared_run_id,
            "--summary-json-path",
            str(artifact_paths["summary_json_path"]),
            "--worker-log-path",
            str(artifact_paths["worker_log_path"]),
            "--progress-json-path",
            str(artifact_paths["progress_json_path"]),
            "--resume-state-json-path",
            str(artifact_paths["resume_state_json_path"]),
            "--worker-video-subdir",
            str(artifact_paths["worker_video_subdir"]),
            "--worker-use-wandb",
            bool_to_cli(cfg.worker_use_wandb),
        ]
        if cfg.force_film_checkpoint is not None:
            cmd.extend(["--force-film-checkpoint", str(cfg.force_film_checkpoint)])
        if cfg.run_id_note is not None:
            cmd.extend(["--run-id-note", cfg.run_id_note])

        print(
            f"Launching worker{worker_id:02d} on GPU {spec['gpu_id']} with tasks {spec['task_orders']} "
            f"(source={resume_state.get('resume_source')})"
        )
        process = subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=env)
        launched[-1]["process"] = process

    failed_workers = []
    for item in launched:
        if item["process"] is None:
            continue
        return_code = item["process"].wait()
        item["return_code"] = return_code
        if return_code != 0:
            failed_workers.append(item)

    summary_payloads = []
    for item in launched:
        summary_path = Path(item["artifact_paths"]["summary_json_path"])
        if not summary_path.exists():
            # 对于“恢复时已判定为不用重新拉起”的 worker，历史上不一定真的存在 summary.json；
            # 例如旧运行只留下了 worker txt，这时就直接把标准化恢复状态转成 summary 参与总汇总。
            if item["skipped"]:
                summary_payloads.append(
                    build_summary_from_resume_state(cfg, item["spec"], item["resume_state"], item["artifact_paths"])
                )
                continue
            raise FileNotFoundError(f"Worker summary not found: {summary_path}")
        summary_payloads.append(read_json_file(summary_path))

    task_rows = []
    total_episodes = 0
    total_successes = 0
    total_rollout_videos = 0
    total_success_videos = 0
    total_failed_videos = 0
    for payload in summary_payloads:
        task_rows.extend(payload["task_results"])
        total_episodes += int(payload["total_episodes"])
        total_successes += int(payload["total_successes"])
        total_rollout_videos += int(payload["total_rollout_videos"])
        total_success_videos += int(payload["total_success_videos"])
        total_failed_videos += int(payload["total_failed_videos"])

    task_rows.sort(key=lambda item: item["task_id"])
    suite_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0

    log_lines = [
        f"Task suite: {cfg.task_suite_name}",
        f"Checkpoint: {cfg.pretrained_checkpoint}",
        f"Use force FiLM: {cfg.use_force_film}",
        f"Force-FiLM checkpoint: {cfg.force_film_checkpoint}",
        f"Force-FiLM encoder embedding type: {cfg.force_encoder_embedding_type}",
        f"Force-FiLM history length: {cfg.force_history_length}",
        f"Constant Force-FiLM input before first contact: {cfg.constant_force_film_input_before_first_contact}",
        f"Unnorm key: {cfg.unnorm_key}",
        f"Local video dir: {cfg.local_video_dir}",
        f"Shared run id: {shared_run_id}",
        f"Run mode: {'resume' if is_resume_mode else 'fresh'}",
        "",
        "Worker layout:",
    ]
    for item in launched:
        spec = item["spec"]
        worker_state = item["resume_state"]
        worker_status = "skipped-resume" if item["skipped"] else f"return_code={item['return_code']}"
        log_lines.append(
            f"- worker{spec['worker_id']:02d}: gpu={spec['gpu_id']}, tasks={spec['task_orders']}, "
            f"status={worker_status}, resume_source={worker_state.get('resume_source')}, "
            f"completed_before_resume={worker_state.get('completed_task_ids')}"
        )
    log_lines.extend(["", "Per-task success rates:"])
    for row in task_rows:
        log_lines.append(
            f"- task_{row['task_id']:02d}: {row['task_name']} | {row['successes']} / {row['episodes']} | {row['success_rate'] * 100:.1f}%"
        )
    log_lines.extend(
        [
            "",
            f"Suite final success rate: {suite_success_rate}",
            f"Total episodes: {total_episodes}",
            f"Total successes: {total_successes}",
            f"Total rollout videos: {total_rollout_videos}",
            f"Total successful rollout videos: {total_success_videos}",
            f"Total failed rollout videos: {total_failed_videos}",
            "",
            "Worker logs:",
        ]
    )
    for item in launched:
        log_lines.append(f"- {item['artifact_paths']['worker_log_path']}")
    write_total_log(total_log_path, log_lines)

    if failed_workers:
        failed_names = ", ".join(f"worker{item['spec']['worker_id']:02d}" for item in failed_workers)
        raise RuntimeError(f"Parallel LIBERO evaluation failed for: {failed_names}. See worker logs under {cfg.local_log_dir}.")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=shared_run_id)
        task_table = wandb.Table(columns=["task_id", "task_name", "successes", "episodes", "success_rate"])
        for row in task_rows:
            task_table.add_data(
                row["task_id"],
                row["task_name"],
                row["successes"],
                row["episodes"],
                row["success_rate"],
            )
        # 这里给 W&B key 加独立前缀，只区分可视化结果名称，不改变本地评测、汇总和 project 位置。
        wandb_result_prefix = str(cfg.wandb_result_prefix).strip("/")
        task_table_key = f"{wandb_result_prefix}/success_rate/task_table"
        suite_final_key = f"{wandb_result_prefix}/success_rate/suite_final"
        wandb.log(
            {
                task_table_key: task_table,
                suite_final_key: suite_success_rate,
            }
        )
        wandb.finish()

    print(f"Parallel evaluation complete. Summary log: {total_log_path}")
    print(f"Suite final success rate: {suite_success_rate:.4f}")


if __name__ == "__main__":
    launch_parallel_eval()
