"""
web_teleop.py

【脚本定位】
这是一个“无头环境浏览器键盘遥操作”脚本。

它和旧版最大的区别是：
1. **控制链路**改成浏览器键盘通过 WebSocket 发到服务器；
2. **监看链路**仍通过 FastAPI + WebSocket 把视频、六维力和状态推送到浏览器。

这样做的原因很简单：
- 你当前是 VSCode Remote / 远程服务器场景，物理键盘并不在服务器上；
- 所以服务器本地 `/dev/input/event*` 根本看不到你本地电脑的键盘；
- 最可靠的方案是让浏览器同时承担“看 + 控制”，并用 WebSocket 保持双向实时通信。

【你现在怎么用它】
1. 启动这个脚本；
2. 用本地电脑浏览器打开页面；
3. 点击页面让它获得焦点；
4. 在浏览器里直接按键控制机械臂，并实时看画面和六维力。
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---- 路径设定 ----
# 这里不再写死 '/data1/ps/LIBERO'，而是根据当前文件实际位置推根目录。
ROOT_DIR = Path(__file__).resolve().parents[1]
LIBERO_DIR = ROOT_DIR / "LIBERO"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(LIBERO_DIR))

_BACKEND_PROBE_SNIPPETS = {
    "egl": """
import os
os.environ["MUJOCO_GL"] = "egl"
try:
    from mujoco.egl import create_initialized_egl_device_display
    from mujoco.egl import egl_ext as EGL
    display = create_initialized_egl_device_display()
    if display == EGL.EGL_NO_DISPLAY:
        raise ImportError(
            "Cannot initialize a EGL device display. This likely means that your EGL driver "
            "does not support the PLATFORM_DEVICE extension, which is required for creating "
            "a headless rendering context."
        )
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
""",
    "osmesa": """
import os
os.environ["MUJOCO_GL"] = "osmesa"
try:
    from mujoco.osmesa import GLContext
    ctx = GLContext(32, 32)
    ctx.make_current()
    ctx.free()
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
""",
    "glfw": """
import os
import warnings
import glfw
warnings.simplefilter("ignore")
if not glfw.init():
    display = os.environ.get("DISPLAY", "")
    print(f"GLFW init failed (DISPLAY={display!r})", file=sys.stderr)
    raise SystemExit(1)
glfw.terminate()
""",
}


def _ensure_runtime_cache_dir(*relative_parts: str) -> Path:
    candidates = []

    env_tmpdir = os.environ.get("TMPDIR", "").strip()
    if env_tmpdir:
        candidates.append(Path(env_tmpdir))

    candidates.extend(
        [
            Path("/dev/shm"),
            Path("/tmp"),
            Path("/var/tmp"),
            Path.home() / ".cache" / "openvla_web_teleop",
        ]
    )

    for base_dir in candidates:
        try:
            if base_dir.exists() and base_dir.is_dir() and os.access(base_dir, os.W_OK):
                if not relative_parts:
                    return base_dir

                target_dir = base_dir.joinpath(*relative_parts)
                target_dir.mkdir(parents=True, exist_ok=True)
                if os.access(target_dir, os.W_OK):
                    return target_dir
                continue

            base_dir.mkdir(parents=True, exist_ok=True)
            if os.access(base_dir, os.W_OK):
                if not relative_parts:
                    return base_dir

                target_dir = base_dir.joinpath(*relative_parts)
                target_dir.mkdir(parents=True, exist_ok=True)
                if os.access(target_dir, os.W_OK):
                    return target_dir
        except OSError:
            continue

    raise RuntimeError("未找到可写的运行时缓存目录，请检查 /tmp、/var/tmp 或 ~/.cache 的写权限。")


def _configure_runtime_environment():
    runtime_tmpdir = _ensure_runtime_cache_dir()
    mpl_config_dir = _ensure_runtime_cache_dir()

    os.environ["TMPDIR"] = str(runtime_tmpdir)
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))


def _probe_render_backend(backend: str):
    snippet = _BACKEND_PROBE_SNIPPETS[backend]
    env = os.environ.copy()
    env["MUJOCO_GL"] = backend

    result = subprocess.run(
        [sys.executable, "-c", "import sys\n" + snippet],
        cwd=str(ROOT_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    detail = (result.stderr or result.stdout).strip()
    if result.returncode == 0:
        return True, detail

    if not detail:
        detail = f"probe exited with code {result.returncode}"

    interesting_prefixes = (
        "ImportError:",
        "RuntimeError:",
        "AttributeError:",
        "GLFW init failed",
    )
    for line in detail.splitlines():
        stripped = line.strip()
        if stripped.startswith(interesting_prefixes):
            return False, stripped

    return False, detail.splitlines()[-1].strip()


def _select_render_backend() -> str:
    explicit_backend = os.environ.get("MUJOCO_GL", "").strip().lower()
    candidate_backends = [explicit_backend] if explicit_backend else ["egl", "osmesa", "glfw"]
    failure_messages = []

    for backend in candidate_backends:
        ok, detail = _probe_render_backend(backend)
        if ok:
            os.environ["MUJOCO_GL"] = backend
            return backend
        failure_messages.append(f"- `{backend}`: {detail}")

    failure_summary = "\n".join(failure_messages)
    raise RuntimeError(
        "当前主机上没有可用的 MuJoCo 渲染后端，无法启动 LIBERO web teleop。\n"
        "已检查的后端如下：\n"
        f"{failure_summary}\n"
        "请切换到支持 headless EGL 的服务器，或在有图形显示环境的机器上运行。"
    )


def _import_runtime_components():
    from libero.libero.envs.env_wrapper import ControlEnv
    from experiments.robot.libero.data_regeneration.realtime_control_and_monitor import (
        BrowserKeyboardController,
        RealtimeMonitorState,
        start_monitor_server,
    )
    from experiments.robot.libero.libero_utils import create_combined_frame, extract_suction_wrench

    return (
        ControlEnv,
        BrowserKeyboardController,
        RealtimeMonitorState,
        start_monitor_server,
        create_combined_frame,
        extract_suction_wrench,
    )


def build_parser():
    """定义命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bddl_file",
        type=str,
        default="LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl",
        help="要遥操作的 BDDL 文件。",
    )
    parser.add_argument("--camera_height", type=int, default=256, help="相机图像高度。")
    parser.add_argument("--camera_width", type=int, default=256, help="相机图像宽度。")
    parser.add_argument("--control_freq", type=int, default=20, help="环境控制频率。")
    parser.add_argument("--movement_speed", type=float, default=0.4, help="浏览器键盘控制的平移速度。")
    parser.add_argument("--web_host", type=str, default="0.0.0.0", help="监看页面监听地址。")
    parser.add_argument("--web_port", type=int, default=8081, help="监看页面监听端口。")
    return parser


def run_simulation(args):
    """
    运行主仿真循环。

    这条循环只做三件事：
    1. 从浏览器键盘控制器读取当前动作；
    2. 用动作推进环境一步；
    3. 把新画面和六维力推送给 WebSocket 监看服务。
    """
    _configure_runtime_environment()
    selected_backend = _select_render_backend()
    (
        ControlEnv,
        BrowserKeyboardController,
        RealtimeMonitorState,
        start_monitor_server,
        create_combined_frame,
        extract_suction_wrench,
    ) = _import_runtime_components()

    env = None
    server = None
    controller = None
    monitor_state = None

    try:
        monitor_state = RealtimeMonitorState(
            title="LIBERO 浏览器键盘低延迟遥操作",
            jpeg_quality=70,
            update_hz=args.control_freq,
        )

        controller = BrowserKeyboardController(
            movement_speed=args.movement_speed,
            enable_handover=False,
            reset_suction_on_reset=False,
        )

        # 对齐数据采集脚本的浏览器交互方式：先起服务，再建环境。
        _, server = start_monitor_server(
            monitor_state,
            host=args.web_host,
            port=args.web_port,
            title="LIBERO 低延迟遥操作监看",
            subtitle="浏览器页面负责视频显示、键盘控制和吸盘状态显示；需先在页面中获得焦点后才能发送控制指令。",
            keyboard_lines=[
                "先点击页面，使浏览器获得键盘焦点",
                "浏览器键盘 W / S：X 方向前后",
                "浏览器键盘 A / D：Y 方向左右",
                "浏览器键盘 ↑ / ↓：Z 方向上下",
                "浏览器键盘 SPACE：切换吸盘开 / 关",
                "浏览器键盘 ESC：退出脚本",
            ],
            control_state=controller,
            show_wrench_panels=False,
            keyboard_help_position="below_video",
            require_monitor_for_actions=True,
        )

        print(f"[SIM] 使用 MuJoCo 渲染后端: {selected_backend}")
        print(
            f"[SIM] 浏览器监看页面: http://{args.web_host if args.web_host != '0.0.0.0' else '127.0.0.1'}:{args.web_port}"
        )

        # 初始化环境。
        env = ControlEnv(
            bddl_file_name=args.bddl_file,
            robots=["Panda"],
            controller="OSC_POSE",
            gripper_types="suction",
            use_camera_obs=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_heights=args.camera_height,
            camera_widths=args.camera_width,
            ignore_done=True,
            control_freq=args.control_freq,
        )
        obs = env.reset()
        info = {}

        print("\n" + "=" * 60)
        print("低延迟遥操作已启动")
        print(f"浏览器监看页面: http://{args.web_host if args.web_host != '0.0.0.0' else '127.0.0.1'}:{args.web_port}")
        print("浏览器页面获得焦点后，可直接用键盘控制机械臂。")
        print("=" * 60 + "\n")

        # 主循环持续运行，直到用户按 Q / ESC 请求退出。
        while not monitor_state.shutdown_event.is_set():
            if controller.consume_shutdown_request():
                print("[SIM] 收到浏览器退出请求，准备关闭。")
                break

            action = controller.get_action(action_dim=7)
            obs, reward, done, info = env.step(action.tolist())

            # teleop 场景里如果任务 done，就直接 reset 开始下一个回合。
            if done:
                print("[SIM] 当前回合已完成，自动重置环境。")
                obs = env.reset()
                info = {}

            # 生成监看图，并推送给浏览器。
            wrench = extract_suction_wrench(obs, info)
            combined_frame = create_combined_frame(
                obs["agentview_image"],
                wrench,
                obs["robot0_eye_in_hand_image"],
                img_size=args.camera_height,
            )
            monitor_state.update(
                frame_rgb=combined_frame,
                mode_label="MANUAL_BROWSER",
                suction_on=bool(info.get("suction_on", False)),
                suction_latched=bool(info.get("suction_latched", False)),
                suction_desired=controller.get_desired_suction_state(),
                suction_wrench=wrench,
                suction_wrench_total_load=np.asarray(
                    info.get("suction_wrench_total_load_world", np.zeros(6, dtype=np.float64)),
                    dtype=np.float64,
                ),
                extra_lines=[
                    f"reward={reward}",
                    f"done={done}",
                    f"suction_contact={info.get('suction_cup_contact', False)}",
                    f"suction_stable_steps={info.get('suction_stable_steps', 0)}",
                    # 下面两行显示吸盘投影面积重合度，方便直接观察 90% 吸附门槛是否通过。
                    f"suction_overlap_ratio={float(info.get('suction_overlap_ratio', -1.0)):.3f}",
                    f"suction_overlap_passed={info.get('suction_overlap_passed', False)}",
                ],
            )

            time.sleep(1.0 / args.control_freq)
    finally:
        if monitor_state is not None:
            monitor_state.shutdown_event.set()
        if server is not None:
            server.should_exit = True
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                    print("[SIM] Environment closed cleanly.")
                except Exception as exc:
                    print(f"[SIM] Failed to close environment cleanly: {exc}")


if __name__ == "__main__":
    try:
        run_simulation(build_parser().parse_args())
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
