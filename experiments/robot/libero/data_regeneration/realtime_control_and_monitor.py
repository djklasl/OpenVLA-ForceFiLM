"""
realtime_control_and_monitor.py

【模块定位】
这个模块负责“浏览器键盘 + 双 WebSocket 双向通信”的实时控制链路。

它把实时遥操作拆成两部分：
1. 浏览器通过独立的 control WebSocket 把键盘按下 / 松开事件、按钮点击事件发给 Python；
2. Python 通过独立的 monitor WebSocket，把最新视频帧、模式状态、吸盘状态和六维力数据推回浏览器。

【为什么现在要这样做】
- 你当前是通过 VSCode Remote / SSH 连接服务器，物理键盘并不在服务器上；
- 因此服务器上的 `/dev/input/event*` 并不能读到你本地电脑的键盘；
- 所以控制端必须回到“浏览器捕获键盘，再把按键发给服务器”的结构。

【这个模块提供什么】
- `BrowserKeyboardController`
  服务端保存浏览器当前按键状态，并把它转换成环境动作。
- `RealtimeMonitorState`
  保存要推送给浏览器的最新一帧 JPEG 图像和状态文本。
- `start_monitor_server`
  启动 FastAPI + WebSocket 服务，同时负责接收控制消息和推送监看消息。
"""

from __future__ import annotations

import io
import json
import threading
from typing import Iterable

import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 FastAPI。请先执行: pip install fastapi 'uvicorn[standard]'"
    ) from exc

try:
    import uvicorn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 uvicorn。请先执行: pip install 'uvicorn[standard]'"
    ) from exc

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "缺少 Pillow。请先执行: pip install pillow"
    ) from exc


# ------------------------------------------------------------------------------
# 浏览器键位定义
# ------------------------------------------------------------------------------
# 这里统一使用浏览器 KeyboardEvent.code，而不是 event.key。
# 原因是 code 更稳定，不会因为输入法或键盘布局变化而改变语义。
KEY_MOVE_FORWARD = "KeyW"
KEY_MOVE_BACKWARD = "KeyS"
KEY_MOVE_LEFT = "KeyA"
KEY_MOVE_RIGHT = "KeyD"
KEY_MOVE_UP = "ArrowUp"
KEY_MOVE_DOWN = "ArrowDown"
KEY_TOGGLE_SUCTION = "Space"
KEY_HANDOVER = "KeyH"
KEY_RESTART = "KeyR"
KEY_ESC = "Escape"


def _encode_jpeg(frame_rgb: np.ndarray, jpeg_quality: int = 80) -> bytes:
    """把 RGB 图像编码成 JPEG 字节流，供 WebSocket 推送。"""
    if cv2 is not None:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if ok:
            return buffer.tobytes()

    image = Image.fromarray(frame_rgb)
    binary = io.BytesIO()
    image.save(binary, format="JPEG", quality=int(jpeg_quality))
    return binary.getvalue()


class BrowserKeyboardController:
    """
    维护浏览器当前按键状态，并把它转换成环境动作。

    这个类只存在于服务端。浏览器每次按键变化时通过 WebSocket 发消息，
    服务端把“当前有哪些键仍在按下”记录到 `pressed_keys` 里。

    这样主仿真循环只需要每步调用 `get_action()`，就能得到当前动作。
    """

    def __init__(
        self,
        *,
        movement_speed: float = 0.8,
        enable_handover: bool = False,
        reset_suction_on_reset: bool = True,
    ):
        self.movement_speed = float(movement_speed)
        self.enable_handover = bool(enable_handover)
        self.reset_suction_on_reset = bool(reset_suction_on_reset)

        # pressed_keys 保存“当前仍然处于按下状态”的浏览器键。
        self.pressed_keys: set[str] = set()

        # suction_state 仍然用 -1 / +1 这种离散值，和现有吸盘控制语义保持一致。
        self.suction_state = -1.0

        # 这两个标志是“一次性事件”，由主循环读取后清空。
        self.handover_requested = False
        self.restart_requested = False
        self.shutdown_requested = False
        self.button_action: str | None = None

        self.lock = threading.Lock()

    def reset_episode_state(self):
        """每条新 episode 开始时清空按键状态，避免上一个回合残留。"""
        with self.lock:
            self.pressed_keys.clear()
            self.handover_requested = False
            self.restart_requested = False
            self.shutdown_requested = False
            self.button_action = None
            if self.reset_suction_on_reset:
                self.suction_state = -1.0

    def clear_pressed_keys(self):
        """
        清空当前按下状态。

        这个函数会在两类场景里触发：
        1. 浏览器页面失焦；
        2. WebSocket 断开。

        这样可以避免用户窗口切走后，服务器还以为某个方向键一直按着。
        """
        with self.lock:
            self.pressed_keys.clear()

    def consume_handover_request(self) -> bool:
        """主循环读取一次 handover 请求，并立即把它消费掉。"""
        with self.lock:
            requested = self.handover_requested
            self.handover_requested = False
            return requested

    def consume_shutdown_request(self) -> bool:
        """主循环读取一次退出请求，并立即把它消费掉。"""
        with self.lock:
            requested = self.shutdown_requested
            self.shutdown_requested = False
            return requested

    def consume_restart_request(self) -> bool:
        """主循环读取一次“重启当前 demo”请求，并立即把它消费掉。"""
        with self.lock:
            requested = self.restart_requested
            self.restart_requested = False
            return requested

    def consume_button_action(self) -> str | None:
        """主循环读取一次页面按钮点击事件，并立即把它消费掉。"""
        with self.lock:
            action = self.button_action
            self.button_action = None
            return action

    def get_action(self, action_dim: int) -> np.ndarray:
        """
        把当前浏览器按键状态转换成环境动作。

        这里只控制平移三维和最后一维的吸盘开关。
        其余维度全部填 0，保持和当前 OSC_POSITION / suction 的使用方式一致。
        """
        with self.lock:
            action = np.zeros(action_dim, dtype=np.float64)

            if KEY_MOVE_FORWARD in self.pressed_keys:
                action[0] -= self.movement_speed
            if KEY_MOVE_BACKWARD in self.pressed_keys:
                action[0] += self.movement_speed
            if KEY_MOVE_LEFT in self.pressed_keys:
                action[1] += self.movement_speed
            if KEY_MOVE_RIGHT in self.pressed_keys:
                action[1] -= self.movement_speed
            if KEY_MOVE_UP in self.pressed_keys:
                action[2] += self.movement_speed
            if KEY_MOVE_DOWN in self.pressed_keys:
                action[2] -= self.movement_speed

            action[-1] = self.suction_state
            return action

    def get_desired_suction_state(self) -> float:
        """返回控制器内部记录的吸盘目标开关状态。"""
        with self.lock:
            return float(self.suction_state)

    def handle_keydown(self, code: str):
        """
        处理浏览器发来的 keydown。

        注意这里的“开关类逻辑”只在第一次按下时触发一次：
        - Space：切换吸盘开关
        - H：触发 handover
        - R：重启当前 demo
        - ESC：请求退出

        这样就不会因为浏览器长按产生重复 keydown 而抖动。
        """
        if not code:
            return

        with self.lock:
            first_press = code not in self.pressed_keys
            self.pressed_keys.add(code)

            if code == KEY_TOGGLE_SUCTION and first_press:
                self.suction_state = 1.0 if self.suction_state < 0 else -1.0

            if self.enable_handover and code == KEY_HANDOVER and first_press:
                self.handover_requested = True

            if self.enable_handover and code == KEY_RESTART and first_press:
                self.restart_requested = True

            if code == KEY_ESC and first_press:
                self.shutdown_requested = True

    def handle_keyup(self, code: str):
        """处理浏览器发来的 keyup，把对应键从按下集合里移除。"""
        if not code:
            return

        with self.lock:
            self.pressed_keys.discard(code)

    def handle_message(self, message: dict):
        """
        根据浏览器发来的 JSON 消息更新内部状态。

        支持的消息格式：
        - `{"type": "keydown", "code": "KeyW"}`
        - `{"type": "keyup", "code": "KeyW"}`
        - `{"type": "clear_keys"}`
        - `{"type": "button", "button_id": "preview"}`
        """
        message_type = str(message.get("type", "")).strip()
        code = str(message.get("code", "")).strip()
        button_id = str(message.get("button_id", "")).strip()

        if message_type == "keydown":
            self.handle_keydown(code)
        elif message_type == "keyup":
            self.handle_keyup(code)
        elif message_type == "clear_keys":
            self.clear_pressed_keys()
        elif message_type == "button":
            with self.lock:
                self.button_action = button_id or None


class RealtimeMonitorState:
    """
    保存需要推给浏览器的最新监看数据。

    这个类和输入层解耦：它只负责“把最新状态缓存起来”，
    具体谁来消费、怎么发给浏览器，由 WebSocket 服务决定。
    """

    def __init__(self, *, title: str, jpeg_quality: int = 80, update_hz: float = 20.0):
        self.title = title
        self.jpeg_quality = int(jpeg_quality)
        self.update_hz = float(update_hz)

        self.mode_label = "UNKNOWN"
        self.suction_on = False
        self.suction_latched = False
        self.suction_desired = -1.0
        self.suction_wrench = np.zeros(6, dtype=np.float64)
        self.suction_wrench_total_load = np.zeros(6, dtype=np.float64)
        self.stage_title = "等待页面初始化"
        self.stage_description = "浏览器连上之后，服务端会把当前阶段和按钮说明推送到这里。"
        self.action_buttons: list[dict] = []
        self.extra_lines: list[str] = []

        self._frame_bytes = b""
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()

    def update(
        self,
        *,
        frame_rgb: np.ndarray,
        mode_label: str,
        suction_on: bool,
        suction_latched: bool,
        suction_desired: float,
        suction_wrench: np.ndarray,
        suction_wrench_total_load: np.ndarray | None = None,
        stage_title: str = "",
        stage_description: str = "",
        action_buttons: list[dict] | None = None,
        extra_lines: list[str] | None = None,
        update_hz: float | None = None,
    ):
        """更新最新一帧监看数据。"""
        with self.lock:
            self._frame_bytes = _encode_jpeg(np.asarray(frame_rgb, dtype=np.uint8), jpeg_quality=self.jpeg_quality)
            self.mode_label = str(mode_label)
            self.suction_on = bool(suction_on)
            self.suction_latched = bool(suction_latched)
            self.suction_desired = float(suction_desired)
            self.suction_wrench = np.asarray(suction_wrench, dtype=np.float64).copy()
            if suction_wrench_total_load is None:
                self.suction_wrench_total_load = np.zeros(6, dtype=np.float64)
            else:
                self.suction_wrench_total_load = np.asarray(suction_wrench_total_load, dtype=np.float64).copy()
            self.stage_title = str(stage_title)
            self.stage_description = str(stage_description)
            self.action_buttons = [dict(button) for button in (action_buttons or [])]
            self.extra_lines = list(extra_lines or [])
            if update_hz is not None:
                # 允许不同阶段动态调整监看推送频率。
                self.update_hz = float(update_hz)

    def snapshot(self) -> tuple[dict, bytes]:
        """给 WebSocket 推送线程取一份当前快照。"""
        with self.lock:
            payload = {
                "type": "state",
                "mode_label": self.mode_label,
                "suction_on": self.suction_on,
                "suction_latched": self.suction_latched,
                "suction_desired": self.suction_desired,
                "suction_wrench": self.suction_wrench.tolist(),
                "suction_wrench_total_load": self.suction_wrench_total_load.tolist(),
                "stage_title": self.stage_title,
                "stage_description": self.stage_description,
                "action_buttons": list(self.action_buttons),
                "extra_lines": list(self.extra_lines),
            }
            return payload, bytes(self._frame_bytes)


def _build_keyboard_help_html(keyboard_lines: Iterable[str]) -> str:
    """把按键说明列表渲染成 HTML `<li>`。"""
    return "\n".join(f"<li>{line}</li>" for line in keyboard_lines)



def build_monitor_html(
    title: str,
    subtitle: str,
    keyboard_lines: list[str],
    control_enabled: bool,
    *,
    show_wrench_panels: bool = True,
    keyboard_help_position: str = "sidebar",
    require_monitor_for_actions: bool = True,
    video_max_width_px: int | None = None,
) -> str:
    """
    构建浏览器页面 HTML。

    当前页面同时承担两个职责：
    1. 显示视频、状态与必要说明；
    2. 捕获浏览器键盘，把 keydown / keyup 发回服务器。

    这个函数支持页面级定制：
    - `show_wrench_panels=False` 时，可把六维力区块从页面中移除；
    - `keyboard_help_position="below_video"` 时，可把键盘说明紧贴到视频下方；
    - `require_monitor_for_actions=True` 时，只有控制通道和监看通道都连上后，按钮和键盘才会真正生效。
    """
    keyboard_html = _build_keyboard_help_html(keyboard_lines)
    control_notice = "浏览器页面获得焦点后可直接键盘控制。" if control_enabled else "当前页面仅监看，不接收键盘控制。"
    require_monitor_literal = "true" if require_monitor_for_actions else "false"
    video_style = ""
    if video_max_width_px is not None:
        video_style = (
            ".video-wrap { display: flex; justify-content: center; align-items: center; }"
            f"#video {{ width: 100%; max-width: {int(video_max_width_px)}px; margin: 0 auto; image-rendering: auto; }}"
        )

    keyboard_panel_html = f"""
                <div class=\"panel keyboard-panel\">
                    <h3>浏览器键盘说明</h3>
                    <ul class=\"help-list\">{keyboard_html}</ul>
                </div>
    """

    if show_wrench_panels:
        video_footer_html = keyboard_panel_html if keyboard_help_position == "below_video" else ""
        sidebar_help_html = "" if keyboard_help_position == "below_video" else keyboard_panel_html

        lower_section_html = """
        <div class=\"bottom-grid\" id=\"bottom-grid\">
            <div class=\"panel\">
                <h3>接触瞬时外力（世界系）</h3>
                <div id=\"wrench-contact\" class=\"mono\">fx=0.000 fy=0.000 fz=0.000
 tx=0.000 ty=0.000 tz=0.000</div>
            </div>

            <div class=\"panel\">
                <h3>总外载荷调试值（世界系）</h3>
                <div id=\"wrench-total\" class=\"mono\">fx=0.000 fy=0.000 fz=0.000
 tx=0.000 ty=0.000 tz=0.000</div>
            </div>

            <div class=\"panel\">
                <h3>附加状态</h3>
                <div id=\"extra\" class=\"mono\">等待首帧...</div>
            </div>
        </div>
"""
    else:
        # 采集页不显示六维力面板时，把“附加状态”直接放到视频下方，
        # 并且紧贴“浏览器键盘说明”，避免它掉到页面底部难以看到。
        video_footer_html = f"""
                {keyboard_panel_html if keyboard_help_position == "below_video" else ""}
                <div class=\"panel extra-panel\">
                    <h3>附加状态</h3>
                    <div id=\"extra\" class=\"mono\">等待首帧...</div>
                </div>
"""
        sidebar_help_html = "" if keyboard_help_position == "below_video" else f"""
                {keyboard_panel_html}
                <div class=\"panel extra-panel\">
                    <h3>附加状态</h3>
                    <div id=\"extra\" class=\"mono\">等待首帧...</div>
                </div>
"""
        lower_section_html = ""

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 10px; }}
        .container {{ max-width: 1440px; margin: 0 auto; background: #111827; padding: 12px; border-radius: 12px; min-height: calc(100vh - 20px); }}
        .topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }}
        .title-block h1 {{ margin: 0; font-size: 1.45rem; }}
        .title-block p {{ margin: 4px 0 0 0; color: #cbd5e1; font-size: 0.95rem; }}
        .badge-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #1e293b; color: #93c5fd; font-size: 0.92rem; }}
        .layout {{ display: grid; grid-template-columns: minmax(640px, 2fr) minmax(320px, 1fr); gap: 12px; align-items: start; }}
        .left-col, .right-col {{ display: grid; gap: 12px; }}
        .right-col {{ grid-template-rows: auto auto auto auto; }}
        .panel {{ background: #0b1220; border-radius: 10px; padding: 10px 12px; border: 1px solid #1e293b; }}
        .panel h3 {{ margin: 0 0 8px 0; font-size: 1rem; }}
        .status {{ font-size: 0.95rem; line-height: 1.5; }}
        .status-tight div + div {{ margin-top: 4px; }}
        .green {{ color: #34d399; }}
        .yellow {{ color: #fbbf24; }}
        .red {{ color: #f87171; }}
        .mono {{ font-family: monospace; white-space: pre; font-size: 0.9rem; line-height: 1.35; }}
        .video-wrap {{ padding: 8px; }}
        .video-footer {{ display: grid; grid-template-columns: 1fr; gap: 6px; }}
        img {{ width: 100%; height: auto; display: block; border: 2px solid #334155; border-radius: 8px; }}
        {video_style}
        .button-grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
        .action-card {{ background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 10px; }}
        .action-card button {{ width: 100%; border: none; border-radius: 8px; padding: 9px 10px; font-size: 0.98rem; background: #2563eb; color: white; cursor: pointer; }}
        .action-card button:hover {{ background: #1d4ed8; }}
        .action-card button:disabled {{ background: #475569; cursor: not-allowed; }}
        .action-desc {{ margin-top: 8px; color: #cbd5e1; font-size: 0.88rem; line-height: 1.4; }}
        .help-list {{ margin: 0; padding-left: 18px; font-size: 0.9rem; line-height: 1.45; columns: 2; column-gap: 18px; }}
        .help-list li {{ break-inside: avoid; margin-bottom: 4px; }}
        .bottom-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
        .compact-bottom {{ grid-template-columns: 1fr; }}
        .extra-panel {{ margin-top: 0; }}
        .compact-title {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }}
        .compact-title h3 {{ margin: 0; }}
        .ready-note {{ margin-top: 6px; color: #cbd5e1; font-size: 0.88rem; }}
        @media (max-width: 1180px) {{
            .layout {{ grid-template-columns: 1fr; }}
            .bottom-grid {{ grid-template-columns: 1fr; }}
            .help-list {{ columns: 1; }}
        }}
    </style>
</head>
<body>
    <div class=\"container\">
        <div class=\"topbar\">
            <div class=\"title-block\">
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div class=\"badge-row\">
                <span class=\"badge\">{control_notice}</span>
                <span class=\"badge\">控制通道：<span id=\"control-status\" class=\"red\">未连接</span></span>
                <span class=\"badge\">监看通道：<span id=\"monitor-status\" class=\"red\">未连接</span></span>
            </div>
        </div>

        <div class=\"layout\">
            <div class=\"left-col\">
                <div class=\"panel video-wrap\">
                    <img id=\"video\" width=\"768\" height=\"256\" />
                </div>
                <div class=\"video-footer\">
{video_footer_html}
                </div>
            </div>

            <div class=\"right-col\">
                <div class=\"panel status\">
                    <div class=\"compact-title\">
                        <h3>当前阶段</h3>
                        <span id=\"mode\" class=\"yellow\">UNKNOWN</span>
                    </div>
                    <div><span id=\"stage-title\" class=\"green\">等待页面初始化</span></div>
                    <div id=\"stage-description\" style=\"margin-top: 6px;\">浏览器连上之后，服务端会把当前阶段说明推送到这里。</div>
                    <div id=\"ready-note\" class=\"ready-note\">等待控制通道和监看通道都连接成功后，再点击按钮或开始键盘控制。</div>
                </div>

                <div class=\"panel\">
                    <h3>当前可选按钮</h3>
                    <div id=\"button-grid\" class=\"button-grid\">
                        <div class=\"action-card\">
                            <button disabled>等待服务端推送按钮</button>
                            <div class=\"action-desc\">当前还没有可点击的操作按钮。</div>
                        </div>
                    </div>
                </div>

                <div class=\"panel status status-tight\">
                    <h3>吸盘与连接状态</h3>
                    <div>吸盘目标状态：<span id=\"suction-desired\">OFF</span></div>
                    <div>吸盘实际开启：<span id=\"suction-on\">False</span></div>
                    <div>吸盘是否锁附：<span id=\"suction-latched\">False</span></div>
                </div>

{sidebar_help_html}
            </div>
        </div>
{lower_section_html}
    </div>

    <script>
        const CONTROL_ENABLED = {str(control_enabled).lower()};
        const REQUIRE_MONITOR_FOR_ACTIONS = {require_monitor_literal};
        const imgEl = document.getElementById('video');
        const modeEl = document.getElementById('mode');
        const stageTitleEl = document.getElementById('stage-title');
        const stageDescriptionEl = document.getElementById('stage-description');
        const buttonGridEl = document.getElementById('button-grid');
        const suctionDesiredEl = document.getElementById('suction-desired');
        const suctionOnEl = document.getElementById('suction-on');
        const suctionLatchedEl = document.getElementById('suction-latched');
        const wrenchContactEl = document.getElementById('wrench-contact');
        const wrenchTotalEl = document.getElementById('wrench-total');
        const extraEl = document.getElementById('extra');
        const controlStatusEl = document.getElementById('control-status');
        const monitorStatusEl = document.getElementById('monitor-status');
        const readyNoteEl = document.getElementById('ready-note');
        let wsControl = null;
        let wsMonitor = null;
        let lastObjectUrl = null;
        const pressed = new Set();
        let controlConnected = false;
        let monitorConnected = false;

        function pageReady() {{
            return controlConnected && (monitorConnected || !REQUIRE_MONITOR_FOR_ACTIONS);
        }}

        function updateConnectionBadges() {{
            controlStatusEl.textContent = controlConnected ? '已连接' : '未连接';
            controlStatusEl.className = controlConnected ? 'green' : 'red';
            monitorStatusEl.textContent = monitorConnected ? '已连接' : '未连接';
            monitorStatusEl.className = monitorConnected ? 'green' : 'red';
            if (readyNoteEl) {{
                readyNoteEl.textContent = pageReady()
                    ? '页面控制已就绪：现在可以点击按钮，采集阶段也可以直接用浏览器键盘控制。'
                    : '页面仍在连接中：请等待控制通道和监看通道都显示为已连接。';
            }}
        }}

        function formatWrench(w) {{
            return (
                `fx=${{w[0].toFixed(3)}} fy=${{w[1].toFixed(3)}} fz=${{w[2].toFixed(3)}}\n` +
                `tx=${{w[3].toFixed(3)}} ty=${{w[4].toFixed(3)}} tz=${{w[5].toFixed(3)}}`
            );
        }}

        function sendMessage(obj) {{
            if (wsControl && wsControl.readyState === WebSocket.OPEN) {{
                wsControl.send(JSON.stringify(obj));
            }}
        }}

        function renderButtons(buttons) {{
            if (!buttons || buttons.length === 0) {{
                buttonGridEl.innerHTML = `
                    <div class=\"action-card\">
                        <button disabled>当前无可点击按钮</button>
                        <div class=\"action-desc\">如果页面处于自动播放或采集中，这里暂时不会显示额外按钮。</div>
                    </div>
                `;
                return;
            }}

            buttonGridEl.innerHTML = '';
            for (const button of buttons) {{
                const card = document.createElement('div');
                card.className = 'action-card';

                const actionBtn = document.createElement('button');
                actionBtn.textContent = button.label || button.button_id || '未命名按钮';
                actionBtn.disabled = !pageReady();
                actionBtn.onclick = () => {{
                    if (!pageReady()) {{
                        return;
                    }}
                    actionBtn.disabled = true;
                    sendMessage({{ type: 'button', button_id: button.button_id || '' }});
                    window.setTimeout(() => {{
                        actionBtn.disabled = !pageReady();
                    }}, 250);
                }};
                card.appendChild(actionBtn);

                const desc = document.createElement('div');
                desc.className = 'action-desc';
                desc.textContent = button.description || '这个按钮当前没有额外说明。';
                card.appendChild(desc);

                buttonGridEl.appendChild(card);
            }}
        }}

        function updateState(data) {{
            stageTitleEl.textContent = data.stage_title || '未知阶段';
            stageDescriptionEl.textContent = data.stage_description || '当前阶段没有额外说明。';
            modeEl.textContent = data.mode_label;
            suctionDesiredEl.textContent = data.suction_desired > 0 ? 'ON' : 'OFF';
            suctionOnEl.textContent = String(data.suction_on);
            suctionLatchedEl.textContent = String(data.suction_latched);
            const wContact = data.suction_wrench || [0, 0, 0, 0, 0, 0];
            const wTotal = data.suction_wrench_total_load || [0, 0, 0, 0, 0, 0];
            if (wrenchContactEl) {{ wrenchContactEl.textContent = formatWrench(wContact); }}
            if (wrenchTotalEl) {{ wrenchTotalEl.textContent = formatWrench(wTotal); }}
            renderButtons(data.action_buttons || []);
            if (extraEl) {{ extraEl.textContent = (data.extra_lines || []).join('\\n'); }}
        }}

        function clearPressedKeys() {{
            if (!CONTROL_ENABLED) {{
                return;
            }}
            const hadPressed = pressed.size > 0;
            pressed.clear();
            if (!controlConnected) {{
                return;
            }}
            if (!hadPressed) {{
                sendMessage({{ type: 'clear_keys' }});
                return;
            }}
            sendMessage({{ type: 'clear_keys' }});
        }}

        function handleKeyDown(event) {{
            if (!CONTROL_ENABLED) {{
                return;
            }}
            if (!pageReady()) {{
                return;
            }}

            const code = event.code;
            const relevantCodes = new Set(['KeyW', 'KeyS', 'KeyA', 'KeyD', 'ArrowUp', 'ArrowDown', 'Space', 'KeyH', 'KeyR', 'Escape']);
            if (!relevantCodes.has(code)) {{
                return;
            }}

            event.preventDefault();

            if (pressed.has(code)) {{
                return;
            }}

            pressed.add(code);
            sendMessage({{ type: 'keydown', code }});
        }}

        function handleKeyUp(event) {{
            if (!CONTROL_ENABLED) {{
                return;
            }}
            if (!pageReady()) {{
                return;
            }}

            const code = event.code;
            if (!pressed.has(code)) {{
                return;
            }}

            event.preventDefault();
            pressed.delete(code);
            sendMessage({{ type: 'keyup', code }});
        }}

        function attachKeyboardHandlers() {{
            window.addEventListener('keydown', handleKeyDown);
            window.addEventListener('keyup', handleKeyUp);
            window.addEventListener('blur', clearPressedKeys);
            document.addEventListener('visibilitychange', () => {{
                if (document.hidden) {{
                    clearPressedKeys();
                }}
            }});
        }}

        function connectControl() {{
            const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
            wsControl = new WebSocket(`${{proto}}://${{window.location.host}}/ws/control`);

            wsControl.onopen = () => {{
                controlConnected = true;
                updateConnectionBadges();
                renderButtons([]);
            }};

            wsControl.onerror = (event) => {{
                console.error('control websocket error', event);
                if (readyNoteEl) {{
                    readyNoteEl.textContent = '控制通道发生错误：请打开浏览器控制台查看详情，然后等待自动重连。';
                }}
            }};

            wsControl.onclose = () => {{
                controlConnected = false;
                clearPressedKeys();
                updateConnectionBadges();
                renderButtons([]);
                setTimeout(connectControl, 500);
            }};
        }}

        function connectMonitor() {{
            const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
            wsMonitor = new WebSocket(`${{proto}}://${{window.location.host}}/ws/monitor`);
            wsMonitor.binaryType = 'blob';

            wsMonitor.onopen = () => {{
                monitorConnected = true;
                updateConnectionBadges();
                renderButtons([]);
            }};

            wsMonitor.onmessage = (event) => {{
                if (typeof event.data === 'string') {{
                    updateState(JSON.parse(event.data));
                    return;
                }}

                const blob = event.data;
                const url = URL.createObjectURL(blob);
                imgEl.src = url;
                if (lastObjectUrl) {{
                    URL.revokeObjectURL(lastObjectUrl);
                }}
                lastObjectUrl = url;
            }};

            wsMonitor.onerror = (event) => {{
                console.error('monitor websocket error', event);
                if (extraEl) {{
                    extraEl.textContent = '监看通道发生错误：请打开浏览器控制台查看详情，然后等待自动重连。';
                }}
            }};

            wsMonitor.onclose = () => {{
                monitorConnected = false;
                updateConnectionBadges();
                renderButtons([]);
                setTimeout(connectMonitor, 500);
            }};
        }}

        attachKeyboardHandlers();
        updateConnectionBadges();
        connectControl();
        connectMonitor();
    </script>
</body>
</html>
"""


def start_monitor_server(
    monitor_state: RealtimeMonitorState,
    *,
    host: str,
    port: str | int,
    title: str,
    subtitle: str,
    keyboard_lines: list[str],
    control_state: BrowserKeyboardController | None = None,
    show_wrench_panels: bool = True,
    keyboard_help_position: str = "sidebar",
    require_monitor_for_actions: bool = True,
    video_max_width_px: int | None = None,
) -> tuple[threading.Thread, uvicorn.Server]:
    """
    启动 FastAPI + WebSocket 服务。

    如果传入 `control_state`，这个服务就是双向的：
    - 浏览器把按键状态发上来；
    - 服务端把状态和视频帧发回去。

    如果没传 `control_state`，它就退化成纯监看服务。
    """
    html = build_monitor_html(
        title=title,
        subtitle=subtitle,
        keyboard_lines=keyboard_lines,
        control_enabled=control_state is not None,
        show_wrench_panels=show_wrench_panels,
        keyboard_help_position=keyboard_help_position,
        require_monitor_for_actions=require_monitor_for_actions,
        video_max_width_px=video_max_width_px,
    )
    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(html)

    @app.websocket("/ws/monitor")
    async def monitor_websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        async def sender_loop():
            while not monitor_state.shutdown_event.is_set():
                payload, frame_bytes = monitor_state.snapshot()
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))
                if frame_bytes:
                    await websocket.send_bytes(frame_bytes)
                await _sleep_async(1.0 / max(monitor_state.update_hz, 1.0))

        try:
            await sender_loop()
        except WebSocketDisconnect:
            return
        except RuntimeError:
            return

    @app.websocket("/ws/control")
    async def control_websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        async def receiver_loop():
            if control_state is None:
                while not monitor_state.shutdown_event.is_set():
                    await _sleep_async(0.5)
                return

            while not monitor_state.shutdown_event.is_set():
                message_text = await websocket.receive_text()
                try:
                    message = json.loads(message_text)
                except json.JSONDecodeError:
                    continue
                control_state.handle_message(message)

        try:
            await receiver_loop()
        except WebSocketDisconnect:
            return
        except RuntimeError:
            return
        finally:
            if control_state is not None:
                control_state.clear_pressed_keys()

    # 当前环境里默认 WebSocket 后端存在兼容性问题，因此这里显式指定 uvicorn 的后端实现。
    # 这一步只是服务端兼容性设置；如果浏览器页面仍显示“未连接”，还需要同时排查前端脚本是否执行失败。
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        ws="websockets-sansio",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread, server



async def _sleep_async(duration: float):
    """延迟导入 asyncio，避免模块导入阶段拉长启动路径。"""
    import asyncio

    await asyncio.sleep(duration)
