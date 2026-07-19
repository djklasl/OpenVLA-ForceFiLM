"""
collect_libero_suction_handover_dataset_multiuser.py

【脚本作用】
这个脚本是 `collect_libero_suction_handover_dataset.py` 的三人并行采集版本。
它专门服务于 `libero_suction` 任务套件，并且把剩余任务固定分成 3 组：
- worker_a 负责 5 个 task
- worker_b 负责 5 个 task
- worker_c 负责 5 个 task

每次运行时，用户只需要指定 `--worker_name`，脚本就只会处理这一组 task，
并把采集结果写到独立目录里，避免多个采集人员同时写同一个目录产生冲突。

【输入是什么】
1. `--worker_name`
   表示当前这个浏览器实例是给哪一位采集人员使用的。
2. `--libero_raw_data_dir`
   表示官方原始 LIBERO HDF5 所在目录。
3. 其他参数如 `--web_host`、`--web_port`、`--resolution` 等，
   含义和原单人脚本保持一致。

【输出是什么】
1. 每个 worker 自己独立的 HDF5 采集目录，例如：
   - /data1/ps/openvla/data_regeneration/libero_suction_worker_a
   - /data1/ps/openvla/data_regeneration/libero_suction_worker_b
   - /data1/ps/openvla/data_regeneration/libero_suction_worker_c
2. 每个 worker 自己独立的 metainfo JSON。
3. 浏览器监看页面，供远程人员直接通过服务器 IP + 端口访问。

【这个脚本为什么单独存在】
因为原始单人脚本会固定写到 `data_regeneration/libero_suction`，
而且会扫描整套 task。那样如果 3 个人同时并行采集，就会出现：
- 多个人同时写同一个输出目录
- 多个人同时写同一个 metainfo JSON
- 多个人同时竞争一个任务列表

所以这里单独复制出一份“多 worker 并行版”脚本，目的就是：
- 不影响原来的单人脚本
- 让 3 个浏览器实例可以稳定并行工作
- 每个人只看到自己该做的 5 个 task

python /data1/ps/openvla/experiments/robot/libero/data_regeneration/multi_worker/collect_libero_suction_handover_dataset_multiuser.py \
    --worker_name worker_a \
    --libero_raw_data_dir /data1/ps/openvla/LIBERO/libero/datasets/libero_suction \
    --web_port 8081

"""

from __future__ import annotations
# 开启更灵活的类型注解语法，便于直接写 list[str] 这种类型。

import argparse
# 负责解析命令行参数。
import os
# 负责设置环境变量，例如 MuJoCo 的离屏渲染后端。
import sys
# 负责修改 Python 模块搜索路径，确保能 import 到仓库里的模块。
from pathlib import Path
# Path 用来安全地拼接路径，避免直接手写字符串路径。

import h5py
# 用于读取 HDF5，统计原始 demo 数量，以及扫描当前 worker 已完成多少条 demo。
import numpy as np
# 用于创建“全部完成”时的空白画面。

# ------------------------------------------------------------------------------
# 路径初始化
# ------------------------------------------------------------------------------
# ROOT_DIR 指向 openvla 仓库根目录。
ROOT_DIR = Path(__file__).resolve().parents[5]

# LIBERO_DIR 指向仓库里自带的 LIBERO 子目录。
LIBERO_DIR = ROOT_DIR / "LIBERO"

# 继续沿用当前项目里 data_regeneration 作为第一阶段原始采集输出根目录。
DATA_OUTPUT_ROOT = ROOT_DIR / "data_regeneration"

# 离屏渲染仍然默认走 EGL，保证在服务器无头环境下能跑。
os.environ.setdefault("MUJOCO_GL", "egl")

# 把仓库根目录和 LIBERO 子目录加进搜索路径，确保下面的 import 能找到本地代码。
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(LIBERO_DIR))

# 这里直接复用原单人脚本里已经写好的大量稳定逻辑。
# 这样我们只需要额外控制“任务分组”“输出目录”“worker 页面文案”。
from experiments.robot.libero.data_regeneration import collect_libero_suction_handover_dataset as base


# ------------------------------------------------------------------------------
# 固定的 worker 任务分组
# ------------------------------------------------------------------------------
# 这里把 15 个剩余 task 明确分成 3 组，每组 5 个。
# 这样每个 worker 的工作量相同，而且后续不需要再手工临时分配。
WORKER_TASK_GROUPS: dict[str, list[str]] = {
    "worker_a": [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    ],
    "worker_b": [
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ],
    "worker_c": [
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_bowl_on_the_stove",
        "put_the_bowl_on_top_of_the_cabinet",
        "put_the_cream_cheese_in_the_bowl",
    ],
}


def default_output_dir_for_worker(worker_name: str) -> Path:
    """
    根据 worker 名字计算默认输出目录。

    例如：
    - worker_a -> data_regeneration/libero_suction_worker_a
    - worker_b -> data_regeneration/libero_suction_worker_b
    """
    return DATA_OUTPUT_ROOT / f"libero_suction_{worker_name}"


def default_metainfo_path_for_worker(worker_name: str, output_dir: Path) -> Path:
    """
    根据 worker 名字和输出目录计算默认 metainfo JSON 路径。

    这里刻意把 metainfo 文件名也带上 worker 名字，目的有两个：
    1. 一眼看出这是哪个 worker 的记录；
    2. 后续做目录合并时，不会和正式目录中的总 metainfo 混淆。
    """
    return output_dir / f"libero_suction_{worker_name}_raw_collection_metainfo.json"


def build_worker_task_progress_list(task_suite, allowed_task_names: set[str], raw_data_dir: Path, target_dir: Path) -> list[dict]:
    """
    为当前 worker 构造“自己的任务完成度列表”。

    这个函数只会扫描当前 worker 被分到的 5 个 task，不会扫描整套 libero_suction。
    返回的每个字典都复用原脚本里的字段结构，后面主循环就可以继续沿用原来的逻辑。
    """
    task_progress_list: list[dict] = []
    found_task_names: set[str] = set()

    # 遍历 benchmark 里的整套任务，但只保留当前 worker 被分配到的那些 task。
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)

        # 如果这个 task 不属于当前 worker，就直接跳过。
        if task.name not in allowed_task_names:
            continue

        found_task_names.add(task.name)

        # 官方原始 HDF5 的路径固定来自 libero_suction 数据集目录。
        orig_data_path = raw_data_dir / f"{task.name}_demo.hdf5"
        if not orig_data_path.exists():
            raise FileNotFoundError(f"Cannot find raw data file: {orig_data_path}")

        # 读取原始 demo 数量，后面要用它判断当前 worker 的这个 task 是否已经全部完成。
        with h5py.File(orig_data_path, "r") as orig_file:
            raw_demo_count = len(orig_file["data"].keys())

        # 当前 worker 自己的输出文件路径。
        target_data_path = target_dir / f"{task.name}_demo.hdf5"

        # 读取当前 worker 已经完成的 demo 索引。
        completed_demo_indices = base.list_completed_demo_indices(target_data_path)
        completed_demo_count = len(completed_demo_indices)

        # 如果已完成数量大于等于原始 demo 总数，就说明这个 task 在当前 worker 目录下已经采完。
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

    # 如果 benchmark 里没找到某个预期 task，说明任务名写错了，必须立刻报错。
    missing_task_names = sorted(allowed_task_names - found_task_names)
    if missing_task_names:
        raise ValueError(f"These worker tasks are missing from libero_suction benchmark: {missing_task_names}")

    return task_progress_list


def sort_pending_task_list(task_progress_list: list[dict]) -> list[dict]:
    """
    复用原脚本的排序规则，把未完成 task 排出一个稳定顺序。

    排序规则保持不变：
    1. 半完成 task 优先；
    2. 已完成 demo 更多的优先；
    3. 再按 task_id 排序。
    """
    pending_task_list = [info for info in task_progress_list if not info["is_completed"]]
    pending_task_list.sort(
        key=lambda info: (
            0 if info["completed_demo_count"] > 0 else 1,
            -int(info["completed_demo_count"]),
            int(info["task_id"]),
        )
    )
    return pending_task_list


def main(args):
    """
    三人并行版的主流程。

    这份主流程和原单人脚本最大的不同有 4 点：
    1. 只处理当前 worker 被分配到的 5 个 task；
    2. 输出写到当前 worker 自己的目录；
    3. metainfo 写到当前 worker 自己的 JSON；
    4. 浏览器页面标题和额外信息里会显示当前 worker 名字。
    """
    # 先从固定分组表里取出当前 worker 负责的 5 个任务。
    worker_task_names = WORKER_TASK_GROUPS[args.worker_name]
    worker_task_name_set = set(worker_task_names)

    # 如果用户没手工指定输出目录，就用 worker 专属的默认输出目录。
    target_dir = Path(args.output_dir) if args.output_dir else default_output_dir_for_worker(args.worker_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    # metainfo 也默认写到 worker 自己的目录里。
    metainfo_json_out_path = (
        Path(args.metainfo_path)
        if args.metainfo_path
        else default_metainfo_path_for_worker(args.worker_name, target_dir)
    )

    # 先把之前可能已经存在的 metainfo 读回来，保证断点续采时不会覆盖历史记录。
    metainfo_json_dict = base.load_metainfo_json(metainfo_json_out_path)

    # 浏览器控制器继续沿用原脚本那一套实现。
    controller = base.BrowserKeyboardController(
        movement_speed=args.movement_speed,
        enable_handover=True,
        reset_suction_on_reset=True,
    )

    # 监看状态对象也继续沿用原脚本，只是标题里带上 worker 名字，方便页面上区分。
    monitor_state = base.RealtimeMonitorState(
        title=f"LIBERO 吸盘原始采集监看（{args.worker_name}）",
        jpeg_quality=70,
        update_hz=base.DEFAULT_MENU_MONITOR_HZ,
    )

    # 启动浏览器监看服务。这里不加 token，不做鉴权，只靠不同端口区分不同 worker。
    _, server = base.start_monitor_server(
        monitor_state,
        host=args.web_host,
        port=args.web_port,
        title=f"LIBERO 吸盘数据采集监看（{args.worker_name}）",
        subtitle="这是三人并行采集专用页面。当前页面只会显示当前 worker 被分配到的 5 个 task。",
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

    # 打印当前实例的运行信息，便于你把 URL 发给对应人员。
    print("=" * 80)
    print("LIBERO 吸盘三人并行采集服务已启动")
    print(f"当前 worker：{args.worker_name}")
    print(f"浏览器监看页面: http://{args.web_host if args.web_host != '0.0.0.0' else '127.0.0.1'}:{args.web_port}")
    print(f"当前 worker 输出目录: {target_dir}")
    print("当前 worker 负责的 task：")
    for task_name in worker_task_names:
        print(f"  - {task_name}")
    print("=" * 80)

    # 多人并行版仍然复用正式的 libero_suction benchmark。
    benchmark_dict = base.benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()

    # 原始官方数据目录。
    raw_data_dir = Path(args.libero_raw_data_dir)

    # 只为当前 worker 的 5 个任务构造完成度列表。
    task_progress_list = build_worker_task_progress_list(
        task_suite=task_suite,
        allowed_task_names=worker_task_name_set,
        raw_data_dir=raw_data_dir,
        target_dir=target_dir,
    )

    # 只对当前 worker 的任务做“待办排序”。
    pending_task_list = sort_pending_task_list(task_progress_list)

    # 如果当前 worker 的 5 个 task 全都已经完成，就直接显示“全部完成”页面。
    if not pending_task_list:
        base.push_monitor_state(
            monitor_state=monitor_state,
            controller=controller,
            frame_rgb=np.zeros((args.resolution, args.resolution * 3, 3), dtype=np.uint8),
            mode_label=base.STAGE_ALL_DONE,
            stage_title="当前 worker 的任务已全部完成",
            stage_description="当前 worker 被分配的 5 个 task 都已经采完了。你现在可以直接关闭页面。",
            action_buttons=[],
            obs=None,
            info=None,
            extra_lines=[
                f"当前 worker：{args.worker_name}",
                f"当前 worker 输出目录：{target_dir}",
                "当前 worker 已无待采集 task。",
            ],
            update_hz=base.DEFAULT_MENU_MONITOR_HZ,
        )
        print(f"{args.worker_name} 的所有 task 已完成，输出目录：{target_dir}")
        monitor_state.shutdown_event.set()
        server.should_exit = True
        return

    try:
        # 下面这整段流程基本沿用原单人脚本，只是范围缩小到当前 worker 的任务子集。
        for task_info in pending_task_list:
            task = task_info["task"]
            task_description = task.language
            orig_data_path = task_info["orig_data_path"]
            target_data_path = task_info["target_data_path"]

            # 打开官方原始数据，供预览和自动前段参考使用。
            orig_data_file = h5py.File(orig_data_path, "r")
            orig_data = orig_data_file["data"]

            # 目标 HDF5 用追加模式打开，以支持断点续采。
            new_data_file = h5py.File(target_data_path, "a")
            grp = new_data_file["data"] if "data" in new_data_file else new_data_file.create_group("data")

            # 创建 suction 环境和 default 环境。
            env_suction = base.create_suction_env(task, args.resolution, args.control_freq)
            env_default = base.create_default_env(task, args.resolution, args.control_freq)
            env_suction.seed(0)
            env_default.seed(0)

            try:
                # 先找一条成功 demo 作为预览。
                preview_demo_index, found_success = base.find_first_successful_preview_demo_index(
                    orig_data,
                    env_default,
                    args.settle_steps,
                )
                preview_demo_data = orig_data[f"demo_{preview_demo_index}"]
                preview_initial_obs, preview_initial_info = base.prepare_env_at_demo_start(
                    env_default,
                    preview_demo_data["states"][0],
                    args.settle_steps,
                )
                preview_menu_frame = base.build_frame_from_obs(
                    preview_initial_obs,
                    preview_initial_info,
                    args.resolution,
                )

                # 刚进入一个新 task 时，要先显示“任务入口页”。
                first_task_menu = True

                while True:
                    # 进入循环前，重新读取当前 task 已完成多少个 demo，保证断点续采正确。
                    completed_demo_indices = base.list_completed_demo_indices(target_data_path)
                    next_demo_index = base.next_missing_demo_index(len(orig_data.keys()), completed_demo_indices)

                    # 如果没有缺失 demo，说明当前 task 已全部完成。
                    if next_demo_index is None:
                        print(f"worker={args.worker_name} task '{task_description}' 已全部采集完成。")
                        break

                    # 清掉上一个阶段可能残留的按钮事件和按键状态。
                    base.clear_transient_browser_events(controller)

                    if first_task_menu:
                        # 任务入口页：只显示“播放原始轨迹 / 退出”。
                        stage_description = (
                            "你现在进入了当前 worker 的一个新 task。先点击“播放原始LIBERO轨迹”，"
                            "完整看一遍成功示范，确认目标物体和目标位置。"
                        )
                        if not found_success:
                            stage_description += " 当前没有找到成功 demo，预览时会回退播放 demo_0。"

                        base.push_monitor_state(
                            monitor_state=monitor_state,
                            controller=controller,
                            frame_rgb=preview_menu_frame,
                            mode_label=base.STAGE_TASK_MENU,
                            stage_title="任务入口页",
                            stage_description=stage_description,
                            action_buttons=base.TASK_MENU_BUTTONS,
                            obs=preview_initial_obs,
                            info=preview_initial_info,
                            extra_lines=[
                                f"当前 worker：{args.worker_name}",
                                f"当前任务：{task_description}",
                                f"下一条待采集示范：demo_{next_demo_index}",
                                f"当前将播放的预览示范：demo_{preview_demo_index}",
                                f"当前任务完成进度：{len(completed_demo_indices)} / {len(orig_data.keys())}",
                            ],
                            update_hz=base.DEFAULT_MENU_MONITOR_HZ,
                        )

                        button_choice = base.wait_for_button_choice(
                            controller,
                            allowed_button_ids={base.BUTTON_PREVIEW, base.BUTTON_EXIT},
                        )
                        if button_choice == base.BUTTON_EXIT:
                            raise KeyboardInterrupt("用户在任务入口页点击退出。")

                        # 播放原始轨迹预览。
                        _, _, preview_menu_frame = base.play_preview_demo(
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

                    # 预览完成页：显示“数据采集 / 重播 / 退出”。
                    while True:
                        base.clear_transient_browser_events(controller)

                        base.push_monitor_state(
                            monitor_state=monitor_state,
                            controller=controller,
                            frame_rgb=preview_menu_frame,
                            mode_label=base.STAGE_POST_PREVIEW_MENU,
                            stage_title="预览完成页",
                            stage_description=(
                                "你已经看完当前 task 的原始成功轨迹。"
                                "如果已经确认清楚目标物体和目标位置，就点击“数据采集”；"
                                "如果还想再看一遍，就点“重播”；如果不想继续，就点“退出”。"
                            ),
                            action_buttons=base.POST_PREVIEW_BUTTONS,
                            obs=preview_initial_obs,
                            info=preview_initial_info,
                            extra_lines=[
                                f"当前 worker：{args.worker_name}",
                                f"当前任务：{task_description}",
                                f"下一条待采集示范：demo_{next_demo_index}",
                                f"当前预览示范：demo_{preview_demo_index}",
                                f"当前任务完成进度：{len(completed_demo_indices)} / {len(orig_data.keys())}",
                            ],
                            update_hz=base.DEFAULT_MENU_MONITOR_HZ,
                        )

                        button_choice = base.wait_for_button_choice(
                            controller,
                            allowed_button_ids={base.BUTTON_COLLECT, base.BUTTON_REPLAY, base.BUTTON_EXIT},
                        )
                        if button_choice == base.BUTTON_EXIT:
                            raise KeyboardInterrupt("用户在预览完成页点击退出。")

                        if button_choice == base.BUTTON_REPLAY:
                            # 如果用户还想再看一遍原始轨迹，就重新播放一次。
                            _, _, preview_menu_frame = base.play_preview_demo(
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

                        # 只有点击“数据采集”，才会跳出菜单，真正采 demo。
                        break

                    # 调用原单人脚本里的单条 demo 采集逻辑。
                    collect_result = base.collect_one_demo(
                        task=task,
                        demo_index=next_demo_index,
                        orig_data=orig_data,
                        env_suction=env_suction,
                        env_default=env_default,
                        controller=controller,
                        monitor_state=monitor_state,
                        args=args,
                    )

                    # 如果用户按了 R，则当前 demo 直接作废，回到“数据采集 / 重播 / 退出”页面。
                    if collect_result["status"] == "restart":
                        continue

                    # 正常采完一条 demo，就把它正式写入 worker 自己的 HDF5。
                    base.save_episode_to_hdf5(
                        grp,
                        demo_name=f"demo_{next_demo_index}",
                        buffers=collect_result["buffers"],
                        success=collect_result["success"],
                        handover_step=collect_result["handover_step"],
                        source_demo_index=collect_result["source_demo_index"],
                    )
                    new_data_file.flush()

                    # 同时更新 worker 自己的 metainfo JSON。
                    metainfo_json_dict.setdefault(task.name, {})
                    metainfo_json_dict[task.name][f"demo_{next_demo_index}"] = {
                        "success": bool(collect_result["success"]),
                        "handover_step": int(collect_result["handover_step"]),
                        "source_demo_index": int(collect_result["source_demo_index"]),
                        "demo_source": "auto_then_manual",
                        "teleop_device": "browser_websocket",
                        "worker_name": args.worker_name,
                    }
                    base.save_metainfo_json(metainfo_json_out_path, metainfo_json_dict)

                    print(
                        f"worker={args.worker_name} 已保存 {task.name} / demo_{next_demo_index}："
                        f"success={collect_result['success']}, "
                        f"handover_step={collect_result['handover_step']}"
                    )
            finally:
                # 当前 task 无论正常结束还是中途退出，都要把环境和文件句柄关掉。
                env_default.close()
                env_suction.close()
                orig_data_file.close()
                new_data_file.close()
                print(f"worker={args.worker_name} task '{task_description}' 当前输出文件：{target_data_path}")
    finally:
        # 脚本结束时，确保监看线程和 uvicorn 服务都收到退出信号。
        monitor_state.shutdown_event.set()
        server.should_exit = True
        print(f"worker={args.worker_name} 原始采集流程结束。输出目录：{target_dir}")
        print(f"worker={args.worker_name} metainfo JSON：{metainfo_json_out_path}")


if __name__ == "__main__":
    # 这里定义多 worker 版本支持的全部命令行参数。
    parser = argparse.ArgumentParser()

    # 这个脚本只服务于 libero_suction，所以这里把 suite choice 锁死。
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_suction"],
        default="libero_suction",
        help="这个并行脚本只用于 libero_suction。",
    )
    parser.add_argument(
        "--worker_name",
        type=str,
        choices=sorted(WORKER_TASK_GROUPS.keys()),
        required=True,
        help="当前浏览器实例属于哪个 worker。",
    )
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        required=True,
        help="官方原始 LIBERO HDF5 数据集目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="当前 worker 的输出目录。不传时会自动写到 data_regeneration/libero_suction_worker_x。",
    )
    parser.add_argument(
        "--metainfo_path",
        type=str,
        default=None,
        help="当前 worker 的 metainfo JSON 路径。不传时默认写到当前 worker 输出目录下。",
    )
    parser.add_argument("--resolution", type=int, default=base.DEFAULT_RESOLUTION, help="相机图像分辨率。")
    parser.add_argument("--control_freq", type=int, default=base.DEFAULT_CONTROL_FREQ, help="环境控制频率。")
    parser.add_argument("--settle_steps", type=int, default=base.DEFAULT_SETTLE_STEPS, help="初始化后等待环境稳定的步数。")
    parser.add_argument("--manual_max_steps", type=int, default=base.DEFAULT_MANUAL_MAX_STEPS, help="handover 后允许的最大人工控制步数。")
    parser.add_argument("--web_host", type=str, default="0.0.0.0", help="浏览器监看页面监听地址。")
    parser.add_argument("--web_port", type=int, required=True, help="浏览器监看页面监听端口。")
    parser.add_argument("--movement_speed", type=float, default=0.3, help="人工模式下浏览器键盘平移速度。")
    parser.add_argument(
        "--auto_replay_speed_scale",
        type=float,
        default=base.DEFAULT_AUTO_REPLAY_SPEED_SCALE,
        help="自动回放阶段的速度倍率。默认 0.5 表示自动段按 0.5x 速度回放。",
    )

    # 解析参数并进入主流程。
    main(parser.parse_args())
