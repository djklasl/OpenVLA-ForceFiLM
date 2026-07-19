"""
inspect_failed_suction_regeneration.py

导出 stage-2 再生成中被判失败的 suction demo 回放视频，帮助人工确认这些 demo 是否真的失败。

这个脚本不会修改任何原始采集 HDF5，也不会修改 processed HDF5；
它只会读取：
- data_regeneration/{task_suite} 下的 stage-1 原始采集 HDF5
- data_regeneration/processed_{task_suite}/{task_suite}_metainfo.json

然后按 stage-2 当前逻辑重新 replay 被判失败的 demo，并导出 mp4。

运行示例：
1. 导出当前 suite 全部失败 demo 视频
   python experiments/robot/libero/data_regeneration/inspect_failed_suction_regeneration.py        --libero_task_suite libero_object        --libero_raw_data_dir ./data_regeneration/libero_object

2. 只导出某个 task 的失败 demo
   python experiments/robot/libero/data_regeneration/inspect_failed_suction_regeneration.py        --libero_task_suite libero_object        --libero_raw_data_dir ./data_regeneration/libero_object        --task_name pick_up_the_orange_juice_and_place_it_in_the_basket

3. 只导出某一条失败 demo
   python experiments/robot/libero/data_regeneration/inspect_failed_suction_regeneration.py        --libero_task_suite libero_object        --libero_raw_data_dir ./data_regeneration/libero_object        --task_name pick_up_the_orange_juice_and_place_it_in_the_basket        --demo_index 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import tqdm

ROOT_DIR = Path(__file__).resolve().parents[4]
LIBERO_DIR = ROOT_DIR / "LIBERO"

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(LIBERO_DIR))

from libero.libero import benchmark

from experiments.robot.libero.data_regeneration.regenerate_suction_libero_dataset import (
    DEFAULT_CONTROL_FREQ,
    DEFAULT_DUMMY_STEPS,
    DEFAULT_RESOLUTION,
    DEFAULT_TAIL_DUMMY_STEPS,
    build_replay_actions,
    create_suction_env,
    default_processed_dir_for_suite,
)
from experiments.robot.libero.libero_utils import get_libero_dummy_action, save_rollout_video


def load_failed_demo_map(metainfo_path: Path) -> dict[str, list[int]]:
    """从 stage-2 metainfo 中读取失败 demo 列表。"""
    with open(metainfo_path, 'r', encoding='utf-8') as f:
        metainfo = json.load(f)

    failed_demo_map: dict[str, list[int]] = {}
    for task_key, demos in metainfo.items():
        failed_indices = []
        for demo_name, info in sorted(demos.items(), key=lambda kv: int(kv[0].split('_')[1])):
            if not bool(info.get('success', False)):
                failed_indices.append(int(demo_name.split('_')[1]))
        if failed_indices:
            failed_demo_map[task_key] = failed_indices
    return failed_demo_map


def filter_failed_demo_map(failed_demo_map: dict[str, list[int]], task_name: str | None, demo_index: int | None) -> dict[str, list[int]]:
    """按 task_name / demo_index 过滤失败 demo。"""
    filtered = failed_demo_map
    if task_name is not None:
        filtered = {k: v for k, v in filtered.items() if k == task_name}
    if demo_index is not None:
        filtered = {k: [i for i in v if i == demo_index] for k, v in filtered.items()}
        filtered = {k: v for k, v in filtered.items() if v}
    return filtered


def replay_failed_demo(env, demo_data, tail_dummy_steps: int) -> tuple[list, dict]:
    """按最新 stage-2 逻辑重放一条失败 demo，并返回视频帧与回放统计。"""
    raw_actions = demo_data['actions'][()]
    raw_states = demo_data['states'][()]

    env.reset()
    obs = env.set_init_state(raw_states[0])
    info = {}
    dummy_success = True
    for _ in range(DEFAULT_DUMMY_STEPS):
        obs, _, done, info = env.step(get_libero_dummy_action('openvla'))
        if done:
            dummy_success = False
            break

    replay_actions, num_noops_filtered = build_replay_actions(raw_actions, tail_dummy_steps)
    rollout_images = []
    done = False

    if dummy_success:
        for action in replay_actions:
            rollout_images.append(obs['agentview_image'].copy())
            obs, _, done, info = env.step(action.tolist())
            if done:
                rollout_images.append(obs['agentview_image'].copy())
                break

    success = bool(dummy_success and (done or env.check_success()))
    stats = {
        'raw_steps': int(len(raw_actions)),
        'num_noops_filtered': int(num_noops_filtered),
        'kept_steps': int(max(len(replay_actions) - tail_dummy_steps, 0)),
        'final_replay_steps': int(len(replay_actions)),
        'tail_dummy_steps_appended': int(tail_dummy_steps),
        'replay_success': bool(success),
    }
    return rollout_images, stats


def build_task_lookup(task_suite) -> dict[str, object]:
    """
    为当前 benchmark suite 建立一个“任务查找表”。

    为什么这里不能只用 task.name：
    - stage-2 的 metainfo JSON 里，task 键使用的是
      `task.language.replace(" ", "_")`
    - 但 benchmark 里的真实 task 文件名使用的是 `task.name`
    - 对带场景前缀的任务，这两者并不相同，例如：
      - metainfo 里可能是 `put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
      - 真实 task.name 则是 `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`

    所以这里必须同时支持两种 key：
    1. task.name
    2. task.language.replace(" ", "_")

    这样 inspect 脚本才能既兼容旧的命名方式，也兼容当前 stage-2 metainfo 的写法。
    """
    task_lookup: dict[str, object] = {}

    for task_index in range(task_suite.n_tasks):
        # 先拿到 benchmark 里的原始 task 对象。
        task = task_suite.get_task(task_index)

        # 第一种 key：真实的 task.name，对应 HDF5 文件名和 benchmark 注册名。
        task_lookup[task.name] = task

        # 第二种 key：stage-2 metainfo 实际写入的 task key。
        # 这里与 regenerate_suction_libero_dataset.py 里的写法保持严格一致。
        language_key = task.language.replace(" ", "_")
        task_lookup[language_key] = task

    return task_lookup


def main(args):
    raw_data_dir = Path(args.libero_raw_data_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve() if args.processed_dir is not None else default_processed_dir_for_suite(args.libero_task_suite)
    metainfo_path = Path(args.processed_metainfo_path).resolve() if args.processed_metainfo_path is not None else processed_dir / f'{args.libero_task_suite}_metainfo.json'

    if not metainfo_path.exists():
        raise FileNotFoundError(f'Cannot find processed metainfo JSON: {metainfo_path}')

    failed_demo_map = load_failed_demo_map(metainfo_path)
    failed_demo_map = filter_failed_demo_map(failed_demo_map, args.task_name, args.demo_index)
    if not failed_demo_map:
        raise ValueError('No failed demos matched the requested filters.')

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    # 这里不能只按 task.name 建表。
    # 因为 processed metainfo 里的 task key 用的是 language 版本，
    # 所以要建立一个“双键兼容”的查找表，避免再出现
    # “Task xxx not found in benchmark suite libero_suction” 这类误报。
    task_lookup = build_task_lookup(task_suite)

    export_subdir = args.output_subdir or f'failed_regeneration_{args.libero_task_suite}_after_tail_release'
    total_failures = sum(len(v) for v in failed_demo_map.values())
    print(f'Found {total_failures} failed demos to inspect.')

    with tqdm.tqdm(total=total_failures, desc='Failed demos', dynamic_ncols=True) as pbar:
        for task_name, failed_indices in failed_demo_map.items():
            if task_name not in task_lookup:
                raise KeyError(f'Task {task_name} not found in benchmark suite {args.libero_task_suite}')

            task = task_lookup[task_name]
            raw_data_path = raw_data_dir / f'{task.name}_demo.hdf5'
            if not raw_data_path.exists():
                raise FileNotFoundError(f'Cannot find stage-1 raw data file: {raw_data_path}')

            env = create_suction_env(task, args.resolution, args.control_freq)
            env.seed(0)
            env.reset()

            with h5py.File(raw_data_path, 'r') as raw_h5:
                raw_data = raw_h5['data']
                for demo_index in failed_indices:
                    demo_name = f'demo_{demo_index}'
                    if demo_name not in raw_data:
                        raise KeyError(f'{demo_name} not found in {raw_data_path}')

                    rollout_images, stats = replay_failed_demo(env, raw_data[demo_name], args.tail_dummy_steps)
                    task_desc = f'{task.name}__{demo_name}'
                    video_path = save_rollout_video(
                        rollout_images,
                        idx=demo_index,
                        success=stats['replay_success'],
                        task_description=task_desc,
                        subdir=export_subdir,
                    )
                    tqdm.tqdm.write(
                        f"Inspected {task.name}/{demo_name}: replay_success={stats['replay_success']}, "
                        f"raw_steps={stats['raw_steps']}, kept_steps={stats['kept_steps']}, "
                        f"final_replay_steps={stats['final_replay_steps']}, noops={stats['num_noops_filtered']}, "
                        f"tail_dummy_steps={stats['tail_dummy_steps_appended']}, video={video_path}"
                    )
                    pbar.update(1)
                    pbar.set_postfix(task=task.name, demo=demo_index)

            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--libero_task_suite',
        type=str,
        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90', 'libero_suction'],
        required=True,
        help='要检查的 LIBERO task suite 名称。',
    )
    parser.add_argument(
        '--libero_raw_data_dir',
        type=str,
        required=True,
        help='stage-1 原始采集 HDF5 目录。',
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default=None,
        help='stage-2 processed 目录；默认使用 data_regeneration/processed_{task_suite}。',
    )
    parser.add_argument(
        '--processed_metainfo_path',
        type=str,
        default=None,
        help='stage-2 metainfo JSON 路径；默认使用 processed 目录下的 {task_suite}_metainfo.json。',
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default=None,
        help='只检查某个 task.name 的失败 demo。',
    )
    parser.add_argument(
        '--demo_index',
        type=int,
        default=None,
        help='只检查某个 demo 索引。',
    )
    parser.add_argument(
        '--output_subdir',
        type=str,
        default=None,
        help='rollouts 下的视频子目录名；默认 failed_regeneration_{task_suite}。',
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=DEFAULT_RESOLUTION,
        help='相机图像分辨率。',
    )
    parser.add_argument(
        '--control_freq',
        type=int,
        default=DEFAULT_CONTROL_FREQ,
        help='环境控制频率。',
    )
    parser.add_argument(
        '--tail_dummy_steps',
        type=int,
        default=DEFAULT_TAIL_DUMMY_STEPS,
        help='回放时尾部额外追加多少步吸盘关闭的 dummy action。',
    )
    main(parser.parse_args())
