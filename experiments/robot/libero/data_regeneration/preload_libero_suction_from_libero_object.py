"""
preload_libero_suction_from_libero_object.py

这个脚本的作用：
1. 读取 `select_task/libero_suction_task_selection.json`。
2. 找出其中 `selection == "yes"` 且来源于 `libero_object` 的任务。
3. 把这些任务在 `data_regeneration/libero_object` 中已经采好的 stage-1 原始 HDF5，
   复制到 `data_regeneration/libero_suction`。
4. 同时把对应 task 的 metainfo 条目从
   `libero_object_raw_collection_metainfo.json` 合并到
   `libero_suction_raw_collection_metainfo.json`。

这样后续再运行：
`collect_libero_suction_handover_dataset.py --libero_task_suite libero_suction`
时，这些任务会被现有的“完成度扫描”逻辑直接视为已完成，不需要重复采集。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_SELECTION_JSON = ROOT_DIR / "select_task" / "libero_suction_task_selection.json"
DEFAULT_DATA_REGEN_ROOT = ROOT_DIR / "data_regeneration"
DEFAULT_SOURCE_SUITE_NAME = "libero_object"
DEFAULT_TARGET_SUITE_NAME = "libero_suction"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preload completed stage-1 tasks from libero_object into libero_suction."
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=DEFAULT_SELECTION_JSON,
        help="Path to libero_suction_task_selection.json.",
    )
    parser.add_argument(
        "--data-regeneration-root",
        type=Path,
        default=DEFAULT_DATA_REGEN_ROOT,
        help="Root directory containing stage-1 collection outputs.",
    )
    parser.add_argument(
        "--source-suite-name",
        type=str,
        default=DEFAULT_SOURCE_SUITE_NAME,
        help="Existing suite whose completed stage-1 tasks should be reused.",
    )
    parser.add_argument(
        "--target-suite-name",
        type=str,
        default=DEFAULT_TARGET_SUITE_NAME,
        help="Target suite that should receive the reused tasks.",
    )
    return parser.parse_args()


def load_selection_records(selection_json: Path):
    if not selection_json.exists():
        raise FileNotFoundError(f"Selection JSON not found: {selection_json}")

    payload = json.loads(selection_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unsupported selection JSON format: {selection_json}")

    if not isinstance(records, list):
        raise ValueError(f"Expected a record list in selection JSON: {selection_json}")
    return records


def collect_reusable_tasks(records: list[dict], source_suite_name: str):
    reusable = []
    for record in records:
        if record.get("selection") != "yes":
            continue
        if record.get("suite_name") != source_suite_name:
            continue
        task_name = record.get("task_name")
        if not task_name:
            raise ValueError(f"Missing task_name in reusable record: {record}")
        reusable.append(str(task_name))
    return sorted(set(reusable))


def load_metainfo(json_path: Path):
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_metainfo(json_path: Path, payload: dict):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()

    records = load_selection_records(args.selection_json)
    reusable_tasks = collect_reusable_tasks(records, args.source_suite_name)

    source_suite_dir = args.data_regeneration_root / args.source_suite_name
    target_suite_dir = args.data_regeneration_root / args.target_suite_name
    source_metainfo_path = source_suite_dir / f"{args.source_suite_name}_raw_collection_metainfo.json"
    target_metainfo_path = target_suite_dir / f"{args.target_suite_name}_raw_collection_metainfo.json"

    if not source_suite_dir.exists():
        raise FileNotFoundError(f"Source suite directory not found: {source_suite_dir}")
    if not source_metainfo_path.exists():
        raise FileNotFoundError(f"Source metainfo JSON not found: {source_metainfo_path}")

    source_metainfo = load_metainfo(source_metainfo_path)
    target_metainfo = load_metainfo(target_metainfo_path)

    target_suite_dir.mkdir(parents=True, exist_ok=True)

    copied_hdf5 = []
    skipped_existing_hdf5 = []
    copied_metainfo = []
    skipped_existing_metainfo = []

    for task_name in reusable_tasks:
        source_hdf5 = source_suite_dir / f"{task_name}_demo.hdf5"
        target_hdf5 = target_suite_dir / f"{task_name}_demo.hdf5"

        if not source_hdf5.exists():
            raise FileNotFoundError(f"Missing source stage-1 HDF5: {source_hdf5}")
        if task_name not in source_metainfo:
            raise KeyError(f"Task {task_name} not found in source metainfo JSON: {source_metainfo_path}")

        if target_hdf5.exists():
            skipped_existing_hdf5.append(target_hdf5)
        else:
            shutil.copy2(source_hdf5, target_hdf5)
            copied_hdf5.append(target_hdf5)

        if task_name in target_metainfo:
            skipped_existing_metainfo.append(task_name)
        else:
            target_metainfo[task_name] = source_metainfo[task_name]
            copied_metainfo.append(task_name)

    save_metainfo(target_metainfo_path, target_metainfo)

    print(f"Selection JSON: {args.selection_json}")
    print(f"Source suite dir: {source_suite_dir}")
    print(f"Target suite dir: {target_suite_dir}")
    print(f"Source metainfo: {source_metainfo_path}")
    print(f"Target metainfo: {target_metainfo_path}")
    print(f"Reusable tasks from {args.source_suite_name}: {len(reusable_tasks)}")
    print(f"Copied HDF5 files: {len(copied_hdf5)}")
    print(f"Skipped existing HDF5 files: {len(skipped_existing_hdf5)}")
    print(f"Copied metainfo task entries: {len(copied_metainfo)}")
    print(f"Skipped existing metainfo task entries: {len(skipped_existing_metainfo)}")


if __name__ == "__main__":
    main()
