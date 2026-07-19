"""
merge_libero_suction_worker_outputs.py

【脚本作用】
这个脚本用于把 3 个 worker 各自采集完成的中间结果，复制合并回正式目录
`/data1/ps/openvla/data_regeneration/libero_suction`。

注意：这里执行的是“复制合并”，不是“剪切合并”。
也就是说，worker_a / worker_b / worker_c 目录中的原始 `.hdf5` 和 metainfo JSON
都会被保留下来，不会因为运行本脚本而被删除。

【输入是什么】
1. 3 个 worker 的输出目录：
   - /data1/ps/openvla/data_regeneration/libero_suction_worker_a
   - /data1/ps/openvla/data_regeneration/libero_suction_worker_b
   - /data1/ps/openvla/data_regeneration/libero_suction_worker_c
2. 3 个 worker 各自的 metainfo JSON
3. 正式目标目录：/data1/ps/openvla/data_regeneration/libero_suction
4. 官方原始目录：/data1/ps/openvla/LIBERO/libero/datasets/libero_suction
   这个目录只用于校验每个 task 的 demo 数量是否完整

【输出是什么】
1. 把 3 个 worker 的 task 级 HDF5 复制到正式目录
2. 把 3 个 worker 的 metainfo 条目合并到正式 metainfo JSON
3. 生成一份详细的合并报告 JSON，记录：
   - 每个 task 的源路径和目标路径
   - 源/目标文件大小
   - 源/目标 SHA256
   - 即时复制校验结果
   - 合并后总体校验结果

【直接运行命令】
python /data1/ps/openvla/experiments/robot/libero/data_regeneration/multi_worker/merge_libero_suction_worker_outputs.py
"""

from __future__ import annotations
# 开启更灵活的类型注解语法，方便下面直接写 list[Path] 这类类型提示。

import argparse
# 用于解析命令行参数。
import hashlib
# 用于计算文件 SHA256，确认复制后的目标文件与源文件完全一致。
import json
# 用于读写 metainfo JSON 和合并报告 JSON。
import shutil
# 用于把 worker 的 HDF5 复制到正式目录。
import traceback
# 用于在失败时把完整异常堆栈写进报告，方便事后排查。
from pathlib import Path
# 用于处理路径，避免手写字符串路径时出错。

import h5py
# 用于读取 HDF5，统计 demo 数量，校验 task 文件是否完整。


# ------------------------------------------------------------------------------
# 路径与默认常量
# ------------------------------------------------------------------------------
# ROOT_DIR 指向 openvla 仓库根目录。
ROOT_DIR = Path(__file__).resolve().parents[5]

# 第一阶段所有采集结果都放在 data_regeneration 下。
DATA_REGEN_ROOT = ROOT_DIR / "data_regeneration"

# 正式 libreo_suction 原始重采集目录。
DEFAULT_TARGET_DIR = DATA_REGEN_ROOT / "libero_suction"

# 官方原始数据目录，用于验证每个 task 是否采满了 demo。
DEFAULT_RAW_DATA_DIR = ROOT_DIR / "LIBERO" / "libero" / "datasets" / "libero_suction"

# 默认报告路径直接放在正式目录里，合并结束后一起查看最方便。
DEFAULT_REPORT_PATH = DEFAULT_TARGET_DIR / "libero_suction_merge_report.json"

# 这 3 个 worker 名字和并行采集脚本保持一致。
WORKER_NAMES = ("worker_a", "worker_b", "worker_c")

# 计算 SHA256 时分块读取，避免大文件一次性全部读入内存。
SHA256_CHUNK_SIZE = 1024 * 1024


def default_worker_output_dir(worker_name: str) -> Path:
    """根据 worker 名字计算它的默认输出目录。"""
    return DATA_REGEN_ROOT / f"libero_suction_{worker_name}"


def default_worker_metainfo_path(worker_name: str, worker_dir: Path) -> Path:
    """根据 worker 名字和目录，计算它默认的 metainfo JSON 路径。"""
    return worker_dir / f"libero_suction_{worker_name}_raw_collection_metainfo.json"


def load_json_dict(json_path: Path) -> dict:
    """
    读取 JSON 文件。

    如果文件不存在，就返回空字典，方便第一次合并时继续往下走。
    """
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_dict(json_path: Path, data: dict):
    """
    把字典写回 JSON。

    写报告和写正式 metainfo 都统一走这个函数，避免不同地方写法不一致。
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def count_demo_groups(hdf5_path: Path) -> int:
    """
    统计一个 task HDF5 里包含多少个 demo_x。

    这里只统计 data 组下面，名字以 demo_ 开头的 group 数量。
    """
    with h5py.File(hdf5_path, "r") as h5_file:
        if "data" not in h5_file:
            return 0
        return sum(1 for group_name in h5_file["data"].keys() if group_name.startswith("demo_"))


def compute_file_sha256(file_path: Path) -> str:
    """
    计算一个文件的 SHA256。

    SHA256 能更稳地保证“源文件”和“目标文件”在字节级完全一致。
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(SHA256_CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def build_file_fingerprint(file_path: Path) -> dict:
    """
    为一个文件生成“指纹”信息。

    这个指纹不仅包含 SHA256，也包含文件大小和 demo 数量，便于事后同时看轻量信息和强校验信息。
    """
    return {
        "path": str(file_path),
        "size_bytes": int(file_path.stat().st_size),
        "sha256": compute_file_sha256(file_path),
        "demo_count": int(count_demo_groups(file_path)),
    }


def validate_worker_task_file(worker_task_path: Path, raw_task_path: Path):
    """
    验证某个 worker 产出的 task 文件是否完整。

    校验标准：
    - 官方原始 task 有多少个 demo
    - worker 产出的 task 也必须有同样数量的 demo
    """
    if not raw_task_path.exists():
        raise FileNotFoundError(f"Cannot find raw libero_suction task file: {raw_task_path}")

    raw_demo_count = count_demo_groups(raw_task_path)
    worker_demo_count = count_demo_groups(worker_task_path)

    if worker_demo_count != raw_demo_count:
        raise ValueError(
            f"Worker task file is incomplete: {worker_task_path} "
            f"(worker demos={worker_demo_count}, raw demos={raw_demo_count})"
        )


def collect_worker_hdf5_files(worker_dir: Path) -> list[Path]:
    """
    收集某个 worker 目录下的所有 task HDF5。

    这里故意只收集 `.hdf5` 文件，不把 metainfo JSON 混进去。
    """
    return sorted(worker_dir.glob("*.hdf5"))


def collect_hdf5_files(target_dir: Path) -> list[Path]:
    """统一收集某个目录下的全部 `.hdf5` 文件。"""
    return sorted(target_dir.glob("*.hdf5"))


def verify_source_target_match(source_path: Path, target_path: Path) -> dict:
    """
    对源文件和目标文件做强一致性校验。

    这里只比较字节级内容，因此最终结论主要由 SHA256 是否一致决定。
    """
    source_fingerprint = build_file_fingerprint(source_path)
    target_fingerprint = build_file_fingerprint(target_path)

    return {
        "source": source_fingerprint,
        "target": target_fingerprint,
        "size_match": source_fingerprint["size_bytes"] == target_fingerprint["size_bytes"],
        "demo_count_match": source_fingerprint["demo_count"] == target_fingerprint["demo_count"],
        "sha256_match": source_fingerprint["sha256"] == target_fingerprint["sha256"],
    }


def append_report_failure(report: dict, message: str):
    """把失败信息附加到报告里，便于最后统一输出。"""
    report.setdefault("errors", []).append(message)


def merge_worker_directory(
    *,
    worker_name: str,
    worker_dir: Path,
    worker_metainfo_path: Path,
    raw_data_dir: Path,
    target_dir: Path,
    target_metainfo_dict: dict,
    report: dict,
) -> tuple[int, int]:
    """
    合并一个 worker 目录。

    返回两个数字：
    - 第一个是本 worker 实际复制了多少个 task HDF5
    - 第二个是本 worker 实际合并了多少个 metainfo task 条目
    """
    if not worker_dir.exists():
        raise FileNotFoundError(f"Worker output directory does not exist: {worker_dir}")

    worker_metainfo_dict = load_json_dict(worker_metainfo_path)
    worker_hdf5_files = collect_worker_hdf5_files(worker_dir)

    copied_hdf5_count = 0
    merged_metainfo_count = 0

    # 给当前 worker 单独开一个报告分区，后续所有校验细节都写在这里。
    worker_report = {
        "worker_name": worker_name,
        "worker_dir": str(worker_dir),
        "worker_metainfo_path": str(worker_metainfo_path),
        "expected_task_count": len(worker_hdf5_files),
        "copied_task_count": 0,
        "merged_metainfo_task_count": 0,
        "tasks": [],
    }
    report["workers"].append(worker_report)

    # 先处理 HDF5 文件。
    for worker_task_path in worker_hdf5_files:
        # 当前 task 文件名，例如 xxx_demo.hdf5。
        task_file_name = worker_task_path.name

        # 对应的官方原始 task 文件，用于做完整性校验。
        raw_task_path = raw_data_dir / task_file_name
        validate_worker_task_file(worker_task_path, raw_task_path)

        # 目标正式目录中的路径。
        target_task_path = target_dir / task_file_name

        # 如果正式目录里已经存在同名 task，就立即报错，避免误覆盖。
        if target_task_path.exists():
            raise FileExistsError(f"Target task file already exists: {target_task_path}")

        # 通过校验后，把当前 worker 的完整 task HDF5 复制进正式目录。
        shutil.copy2(worker_task_path, target_task_path)

        # 每复制一个文件后立即做源/目标一致性校验，这是“第一层保险”。
        verification = verify_source_target_match(worker_task_path, target_task_path)
        task_report = {
            "task_file_name": task_file_name,
            "raw_task_path": str(raw_task_path),
            "verification": verification,
            "copy_verification_passed": bool(verification["sha256_match"]),
        }
        worker_report["tasks"].append(task_report)

        if not verification["sha256_match"]:
            raise ValueError(
                f"Copied task file does not match source file: {worker_task_path} -> {target_task_path}"
            )

        copied_hdf5_count += 1

    # 再处理 metainfo 条目。
    for task_name, task_metainfo in worker_metainfo_dict.items():
        # 如果正式 metainfo 里已经有同名 task，也视为冲突。
        if task_name in target_metainfo_dict:
            raise ValueError(f"Target metainfo already contains task: {task_name}")

        target_metainfo_dict[task_name] = task_metainfo
        merged_metainfo_count += 1

    worker_report["copied_task_count"] = copied_hdf5_count
    worker_report["merged_metainfo_task_count"] = merged_metainfo_count

    return copied_hdf5_count, merged_metainfo_count


def run_post_merge_validation(
    *,
    target_dir: Path,
    target_metainfo_path: Path,
    raw_data_dir: Path,
    initial_target_hdf5_count: int,
    expected_added_task_count: int,
    worker_source_paths: list[Path],
    report: dict,
):
    """
    在全部 worker 合并完成后，再做一轮总体校验。

    这里做的是“第二层保险”：
    - 检查总数量是否对
    - 检查正式 metainfo 数量是否对
    - 检查每个目标文件仍然和源文件一致
    - 检查每个目标文件的 demo 数量是否与官方原始数据一致
    """
    target_hdf5_files = collect_hdf5_files(target_dir)
    target_hdf5_names = {path.name for path in target_hdf5_files}
    target_metainfo_dict = load_json_dict(target_metainfo_path)

    expected_final_hdf5_count = initial_target_hdf5_count + expected_added_task_count

    post_report = {
        "initial_target_hdf5_count": int(initial_target_hdf5_count),
        "expected_added_task_count": int(expected_added_task_count),
        "expected_final_hdf5_count": int(expected_final_hdf5_count),
        "actual_final_hdf5_count": int(len(target_hdf5_files)),
        "actual_final_metainfo_task_count": int(len(target_metainfo_dict)),
        "all_worker_tasks_present": True,
        "all_target_hashes_match_sources": True,
        "all_target_demo_counts_match_raw": True,
        "all_target_tasks_have_metainfo": True,
        "task_checks": [],
    }
    report["post_merge_validation"] = post_report

    if len(target_hdf5_files) != expected_final_hdf5_count:
        raise ValueError(
            f"Final target HDF5 count mismatch: got {len(target_hdf5_files)}, "
            f"expected {expected_final_hdf5_count}"
        )

    if len(target_metainfo_dict) != len(target_hdf5_files):
        raise ValueError(
            f"Final metainfo task count mismatch: got {len(target_metainfo_dict)}, "
            f"expected {len(target_hdf5_files)}"
        )

    for worker_source_path in worker_source_paths:
        target_task_path = target_dir / worker_source_path.name

        # 每个源文件在正式目录中都必须找到对应目标文件。
        if target_task_path.name not in target_hdf5_names:
            post_report["all_worker_tasks_present"] = False
            raise FileNotFoundError(f"Merged target task file is missing: {target_task_path}")

        # 每个目标文件都要再次和源文件做 SHA256 校验，防止出现“复制时正确，但后续又被改动”的情况。
        verification = verify_source_target_match(worker_source_path, target_task_path)
        if not verification["sha256_match"]:
            post_report["all_target_hashes_match_sources"] = False
            raise ValueError(f"Post-merge SHA256 mismatch: {worker_source_path} vs {target_task_path}")

        # 每个正式目录里的 task 还要再次核对 demo 数量与官方原始数据一致。
        raw_task_path = raw_data_dir / worker_source_path.name
        raw_demo_count = count_demo_groups(raw_task_path)
        target_demo_count = count_demo_groups(target_task_path)
        demo_count_match = target_demo_count == raw_demo_count
        if not demo_count_match:
            post_report["all_target_demo_counts_match_raw"] = False
            raise ValueError(
                f"Post-merge demo count mismatch: {target_task_path} "
                f"(target demos={target_demo_count}, raw demos={raw_demo_count})"
            )

        # HDF5 文件名形如 xxx_demo.hdf5，需要把后缀和 _demo 去掉，才能对应 metainfo 里的 task 名。
        task_name = worker_source_path.name.removesuffix(".hdf5").removesuffix("_demo")
        has_metainfo = task_name in target_metainfo_dict
        if not has_metainfo:
            post_report["all_target_tasks_have_metainfo"] = False
            raise ValueError(f"Post-merge target metainfo is missing task: {task_name}")

        post_report["task_checks"].append(
            {
                "task_file_name": worker_source_path.name,
                "task_name": task_name,
                "verification": verification,
                "raw_demo_count": int(raw_demo_count),
                "target_demo_count": int(target_demo_count),
                "has_metainfo": bool(has_metainfo),
            }
        )


def main(args):
    """
    主流程：
    1. 读取正式目标目录与正式 metainfo
    2. 依次处理 worker_a / worker_b / worker_c
    3. 复制每个 task HDF5 后立即做 SHA256 一致性校验
    4. 合并完成后，再做一轮总体验证
    5. 写回正式 metainfo 与详细报告
    """
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_data_dir = Path(args.raw_data_dir)

    # 正式总 metainfo 文件默认就放在正式目录里。
    target_metainfo_path = (
        Path(args.target_metainfo_path)
        if args.target_metainfo_path
        else target_dir / "libero_suction_raw_collection_metainfo.json"
    )

    # 报告文件默认也放在正式目录里，方便合并后和最终数据一起查看。
    report_path = Path(args.report_path) if args.report_path else DEFAULT_REPORT_PATH

    # 在真正合并之前，先把正式目录当前已有状态记录下来，方便后面做“合并后总量核对”。
    initial_target_hdf5_files = collect_hdf5_files(target_dir)
    initial_target_hdf5_count = len(initial_target_hdf5_files)

    target_metainfo_dict = load_json_dict(target_metainfo_path)

    # 这个 report 会贯穿整个合并过程。
    report = {
        "script": "merge_libero_suction_worker_outputs.py",
        "target_dir": str(target_dir),
        "target_metainfo_path": str(target_metainfo_path),
        "raw_data_dir": str(raw_data_dir),
        "report_path": str(report_path),
        "copy_mode": "copy2",
        "initial_target_hdf5_count": int(initial_target_hdf5_count),
        "initial_target_metainfo_task_count": int(len(target_metainfo_dict)),
        "workers": [],
        "post_merge_validation": None,
        "final_status": "running",
        "errors": [],
    }

    total_copied_hdf5 = 0
    total_merged_metainfo = 0
    worker_source_paths: list[Path] = []

    try:
        # 依次合并三个 worker 的结果。
        for worker_name in WORKER_NAMES:
            worker_dir = default_worker_output_dir(worker_name)
            worker_metainfo_path = default_worker_metainfo_path(worker_name, worker_dir)
            worker_hdf5_files = collect_worker_hdf5_files(worker_dir)
            worker_source_paths.extend(worker_hdf5_files)

            print(f"开始合并 {worker_name}")
            print(f"  worker 输出目录：{worker_dir}")
            print(f"  worker metainfo：{worker_metainfo_path}")

            copied_hdf5_count, merged_metainfo_count = merge_worker_directory(
                worker_name=worker_name,
                worker_dir=worker_dir,
                worker_metainfo_path=worker_metainfo_path,
                raw_data_dir=raw_data_dir,
                target_dir=target_dir,
                target_metainfo_dict=target_metainfo_dict,
                report=report,
            )

            print(f"  已复制 task HDF5 数量：{copied_hdf5_count}")
            print(f"  已合并 metainfo task 数量：{merged_metainfo_count}")

            total_copied_hdf5 += copied_hdf5_count
            total_merged_metainfo += merged_metainfo_count

        # 三个 worker 全部合并完成后，先把正式 metainfo 写回磁盘，再做总体校验。
        # 这样总体校验可以基于磁盘上的最终状态工作，而不是只基于内存中的字典。
        save_json_dict(target_metainfo_path, target_metainfo_dict)

        run_post_merge_validation(
            target_dir=target_dir,
            target_metainfo_path=target_metainfo_path,
            raw_data_dir=raw_data_dir,
            initial_target_hdf5_count=initial_target_hdf5_count,
            expected_added_task_count=len(worker_source_paths),
            worker_source_paths=worker_source_paths,
            report=report,
        )

        report["total_copied_hdf5"] = int(total_copied_hdf5)
        report["total_merged_metainfo"] = int(total_merged_metainfo)
        report["final_status"] = "passed"
    except Exception as exc:
        # 一旦失败，立即把错误写进报告。
        # 这里不做回滚，也不删除已经复制的文件，保持“失败时保留现场”的策略。
        report["final_status"] = "failed"
        append_report_failure(report, str(exc))
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        # 无论成功还是失败，都把当前报告写到磁盘，方便事后查看。
        save_json_dict(report_path, report)

    print("=" * 80)
    print("libero_suction worker 结果合并完成")
    print(f"正式目标目录：{target_dir}")
    print(f"正式 metainfo：{target_metainfo_path}")
    print(f"合并报告：{report_path}")
    print(f"本次总共复制 task HDF5 数量：{total_copied_hdf5}")
    print(f"本次总共合并 metainfo task 数量：{total_merged_metainfo}")
    print("=" * 80)


if __name__ == "__main__":
    # 定义命令行参数。
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        type=str,
        default=str(DEFAULT_TARGET_DIR),
        help="正式 libero_suction 原始采集目录。",
    )
    parser.add_argument(
        "--target_metainfo_path",
        type=str,
        default=None,
        help="正式目标 metainfo JSON 路径。不传时默认写到 target_dir/libero_suction_raw_collection_metainfo.json。",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=str(DEFAULT_RAW_DATA_DIR),
        help="官方原始 libero_suction 数据目录，用于校验 worker 结果是否采满。",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help="合并后的详细检查报告 JSON 路径。",
    )

    # 解析参数并进入主流程。
    main(parser.parse_args())
