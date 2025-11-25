#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量把指定目录下所有 csv / txt 文件中的旧数据集路径，替换为 *_v1 路径。

内置两条规则:
  1) "data/Surge_Frames/PolypDiag/"          -> "data/Surge_Frames/PolypDiag_v1/"
  2) "data/Surge_Frames/SurgicalActions160/" -> "data/Surge_Frames/SurgicalActions160_v1/"

你也可以通过命令行参数自由指定自定义规则。

默认作用目录:
    data/Surge_Frames

用法示例（在项目根目录运行）:
    # 使用默认两条规则，在 data/Surge_Frames 下递归替换
    python data_process/polypdiag_fix_paths.py

只查看将要替换的数量（不真正改文件）:
    python data_process/polypdiag_fix_paths.py --dry-run

只作用于 SurgicalActions160_v1 目录:
    python data_process/polypdiag_fix_paths.py --root data/Surge_Frames/SurgicalActions160_v1

自定义规则（可以多次指定 --rule old,new）:
    python data_process/polypdiag_fix_paths.py \\
        --rule data/Surge_Frames/PolypDiag/,data/Surge_Frames/PolypDiag_v1/ \\
        --rule data/Surge_Frames/SurgicalActions160/,data/Surge_Frames/SurgicalActions160_v1/
"""

import argparse
from pathlib import Path
from typing import List, Tuple


def replace_in_file(
    file_path: Path,
    rules: List[Tuple[str, str]],
    dry_run: bool = False,
    backup: bool = True,
) -> int:
    """
    在单个文件中按照给定规则做字符串替换，返回总替换次数。
    rules: [(old, new), ...]
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"⚠️ 跳过（编码无法识别）: {file_path}")
        return 0

    total_count = 0
    new_text = text
    for old, new in rules:
        c = new_text.count(old)
        if c > 0:
            total_count += c
            new_text = new_text.replace(old, new)

    if total_count == 0:
        return 0

    if dry_run:
        print(f"[DRY-RUN] {file_path} 将替换 {total_count} 处")
        return total_count

    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        backup_path.write_text(text, encoding="utf-8")

    file_path.write_text(new_text, encoding="utf-8")
    print(f"✅ {file_path} 已替换 {total_count} 处")
    return total_count


def main():
    parser = argparse.ArgumentParser(
        description="批量将 csv/txt 中的旧数据集路径替换为 *_v1 路径（支持多规则）"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/Surge_Frames",
        help="要递归搜索的根目录（默认: data/Surge_Frames）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将要替换的数量，不真正修改文件",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不生成 .bak 备份文件（默认会为每个修改的文件生成一个备份）",
    )
    parser.add_argument(
        "--rule",
        action="append",
        metavar="OLD,NEW",
        help=(
            "自定义替换规则，格式为 OLD,NEW，可重复多次。\n"
            "例如: --rule data/Surge_Frames/PolypDiag/,data/Surge_Frames/PolypDiag_v1/"
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"❌ 根目录不存在: {root.resolve()}")
        return

    # 组装替换规则
    rules: List[Tuple[str, str]] = []

    # 默认规则
    default_rules = [
        ("data/Surge_Frames/PolypDiag/", "data/Surge_Frames/PolypDiag_v1/"),
        ("data/Surge_Frames/SurgicalActions160/", "data/Surge_Frames/SurgicalActions160_v1/"),
    ]
    rules.extend(default_rules)

    # 解析用户自定义规则
    if args.rule:
        for r in args.rule:
            if "," not in r:
                print(f"⚠️ 忽略非法规则（缺少逗号）: {r}")
                continue
            old, new = r.split(",", 1)
            old = old.strip()
            new = new.strip()
            if not old or not new:
                print(f"⚠️ 忽略非法规则（old/new 为空）: {r}")
                continue
            rules.append((old, new))

    print("🧩 使用的替换规则:")
    for old, new in rules:
        print(f"   '{old}' -> '{new}'")

    exts = {".csv", ".txt"}
    total_files = 0
    total_replaced = 0

    print(f"🔍 扫描目录: {root.resolve()}")
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in exts:
            continue

        total_files += 1
        n = replace_in_file(
            file_path,
            rules=rules,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        total_replaced += n

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(
        f"\n📊 模式: {mode}\n"
        f"   扫描文件数: {total_files}\n"
        f"   总替换次数: {total_replaced}"
    )


if __name__ == "__main__":
    main()


