import argparse
from pathlib import Path
from typing import Optional


def rename_dirs(root: Path, ext: Optional[str] = None, dry_run: bool = True) -> int:
    """
    批量把带后缀的文件夹重命名为不带后缀的。

    - root: 根目录，里面包含需要处理的子文件夹
    - ext: 指定只处理某个后缀（例如 ".mp3"），不指定则处理所有带后缀的文件夹
    - dry_run: 仅预览不实际改名

    返回成功重命名的数量。
    """
    if not root.exists() or not root.is_dir():
        print(f"目录不存在或不是文件夹: {root}")
        return 0

    renamed = 0
    for entry in root.iterdir():
        if not entry.is_dir():
            continue

        name = entry.name
        suffix = Path(name).suffix  # 仅移除最后一个扩展名，如 .mp3
        if not suffix:
            continue

        if ext and suffix.lower() != ext.lower():
            continue

        target_name = name[: -len(suffix)] if name.endswith(suffix) else name
        if target_name == name:
            continue

        target_path = entry.with_name(target_name)

        if target_path.exists():
            print(f"跳过，目标已存在: {entry} -> {target_path}")
            continue

        if dry_run:
            print(f"预览：{entry} -> {target_path}")
        else:
            entry.rename(target_path)
            print(f"已重命名：{entry} -> {target_path}")
            renamed += 1

    if dry_run:
        print("完成预览（未做实际修改）。")
    else:
        print(f"完成，成功重命名 {renamed} 个文件夹。")
    return renamed


def main():
    parser = argparse.ArgumentParser(
        description="批量把带后缀的文件夹重命名为不带后缀的"
    )
    parser.add_argument(
        "root",
        help="根目录路径（包含要处理的子文件夹）",
    )
    parser.add_argument(
        "--ext",
        default=None,
        help="只处理指定后缀，例如 .mp3；默认处理所有带后缀的文件夹",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="执行实际重命名（默认仅预览）",
    )

    args = parser.parse_args()
    root = Path(args.root)
    rename_dirs(root=root, ext=args.ext, dry_run=not args.apply)


if __name__ == "__main__":
    main()
