from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, Tuple


DEFAULT_SRC = Path(r"D:\AlbumIndexedMusic")
DEFAULT_DEST = Path(__file__).resolve().parent / "music"

# 常见音乐文件后缀（小写）
AUDIO_EXTS = {
    ".mp3",
    ".flac",
    ".wav",
    ".m4a",
    ".ogg",
    ".oga",
    ".opus",
    ".aac",
    ".wma",
}


def _is_audio(path: Path, exts: Iterable[str] = AUDIO_EXTS) -> bool:
    return path.is_file() and path.suffix.lower() in exts


def _file_hash(path: Path, algo: str = "sha1", chunk_size: int = 1024 * 1024) -> str:
    """计算文件哈希（默认 SHA-1），用于去重和命名后缀。"""
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _propose_name(src: Path, dest_dir: Path) -> Path:
    """
    初始扁平化命名策略：`<上一级文件夹>__<文件名><后缀>`。
    例如：AlbumA/01.mp3 → music/AlbumA__01.mp3
    """
    parent = src.parent.name or "root"
    name = f"{parent}__{src.stem}{src.suffix}"
    return dest_dir / name


def _resolve_collision(src: Path, dest: Path) -> Tuple[Path, bool]:
    """
    解决命名冲突：
    - 若已存在且内容相同（尺寸和哈希一致），返回现有路径并标记为重复（跳过）。
    - 若内容不同，使用源文件哈希的前 8 位作为后缀重命名：
      e.g. AlbumA__01.mp3 → AlbumA__01.1a2b3c4d.mp3
    返回：(最终目标路径, 是否重复文件)。
    """
    if not dest.exists():
        return dest, False

    # 快速判断：尺寸相同再比哈希
    same_size = dest.stat().st_size == src.stat().st_size
    if same_size:
        try:
            if _file_hash(dest) == _file_hash(src):
                return dest, True  # 内容相同，视为重复，跳过
        except Exception:
            pass

    # 内容不同，基于哈希后缀重命名
    short = _file_hash(src)[:8]
    renamed = dest.with_name(f"{dest.stem}.{short}{dest.suffix}")
    if renamed.exists():
        # 极端情况：再加数字后缀
        idx = 2
        while True:
            candidate = dest.with_name(f"{dest.stem}.{short}.{idx}{dest.suffix}")
            if not candidate.exists():
                renamed = candidate
                break
            idx += 1
    return renamed, False


def flatten_music(
    src_root: Path = DEFAULT_SRC,
    dest_dir: Path = DEFAULT_DEST,
    action: str = "copy",
) -> dict:
    """
    扁平化复制/移动 `src_root` 下所有子文件夹中的音乐文件到 `dest_dir`。

    参数:
    - src_root: 源根目录（遍历其所有子文件夹）
    - dest_dir: 目标目录（项目下 `music`）
    - action: 操作类型，`copy` 或 `move`

    返回统计信息字典：{"total": int, "copied": int, "moved": int, "skipped": int, "renamed": int}
    """
    src_root = Path(src_root)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"源目录不存在：{src_root}")

    total = copied = moved = skipped = renamed = 0

    for p in src_root.rglob("*"):
        if not _is_audio(p):
            continue
        total += 1

        target = _propose_name(p, dest_dir)
        final_target, is_dup = _resolve_collision(p, target)

        if is_dup:
            skipped += 1
            continue

        if final_target != target:
            renamed += 1

        if action == "move":
            shutil.move(str(p), str(final_target))
            moved += 1
        else:
            shutil.copy2(str(p), str(final_target))
            copied += 1

    return {
        "total": total,
        "copied": copied,
        "moved": moved,
        "skipped": skipped,
        "renamed": renamed,
        "dest": str(dest_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 D:\\AlbumIndexedMusic 下的音乐文件扁平化到项目下 music 目录"
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help="源根目录（默认 D:\\AlbumIndexedMusic）",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="目标目录（默认项目根下的 music）",
    )
    parser.add_argument(
        "--action",
        choices=["copy", "move"],
        default="copy",
        help="执行复制或移动（默认 copy）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats = flatten_music(args.src, args.dest, args.action)
    print(
        "完成：total={total}, copied={copied}, moved={moved}, skipped={skipped}, renamed={renamed}, dest={dest}".format(
            **stats
        )
    )