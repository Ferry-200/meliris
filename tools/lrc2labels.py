from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple


# 与 Rust 版本保持一致的时间戳解析：例如 [mm:ss.xx] 或 [mm:ss.xxx]
TIMESTAMP_RE = re.compile(r"\[(\d{2}):(\d{2}\.\d{2,3})\]")

# 过滤元数据/版权提示的关键字
METADATA_PATTERNS = (
    "：",  # 中文冒号（如 歌手：xxx）
    " : ",
    " - ",
)
COPYRIGHT_PATTERNS = (
    "QQ音乐享有",
    "TME享有",
    "文曲大模型",
)

# 常见音频扩展名
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


def _read_embedded_lyrics(path: Path) -> Optional[str]:
    """尝试从音频标签中读取内嵌歌词（LRC/USLT/VorbisComment）。

    优先返回文本字符串；失败时返回 None。
    """
    try:
        from mutagen import File as MFile
        from mutagen.id3 import USLT, ID3
        from mutagen.flac import FLAC
        from mutagen.oggvorbis import OggVorbis
    except Exception:
        print(f"_mutagen 导入失败：{path}")
        return None

    audio = MFile(str(path))
    if audio is None:
        return None

    # MP3 / ID3
    if isinstance(audio, ID3) or hasattr(audio, "tags"):
        try:
            id3 = audio if isinstance(audio, ID3) else getattr(audio, "tags", None)
            if id3:
                # USLT (Unsynchronized lyrics/text transcription)
                for frame in id3.getall("USLT"):
                    if isinstance(frame, USLT) and frame.text:
                        return frame.text
                # 常见自定义文本帧（不一定存在）
                for frame in id3.getall("TXXX"):
                    if hasattr(frame, "text") and frame.text:
                        # 可能包含 LRC 字符串
                        return "\n".join(frame.text) if isinstance(frame.text, list) else str(frame.text)
        except Exception:
            pass

    # FLAC / VorbisComment
    try:
        if isinstance(audio, FLAC):
            vc = audio.tags
            if vc:
                for key in ("LYRICS", "UNSYNCEDLYRICS", "USLT", "LRC"):
                    if key in vc and vc[key]:
                        val = vc[key]
                        return "\n".join(val) if isinstance(val, list) else str(val)
    except Exception:
        pass

    # OGG Vorbis
    try:
        if isinstance(audio, OggVorbis):
            vc = audio.tags
            if vc:
                for key in ("LYRICS", "UNSYNCEDLYRICS", "USLT", "LRC"):
                    if key in vc and vc[key]:
                        val = vc[key]
                        return "\n".join(val) if isinstance(val, list) else str(val)
    except Exception:
        pass

    return None


def _read_sidecar_lrc(path: Path) -> Optional[str]:
    """读取同名 .lrc 辅助文件（若存在）。"""
    lrc = path.with_suffix(".lrc")
    if lrc.exists():
        try:
            return lrc.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            try:
                return lrc.read_text(encoding="gbk", errors="ignore")
            except Exception:
                return None
    return None


def _parse_lrc_to_timestamps(lyric_text: str) -> Tuple[list, bool]:
    """按 Rust 逻辑解析 LRC，返回 (timestamps, has_end_label)。

    timestamps: 列表中每项为 {"start": (m:int, s:float), "end": Optional[(m:int, s:float)], "content": str}
    has_end_label: 是否使用过“空行作为上一句结束”的标记（决定输出目录是否进入 perfect 子目录）。
    """
    timestamps = []
    has_end_label = False
    for raw_line in lyric_text.splitlines():
        m = TIMESTAMP_RE.search(raw_line)
        if not m:
            continue
        try:
            minutes = int(m.group(1))
            seconds = float(m.group(2))
        except Exception:
            continue

        # 去掉时间戳后的内容
        content = TIMESTAMP_RE.sub("", raw_line)
        content = str(content)

        if not content.strip():
            # 空白行标记上一句的结束时间
            has_end_label = True
            if timestamps:
                timestamps[-1]["end"] = (minutes, seconds)
            continue

        # 过滤元数据与版权提示行
        if any(p in content for p in METADATA_PATTERNS):
            continue
        if any(p in content for p in COPYRIGHT_PATTERNS):
            continue

        timestamps.append(
            {
                "start": (minutes, seconds),
                "end": None,
                "content": content,
            }
        )

    return timestamps, has_end_label


def _timestamps_to_labels(timestamps: list) -> list:
    """根据 Rust 逻辑将时间戳序列转换为 labels（带推断的 end）。"""
    labels = []
    n = len(timestamps)
    for i in range(n):
        if i != n - 1:
            # 去除时间完全重复的行
            if (
                timestamps[i]["start"][0] == timestamps[i + 1]["start"][0]
                and timestamps[i]["start"][1] == timestamps[i + 1]["start"][1]
            ):
                continue

            if timestamps[i]["end"] is not None:
                labels.append(timestamps[i])
            else:
                labels.append(
                    {
                        "start": timestamps[i]["start"],
                        "end": timestamps[i + 1]["start"],
                        "content": timestamps[i]["content"],
                    }
                )
        else:
            if timestamps[i]["end"] is not None:
                labels.append(timestamps[i])
    return labels


def trans_lrc_to_labels(src_path: Path) -> dict:
    """递归扫描 `src_path` 下音频文件，读取歌词（内嵌或同名 .lrc），
    解析并生成 labels JSON，输出到 ./labels 或 ./labels/perfect。

    返回统计：{"total": int, "written": int, "skipped": int}
    """
    total = 0
    written = 0
    skipped = 0

    labels_root = Path(".") / "labels"
    (labels_root).mkdir(parents=True, exist_ok=True)
    (labels_root / "perfect").mkdir(parents=True, exist_ok=True)

    for p in Path(src_path).rglob("*"):
        if not (p.is_file() and p.suffix.lower() in AUDIO_EXTS):
            continue
        total += 1

        # 优先内嵌歌词，其次同名 .lrc
        lyric = _read_embedded_lyrics(p)
        if not lyric:
            lyric = _read_sidecar_lrc(p)
        if not lyric:
            skipped += 1
            continue

        timestamps, has_end_label = _parse_lrc_to_timestamps(lyric)
        labels = _timestamps_to_labels(timestamps)

        # 输出路径：./labels[/perfect]/<文件名去扩展>.json
        out_dir = labels_root / ("perfect" if has_end_label else "")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (p.stem + ".json")
        out_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
        written += 1

    return {"total": total, "written": written, "skipped": skipped}


def main():
    # 默认迁移 Rust main.rs 的行为：处理 D:\meliris\music
    default_src = Path(r"D:\meliris\music")
    stats = trans_lrc_to_labels(default_src)
    print("labels generated:", stats)


if __name__ == "__main__":
    main()
