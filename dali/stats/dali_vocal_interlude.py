from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Segment:
    start: float
    end: float


def _to_seconds(mm_ss: object) -> float:
    """Convert [min, sec] or (min, sec) to seconds."""
    if not (isinstance(mm_ss, (list, tuple)) and len(mm_ss) == 2):
        raise ValueError(f"Invalid time tuple: {mm_ss!r}")
    m = float(mm_ss[0])
    s = float(mm_ss[1])
    return m * 60.0 + s


def _load_segments(label_path: Path) -> list[Segment]:
    data = json.loads(label_path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list")

    segs: list[Segment] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item:
            continue
        try:
            start = _to_seconds(item["start"])
            end = _to_seconds(item["end"])
        except Exception:
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        if end <= start:
            continue
        segs.append(Segment(start=start, end=end))

    segs.sort(key=lambda s: (s.start, s.end))
    return segs


def _merge_union(segs: Iterable[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for s in segs:
        if not merged:
            merged.append(s)
            continue
        last = merged[-1]
        if s.start <= last.end:
            merged[-1] = Segment(start=last.start, end=max(last.end, s.end))
        else:
            merged.append(s)
    return merged


def _union_length(segs: Iterable[Segment]) -> float:
    merged = _merge_union(segs)
    return float(sum(s.end - s.start for s in merged))


def _bucket_length_seconds(length_sec: float, *, mode: str) -> int:
    """Bucket a duration into integer seconds.

    mode:
      - ceil: (0,1] -> 1, (1,2] -> 2, ...; 0 -> 0
      - floor: [0,1) -> 0, [1,2) -> 1, ...
      - round: round to nearest int
    """
    if not math.isfinite(length_sec):
        return 0
    if length_sec < 0:
        return 0
    if mode == "ceil":
        return int(math.ceil(length_sec))
    if mode == "floor":
        return int(math.floor(length_sec))
    if mode == "round":
        return int(round(length_sec))
    raise ValueError(f"Unknown binning mode: {mode}")


def _histogram(lengths: Iterable[float], *, mode: str) -> dict[int, int]:
    hist: dict[int, int] = {}
    for x in lengths:
        b = _bucket_length_seconds(float(x), mode=mode)
        hist[b] = hist.get(b, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: kv[0]))


def _interlude_segments(
    vocal_segs: list[Segment],
    *,
    mode: str,
    span_start: float,
    span_end: float,
) -> list[Segment]:
    """Derive interlude segments from vocal segments.

    mode:
      - between_vocals: only gaps between consecutive vocal segments
      - span: gaps covering the whole [span_start, span_end], i.e. includes leading/trailing gaps
    """
    if span_end <= span_start:
        return []
    if not vocal_segs:
        return [Segment(start=span_start, end=span_end)] if mode == "span" else []

    merged = _merge_union(
        [
            Segment(start=max(s.start, span_start), end=min(s.end, span_end))
            for s in vocal_segs
            if min(s.end, span_end) > max(s.start, span_start)
        ]
    )
    if not merged:
        return [Segment(start=span_start, end=span_end)] if mode == "span" else []

    gaps: list[Segment] = []

    if mode == "span":
        # leading gap
        if merged[0].start > span_start:
            gaps.append(Segment(start=span_start, end=merged[0].start))

    # internal gaps
    for a, b in zip(merged, merged[1:]):
        if b.start > a.end:
            gaps.append(Segment(start=a.end, end=b.start))

    if mode == "span":
        # trailing gap
        if merged[-1].end < span_end:
            gaps.append(Segment(start=merged[-1].end, end=span_end))

    return gaps


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "统计 labels_dali 的标注“长度分布”：\n"
            "- 人声标注长度：每条歌词区间(end-start) 的时长，按 1 秒 bucket 统计(例如 1s/5s 有多少条)。\n"
            "- 间奏标注长度：由相邻人声区间之间的空隙(gap) 生成“间奏段”，同样按 1 秒 bucket 统计。\n"
            "说明：labels_dali JSON 本身只包含人声(歌词)区间，间奏来自人声区间之间的空白。"
        )
    )
    p.add_argument(
        "--labels-root",
        type=str,
        default=str(Path(".") / "labels_dali"),
        help="labels_dali 文件夹路径",
    )
    p.add_argument(
        "--span",
        choices=["zero_to_last", "first_to_last"],
        default="first_to_last",
        help=(
            "用于生成间奏段的统计跨度："
            "zero_to_last 表示从 0 秒到最后一句结束；"
            "first_to_last 表示从第一句开始到最后一句结束(默认)。"
        ),
    )
    p.add_argument(
        "--interlude-mode",
        choices=["between_vocals", "span"],
        default="span",
        help=(
            "间奏段生成方式：between_vocals 只统计人声之间的空隙；"
            "span 还会把开头/结尾空白也计为间奏。(default)"
        ),
    )
    p.add_argument(
        "--binning",
        choices=["ceil", "floor", "round"],
        default="floor",
        help="长度分桶方式：ceil / floor(default) / round",
    )
    p.add_argument(
        "--export-dir",
        type=str,
        default="",
        help=(
            "可选：导出目录。会写出每首歌的 per-second CSV，"
            "以及 overall_stats.json。留空则只打印汇总。"
        ),
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="仅处理前 N 个文件(调试用)，0 表示不限制",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    labels_root = Path(args.labels_root)
    if not labels_root.exists():
        raise SystemExit(f"labels_root not found: {labels_root}")

    export_dir = Path(args.export_dir) if args.export_dir else None
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_root.glob("*.json"))
    if args.max_files and args.max_files > 0:
        label_files = label_files[: args.max_files]

    overall = {
        "labels_root": str(labels_root),
        "span": args.span,
        "interlude_mode": args.interlude_mode,
        "binning": args.binning,
        "files": 0,
        "files_skipped": 0,
        "vocal_segments": 0,
        "interlude_segments": 0,
        "vocal_hist": {},
        "interlude_hist": {},
    }

    per_file_rows: list[dict] = []

    for pth in label_files:
        try:
            segs = _load_segments(pth)
        except Exception:
            overall["files_skipped"] += 1
            continue

        if not segs:
            overall["files_skipped"] += 1
            continue

        first_start = segs[0].start
        last_end = max(s.end for s in segs)

        span_start = 0.0 if args.span == "zero_to_last" else float(first_start)
        span_end = float(last_end)

        # Vocal label lengths: use original segments (each lyric line) clipped to span.
        vocal_label_segs = [
            Segment(start=max(s.start, span_start), end=min(s.end, span_end))
            for s in segs
            if min(s.end, span_end) > max(s.start, span_start)
        ]
        vocal_lengths = [s.end - s.start for s in vocal_label_segs]
        vocal_hist = _histogram(vocal_lengths, mode=args.binning)

        # Interlude segments: gaps between merged vocal segments.
        interlude_segs = _interlude_segments(
            segs,
            mode=args.interlude_mode,
            span_start=span_start,
            span_end=span_end,
        )
        interlude_lengths = [s.end - s.start for s in interlude_segs]
        interlude_hist = _histogram(interlude_lengths, mode=args.binning)

        overall["files"] += 1
        overall["vocal_segments"] += len(vocal_label_segs)
        overall["interlude_segments"] += len(interlude_segs)
        # merge hist into overall
        for k, v in vocal_hist.items():
            overall["vocal_hist"][str(k)] = int(overall["vocal_hist"].get(str(k), 0)) + int(v)
        for k, v in interlude_hist.items():
            overall["interlude_hist"][str(k)] = int(overall["interlude_hist"].get(str(k), 0)) + int(v)

        row = {
            "id": pth.stem,
            "span_start_sec": span_start,
            "span_end_sec": span_end,
            "vocal_segments": len(vocal_label_segs),
            "interlude_segments": len(interlude_segs),
            "vocal_hist": {str(k): v for k, v in vocal_hist.items()},
            "interlude_hist": {str(k): v for k, v in interlude_hist.items()},
        }
        per_file_rows.append(row)

        if export_dir:
            out_csv = export_dir / f"{pth.stem}_len_hist.csv"
            with out_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["length_sec_bucket", "vocal_count", "interlude_count"])
                keys = sorted({*vocal_hist.keys(), *interlude_hist.keys()})
                for k in keys:
                    w.writerow([k, vocal_hist.get(k, 0), interlude_hist.get(k, 0)])

    print(
        json.dumps(
            {
                "overall": overall
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if export_dir:
        (export_dir / "overall_stats.json").write_text(
            json.dumps({"overall": overall}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Return non-zero if everything got skipped
    return 0 if int(overall["files"]) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
