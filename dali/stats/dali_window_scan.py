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
    if not (isinstance(mm_ss, (list, tuple)) and len(mm_ss) == 2):
        raise ValueError(f"Invalid time tuple: {mm_ss!r}")
    return float(mm_ss[0]) * 60.0 + float(mm_ss[1])


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
        segs.append(Segment(start=float(start), end=float(end)))

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


@dataclass(frozen=True)
class TimelineIndex:
    segs: list[Segment]
    starts: list[float]
    ends: list[float]
    boundaries: list[tuple[float, str]]


def _build_timeline_index(vocal_segs_merged: list[Segment], span_start: float, span_end: float) -> TimelineIndex:
    clipped = [
        Segment(start=max(s.start, span_start), end=min(s.end, span_end))
        for s in vocal_segs_merged
        if min(s.end, span_end) > max(s.start, span_start)
    ]
    starts = [s.start for s in clipped]
    ends = [s.end for s in clipped]

    boundaries: list[tuple[float, str]] = []
    for s in clipped:
        boundaries.append((s.start, "start"))
        boundaries.append((s.end, "end"))
    boundaries.sort(key=lambda x: x[0])

    return TimelineIndex(segs=clipped, starts=starts, ends=ends, boundaries=boundaries)


def _bisect_right(a: list[float], x: float) -> int:
    # local tiny bisect_right to avoid imports
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _contains_vocal(idx: TimelineIndex, a: float, b: float) -> bool:
    """True iff [a,b) is fully inside some merged vocal segment."""
    if b <= a:
        return True
    i = _bisect_right(idx.starts, a) - 1
    if i < 0:
        return False
    return idx.ends[i] >= b


def _overlaps_vocal(idx: TimelineIndex, a: float, b: float) -> bool:
    """True iff [a,b) overlaps any vocal segment."""
    if b <= a:
        return False
    # first segment with end > a
    j = _bisect_right(idx.ends, a)
    if j >= len(idx.segs):
        return False
    return idx.starts[j] < b


def _vocal_overlaps(idx: TimelineIndex, a: float, b: float) -> list[tuple[Segment, float, float]]:
    """Return overlapped (segment, overlap_start, overlap_end) for window [a,b)."""
    if b <= a:
        return []
    j = _bisect_right(idx.ends, a)
    out: list[tuple[Segment, float, float]] = []
    for k in range(j, len(idx.segs)):
        s = idx.segs[k]
        if s.start >= b:
            break
        os = max(a, s.start)
        oe = min(b, s.end)
        if oe > os:
            out.append((s, os, oe))
    return out


def _is_pure_vocal(idx: TimelineIndex, w0: float, w1: float, tol: float) -> bool:
    """Pure-vocal window allowing edge interlude.

    Rule: the window overlaps exactly one merged vocal segment, and the total interlude length
    inside the window (which can only appear at the two edges) is <= 2*tol.
    """
    if w1 <= w0:
        return False
    overlaps = _vocal_overlaps(idx, w0, w1)
    if len(overlaps) != 1:
        return False
    seg, _, _ = overlaps[0]
    left_interlude = max(0.0, seg.start - w0)
    right_interlude = max(0.0, w1 - seg.end)
    return (left_interlude + right_interlude) <= (2.0 * tol)


def _is_pure_interlude(idx: TimelineIndex, w0: float, w1: float, tol: float) -> bool:
    """Pure-interlude window allowing edge vocal.

    Rule: the total vocal overlap inside the window is <= 2*tol, and any vocal overlap must
    be stuck to the window edges (no vocal in the middle).
    """
    if w1 <= w0:
        return False
    overlaps = _vocal_overlaps(idx, w0, w1)
    if not overlaps:
        return True

    allowed = 2.0 * tol
    if len(overlaps) == 1:
        _, os, oe = overlaps[0]
        touches_left = os == w0
        touches_right = oe == w1
        if not (touches_left or touches_right):
            return False
        return (oe - os) <= allowed

    if len(overlaps) == 2:
        (_, os0, oe0), (_, os1, oe1) = overlaps
        if os0 != w0:
            return False
        if oe1 != w1:
            return False
        # Must have a true interlude middle (no overlap between these)
        if oe0 > os1:
            return False
        return ((oe0 - os0) + (oe1 - os1)) <= allowed

    return False


def _has_transition(idx: TimelineIndex, w0: float, w1: float, tol: float) -> bool:
    if w1 <= w0:
        return False
    if tol <= 0:
        # Any boundary inside the window counts as transition (pure cases are checked earlier).
        for t, _kind in idx.boundaries:
            if t < w0:
                continue
            if t > w1:
                break
            return True
        return False

    left = w0 + tol
    right = w1 - tol
    if right <= left:
        return False

    # A transition boundary must have at least tol duration of different states on both sides.
    # For a vocal-start boundary at t:
    #   [t-tol, t) should be interlude (no vocal), [t, t+tol) should be vocal.
    # For a vocal-end boundary at t:
    #   [t-tol, t) should be vocal, [t, t+tol) should be interlude.
    for t, kind in idx.boundaries:
        if t < left:
            continue
        if t > right:
            break
        if (t - tol) < w0 or (t + tol) > w1:
            continue
        if kind == "start":
            if _overlaps_vocal(idx, t - tol, t):
                continue
            if not _contains_vocal(idx, t, t + tol):
                continue
            return True
        if kind == "end":
            if not _contains_vocal(idx, t - tol, t):
                continue
            if _overlaps_vocal(idx, t, t + tol):
                continue
            return True
    return False


def _classify_window(idx: TimelineIndex, w0: float, w1: float, tol: float) -> str:
    if _is_pure_vocal(idx, w0, w1, tol):
        return "pure_vocal"
    if _is_pure_interlude(idx, w0, w1, tol):
        return "pure_interlude"
    if _has_transition(idx, w0, w1, tol):
        return "transition"
    return "other"


def _scan_counts(idx: TimelineIndex, span_start: float, span_end: float, window_sec: float, hop_sec: float, tol: float) -> dict:
    span_len = span_end - span_start
    if span_len < window_sec:
        return {"windows": 0, "pure_vocal": 0, "pure_interlude": 0, "transition": 0, "other": 0}

    n = int(math.floor((span_len - window_sec) / hop_sec)) + 1
    counts = {"windows": n, "pure_vocal": 0, "pure_interlude": 0, "transition": 0, "other": 0}

    for i in range(n):
        w0 = span_start + i * hop_sec
        w1 = w0 + window_sec
        lab = _classify_window(idx, w0, w1, tol)
        counts[lab] += 1

    return counts


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "以指定窗口长度滑动扫描 labels_dali，统计窗口类型数量：\n"
            "- 纯人声：允许窗口内总计 <= 2*tolerance 的间奏，但必须都贴边(不能在中间出现)\n"
            "- 纯间奏：允许窗口内总计 <= 2*tolerance 的人声，但必须都贴边(不能在中间出现)\n"
            "- 切换：窗口内存在人声/间奏边界，且切换点两侧都至少 tolerance\n"
            "同时输出 tolerance=0 与 tolerance=500ms(默认) 两套统计。"
        )
    )
    p.add_argument("--labels-root", type=str, default=str(Path(".") / "labels_dali"))
    p.add_argument("--window-sec", type=float, required=True, help="滑窗长度(秒)")
    p.add_argument("--hop-sec", type=float, default=1.0, help="滑动步长(秒)，默认 1.0")
    p.add_argument(
        "--tolerance-ms",
        type=float,
        default=500.0,
        help="贴边容忍长度(毫秒)，默认 500；纯类允许的异类总长度为 2*tolerance",
    )
    p.add_argument(
        "--span",
        choices=["zero_to_last", "first_to_last"],
        default="first_to_last",
        help="扫描跨度：first_to_last(默认) 或 zero_to_last",
    )
    p.add_argument(
        "--export-csv",
        type=str,
        default="",
        help="可选：导出每首歌统计到 CSV(路径)",
    )
    p.add_argument("--max-files", type=int, default=0, help="仅处理前 N 个文件(调试用)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    labels_root = Path(args.labels_root)
    if not labels_root.exists():
        raise SystemExit(f"labels_root not found: {labels_root}")

    window_sec = float(args.window_sec)
    hop_sec = float(args.hop_sec)
    tol = float(args.tolerance_ms) / 1000.0

    if not (window_sec > 0 and hop_sec > 0):
        raise SystemExit("--window-sec and --hop-sec must be > 0")

    if tol < 0:
        raise SystemExit("--tolerance-ms must be >= 0")

    # Note: window_sec may be <= 2*tolerance; in that case, 'transition' becomes very hard
    # to satisfy, and pure classes may become more permissive.

    export_csv = Path(args.export_csv) if args.export_csv else None

    label_files = sorted(labels_root.glob("*.json"))
    if args.max_files and args.max_files > 0:
        label_files = label_files[: args.max_files]

    overall = {
        "labels_root": str(labels_root),
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "tolerance_ms": float(args.tolerance_ms),
        "span": args.span,
        "files": 0,
        "files_skipped": 0,
        "tolerance_0": {"windows": 0, "pure_vocal": 0, "pure_interlude": 0, "transition": 0, "other": 0},
        "tolerance_applied": {"windows": 0, "pure_vocal": 0, "pure_interlude": 0, "transition": 0, "other": 0},
    }

    per_file_rows: list[dict] = []

    for pth in label_files:
        try:
            segs_raw = _load_segments(pth)
        except Exception:
            overall["files_skipped"] += 1
            continue

        if not segs_raw:
            overall["files_skipped"] += 1
            continue

        merged = _merge_union(segs_raw)
        first_start = merged[0].start
        last_end = max(s.end for s in merged)

        span_start = 0.0 if args.span == "zero_to_last" else float(first_start)
        span_end = float(last_end)

        if span_end <= span_start:
            overall["files_skipped"] += 1
            continue

        idx = _build_timeline_index(merged, span_start=span_start, span_end=span_end)

        c0 = _scan_counts(idx, span_start, span_end, window_sec=window_sec, hop_sec=hop_sec, tol=0.0)
        c1 = _scan_counts(idx, span_start, span_end, window_sec=window_sec, hop_sec=hop_sec, tol=tol)

        overall["files"] += 1
        for k in overall["tolerance_0"].keys():
            overall["tolerance_0"][k] += int(c0[k])
            overall["tolerance_applied"][k] += int(c1[k])

        row = {
            "id": pth.stem,
            "span_start_sec": span_start,
            "span_end_sec": span_end,
            "tolerance_0": c0,
            "tolerance_applied": c1,
        }
        per_file_rows.append(row)

    if export_csv:
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        with export_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "span_start_sec",
                    "span_end_sec",
                    "w0_windows",
                    "w0_pure_vocal",
                    "w0_pure_interlude",
                    "w0_transition",
                    "w0_other",
                    "w1_windows",
                    "w1_pure_vocal",
                    "w1_pure_interlude",
                    "w1_transition",
                    "w1_other",
                ]
            )
            for r in per_file_rows:
                w0 = r["tolerance_0"]
                w1 = r["tolerance_applied"]
                w.writerow(
                    [
                        r["id"],
                        r["span_start_sec"],
                        r["span_end_sec"],
                        w0["windows"],
                        w0["pure_vocal"],
                        w0["pure_interlude"],
                        w0["transition"],
                        w0["other"],
                        w1["windows"],
                        w1["pure_vocal"],
                        w1["pure_interlude"],
                        w1["transition"],
                        w1["other"],
                    ]
                )

    print(json.dumps({"overall": overall, "per_file": per_file_rows if export_csv else None}, ensure_ascii=False, indent=2))

    return 0 if int(overall["files"]) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
