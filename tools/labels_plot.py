from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

from common.config import HOP_SEC
from common.data import labels_to_frame_targets


def _format_time_ms(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:06.3f}"


def _estimate_T_from_labels(labels: List[Dict]) -> int:
    last_end_sec = 0.0
    for it in labels:
        end_m, end_s = it["end"][0], float(it["end"][1])
        end_sec = end_m * 60.0 + end_s
        if end_sec > last_end_sec:
            last_end_sec = end_sec
    return max(1, int(np.ceil(last_end_sec / HOP_SEC)) + 1)


def _collect_label_files(labels_root: Path) -> List[Path]:
    files: List[Path] = []
    for p in labels_root.glob("*.json"):
        if p.is_file():
            files.append(p)
    perfect_root = labels_root / "perfect"
    for p in perfect_root.glob("*.json"):
        if p.is_file():
            files.append(p)
    files.sort(key=lambda x: x.stem.lower())
    return files


def main():
    parser = argparse.ArgumentParser(description="可视化 labels JSON（含 ./labels 与 ./labels/perfect）")
    parser.add_argument("--labels-root", default=str(Path(__file__).resolve().parent.parent / "labels_qrc"))
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    label_files = _collect_label_files(labels_root)
    if not label_files:
        print("未找到 labels JSON 文件。")
        return

    json_path: Optional[Path] = None
    if args.stem:
        for p in label_files:
            if p.stem == args.stem:
                json_path = p
                break
    if json_path is None:
        json_path = label_files[0]

    try:
        labels = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        labels = []

    T = _estimate_T_from_labels(labels)
    times = np.arange(T) * HOP_SEC
    y = labels_to_frame_targets(labels, T)

    fig, ax = plt.subplots(figsize=(12, 4))
    plt.subplots_adjust(bottom=0.2)
    line_label, = ax.plot(times, y, label="label_binary")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Label")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(times[0], times[-1] if len(times) > 0 else 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, pos: _format_time_ms(t)))
    ax.format_coord = lambda x, yv: f"t={_format_time_ms(x)}, value={yv:.3f}"

    title_text = fig.text(0.5, 0.93, f"{json_path.relative_to(labels_root)}", ha="center", va="top")

    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8),
    )
    ann.set_visible(False)

    def _find_segment_at_time(t: float) -> Optional[Dict]:
        for it in labels:
            sm, ss = it["start"][0], float(it["start"][1])
            em, es = it["end"][0], float(it["end"][1])
            ssec = sm * 60.0 + ss
            esec = em * 60.0 + es
            if ssec <= t < esec:
                return it
        return None

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            ann.set_visible(False)
            fig.canvas.draw_idle()
            return
        x = float(event.xdata)
        idx = int(np.clip(round(x / HOP_SEC), 0, T - 1))
        t = times[idx]
        val = float(y[idx])
        seg = _find_segment_at_time(t)
        if seg is None:
            ann.set_visible(False)
            fig.canvas.draw_idle()
            return
        ann.xy = (t, val)
        content = str(seg.get("content", ""))
        sm, ss = seg["start"][0], float(seg["start"][1])
        em, es = seg["end"][0], float(seg["end"][1])
        ann.set_text(f"{_format_time_ms(t)} | {content}\n[{sm:02d}:{ss:06.3f} → {em:02d}:{es:06.3f}]")
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    current_idx = 0
    for i, p in enumerate(label_files):
        if p == json_path:
            current_idx = i
            break

    def update_plot_data(new_json: Path):
        nonlocal json_path, labels, T, times, y
        json_path = new_json
        try:
            labels = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            labels = []
        T = _estimate_T_from_labels(labels)
        times = np.arange(T) * HOP_SEC
        y = labels_to_frame_targets(labels, T)
        line_label.set_xdata(times)
        line_label.set_ydata(y)
        ax.set_xlim(times[0], times[-1] if len(times) > 0 else 1.0)
        title_text.set_text(f"{json_path.relative_to(labels_root)}")
        ann.set_visible(False)
        fig.canvas.draw_idle()

    ax_prev = plt.axes([0.15, 0.06, 0.08, 0.06])
    ax_next = plt.axes([0.25, 0.06, 0.08, 0.06])
    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")

    def on_prev(event):
        nonlocal current_idx
        if not label_files:
            return
        current_idx = (current_idx - 1) % len(label_files)
        update_plot_data(label_files[current_idx])

    def on_next(event):
        nonlocal current_idx
        if not label_files:
            return
        current_idx = (current_idx + 1) % len(label_files)
        update_plot_data(label_files[current_idx])

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    plt.show()


mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei"
]
mpl.rcParams["axes.unicode_minus"] = False


if __name__ == "__main__":
    main()

