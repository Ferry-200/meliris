from __future__ import annotations

import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from pathlib import Path
from typing import Optional, List

from common import SR, HOP_LENGTH, HOP_SEC, load_features, labels_to_frame_targets, apply_hysteresis_seq, compute_metrics_from_arrays
from common.export_utils import compute_frame_dur, build_hyst_segments, make_hyst_payload


class CNNLSTM2Head(nn.Module):
    def __init__(self, input_dim: int, cnn_channels: int = 64, hidden: int = 128, bidirectional: bool = False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head_cls = nn.Linear(out_dim, 1)
        self.head_reg = nn.Linear(out_dim, 1)

    def forward(self, X: torch.Tensor):
        B, T, F = X.shape
        x = X.reshape(B * T, 1, F)
        x = self.cnn(x).squeeze(-1)
        x = x.reshape(B, T, -1)
        out, _ = self.lstm(x)
        logits_cls = self.head_cls(out).squeeze(-1)
        logits_reg = self.head_reg(out).squeeze(-1)
        return logits_cls, logits_reg


def load_model(ckpt_path: Path):
    ck = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ck.get("config", {})
    bilstm = bool(cfg.get("bilstm") or (str(cfg.get("rnn_type", "")).lower() == "bilstm") or cfg.get("bidirectional"))
    feat_dim = int(cfg.get("feature_dim", 0))
    state = ck.get("model_state", ck)
    model = CNNLSTM2Head(input_dim=feat_dim or 256, cnn_channels=64, hidden=128, bidirectional=bilstm)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def run_infer(model: nn.Module, X: np.ndarray):
    with torch.no_grad():
        t = torch.from_numpy(X.astype(np.float32)).unsqueeze(0)
        logits_cls, logits_reg = model(t)
        probs = torch.sigmoid(logits_cls).squeeze(0).detach().cpu().numpy().astype(np.float32)
        ratio = torch.sigmoid(logits_reg).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return probs, ratio


def format_time_ms(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:06.3f}"


def filter_short_runs(pred: np.ndarray, min_frames: int) -> np.ndarray:
    arr = pred.astype(np.float32).copy()
    n = arr.shape[0]
    i = 0
    while i < n:
        if arr[i] >= 0.5:
            j = i + 1
            while j < n and arr[j] >= 0.5:
                j += 1
            if (j - i) < min_frames:
                arr[i:j] = 0.0
            i = j
        else:
            i += 1
    return arr


def find_candidates(music_root: Path, labels_root: Path, with_perfect: bool = False) -> List[Path]:
    perfect_root = labels_root / "perfect"
    candidates: List[Path] = []
    for p in music_root.glob("*"):
        if p.is_file():
            stem = p.stem
            if (not (perfect_root / f"{stem}.json").exists()) or with_perfect:
                candidates.append(p)
    return candidates


def main():
    ap = argparse.ArgumentParser(description="CNN+LSTM 多任务推理可视化（多特征输入）")
    ap.add_argument("--music-root", default=r"D:\meliris\music")
    ap.add_argument("--labels-root", default=str(Path(".") / "labels_qrc"))
    ap.add_argument("--with-perfect", action="store_true")
    ap.add_argument("--stem", default=None)
    ap.add_argument("--audio-path", "--audio", dest="audio_path", default=None)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--export-json", default=None)
    args = ap.parse_args()

    music_root = Path(args.music_root)
    labels_root = Path(args.labels_root)
    ckpt_path = Path(args.ckpt)

    candidates = find_candidates(music_root, labels_root, with_perfect=bool(args.with_perfect))
    audio_path: Optional[Path] = None
    if args.audio_path:
        apath = Path(args.audio_path)
        if (not apath.exists()) or (not apath.is_file()):
            print(f"指定音频不存在或不是文件：{apath}")
            return
        audio_path = apath
        candidates = [apath]
    else:
        if not candidates:
            print("未找到候选音频（不在 labels/perfect 下的同名文件）。")
            return
        if args.stem:
            for p in candidates:
                if p.stem == args.stem:
                    audio_path = p
                    break
            if audio_path is None:
                print(f"未找到指定 stem: {args.stem}，将使用第一个候选。")
        if audio_path is None:
            audio_path = candidates[0]

    stem = audio_path.stem
    print(f"目标音频: {audio_path}")

    if not ckpt_path.exists():
        print(f"未找到模型权重：{ckpt_path}")
        return
    model, cfg = load_model(ckpt_path)
    X = load_features(audio_path)
    T = X.shape[0]
    times = np.arange(T) * HOP_SEC
    probs, ratio = run_infer(model, X)
    mode = str(cfg.get("mode", "vocal"))
    inst_probs = probs if mode == "instrumental" else (1.0 - probs)

    label_json_path = labels_root / f"{stem}.json"
    perfect_json_path = labels_root / "perfect" / f"{stem}.json"
    labels = None
    if label_json_path.exists():
        try:
            labels = json.loads(label_json_path.read_text(encoding="utf-8"))
        except Exception:
            labels = None
    elif perfect_json_path.exists():
        try:
            labels = json.loads(perfect_json_path.read_text(encoding="utf-8"))
        except Exception:
            labels = None
    y_true = labels_to_frame_targets(labels or [], T)
    has_label = labels is not None
    inst_label = (1.0 - y_true) if has_label else np.zeros((T,), dtype=np.float32)

    inst_pred = (inst_probs >= 0.5).astype(np.float32)
    m0 = compute_metrics_from_arrays(inst_pred, inst_label)
    p0, r0, f10 = m0["precision"], m0["recall"], m0["f1"]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.39)
    label_prob = "inst_prob (model p)" if mode == "instrumental" else "inst_prob (1 - vocal_p)"
    line_inst_prob, = ax.plot(times, inst_probs, label=label_prob)
    line_inst_true, = ax.plot(times, inst_label, label="inst_label")
    line_inst_pred, = ax.plot(times, inst_pred, label="inst_pred")
    ax.set_ylabel("Instrumental Probability / Binary")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, pos: format_time_ms(t)))
    ax.format_coord = lambda x, y: f"t={format_time_ms(x)}, value={y:.3f}"

    txt = ax.text(0.01, 1.02, f"Precision: {p0:.3f}  |  Recall: {r0:.3f}  |  F1: {f10:.3f}", transform=ax.transAxes)

    pos = ax.get_position()
    ax_ratio = plt.axes([pos.x0, 0.21, pos.width, 0.17], sharex=ax)
    line_ratio_pred, = ax_ratio.plot(times, ratio, color="tab:orange")
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_ylim(-0.05, 1.05)
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.xaxis.set_major_formatter(FuncFormatter(lambda t, pos: format_time_ms(t)))
    ax_ratio.set_xlim(ax.get_xlim())

    ax_th = plt.axes([0.15, 0.12, 0.7, 0.03])
    s_th = Slider(ax_th, "threshold", 0.0, 1.0, valinit=0.5, valstep=0.01)

    on_init = cfg.get("eval_hyst_on", None)
    off_init = cfg.get("eval_hyst_off", None)
    postproc = str(cfg.get("eval_postproc", "threshold"))
    use_hyst = False
    filter_short = False
    if postproc == "hysteresis" and isinstance(on_init, (int, float)) and isinstance(off_init, (int, float)) and (float(on_init) > float(off_init)):
        use_hyst = True
    th_on0 = float(on_init) if isinstance(on_init, (int, float)) else 0.5
    th_off0 = float(off_init) if isinstance(off_init, (int, float)) else max(0.0, 0.5 - 0.2)

    ax_on = plt.axes([0.15, 0.07, 0.32, 0.03])
    ax_off = plt.axes([0.53, 0.07, 0.32, 0.03])
    s_on = Slider(ax_on, "th_on", 0.0, 1.0, valinit=th_on0, valstep=0.01)
    s_off = Slider(ax_off, "th_off", 0.0, 1.0, valinit=th_off0, valstep=0.01)
    ax_chk = plt.axes([0.90, 0.07, 0.10, 0.10])
    chk = CheckButtons(ax_chk, ["hysteresis", "filter<500ms"], [use_hyst, filter_short])

    if args.export_json:
        export_path = Path(args.export_json)
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            if use_hyst and (th_on0 > th_off0):
                pred_hyst0 = apply_hysteresis_seq(inst_probs, th_on0, th_off0).astype(np.float32)
                frame_dur = compute_frame_dur(times, HOP_SEC)
                segments = build_hyst_segments(pred_hyst0, times, frame_dur)
                payload = make_hyst_payload(stem, audio_path, mode, th_on0, th_off0, SR, HOP_LENGTH, HOP_SEC, times, pred_hyst0, segments)
                export_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                print(f"已导出 inst_pred_hyst 到：{export_path}")
            else:
                print("未启用迟滞或参数非法（th_on <= th_off），跳过导出。")
        except Exception as e:
            print(f"导出 JSON 失败：{e}")

    current_pred = inst_pred.copy()

    def recompute_and_update():
        nonlocal current_pred
        uh = chk.get_status()[0] if hasattr(chk, "get_status") else use_hyst
        fs = chk.get_status()[1] if hasattr(chk, "get_status") else filter_short
        if uh and (s_on.val > s_off.val):
            pred = apply_hysteresis_seq(inst_probs, float(s_on.val), float(s_off.val))
            line_inst_pred.set_label("inst_pred_hyst")
        else:
            pred = (inst_probs >= float(s_th.val)).astype(np.float32)
            line_inst_pred.set_label("inst_pred")
        if fs:
            min_frames = max(1, int(math.ceil(0.500 / HOP_SEC)))
            pred = filter_short_runs(pred, min_frames)
            line_inst_pred.set_label(f"{line_inst_pred.get_label()}+filt")
        current_pred = pred.astype(np.float32)
        line_inst_pred.set_ydata(current_pred)
        m2 = compute_metrics_from_arrays(current_pred, inst_label)
        p2, r2, f2 = m2["precision"], m2["recall"], m2["f1"]
        txt.set_text(f"Precision: {p2:.3f}  |  Recall: {r2:.3f}  |  F1: {f2:.3f}")
        ax.legend(loc="lower left")
        fig.canvas.draw_idle()

    def on_th_change(val):
        recompute_and_update()

    def on_on_change(val):
        recompute_and_update()

    def on_off_change(val):
        recompute_and_update()

    def on_chk_clicked(label):
        recompute_and_update()

    s_th.on_changed(on_th_change)
    s_on.on_changed(on_on_change)
    s_off.on_changed(on_off_change)
    chk.on_clicked(on_chk_clicked)

    title_text = fig.text(0.5, 0.95, f"{stem}", ha="center", va="top")

    recompute_and_update()

    current_idx = 0
    for i, p in enumerate(candidates):
        if p == audio_path:
            current_idx = i
            break

    def update_plot_data(new_audio: Path):
        nonlocal audio_path, stem, X, T, times, labels, y_true, has_label, inst_probs, inst_label
        audio_path = new_audio
        stem = audio_path.stem
        print(f"切换音频: {audio_path}")
        X = load_features(audio_path)
        T = X.shape[0]
        times = np.arange(T) * HOP_SEC
        label_json_path2 = labels_root / f"{stem}.json"
        perfect_json_path2 = labels_root / "perfect" / f"{stem}.json"
        labels = None
        if label_json_path2.exists():
            try:
                labels = json.loads(label_json_path2.read_text(encoding="utf-8"))
            except Exception:
                labels = None
        elif perfect_json_path2.exists():
            try:
                labels = json.loads(perfect_json_path2.read_text(encoding="utf-8"))
            except Exception:
                labels = None
        y_true = labels_to_frame_targets(labels or [], T)
        has_label = labels is not None
        probs2, ratio2 = run_infer(model, X)
        inst_probs = probs2 if mode == "instrumental" else (1.0 - probs2)
        line_ratio_pred.set_xdata(times)
        line_ratio_pred.set_ydata(ratio2)
        inst_label = (1.0 - y_true) if has_label else np.zeros((T,), dtype=np.float32)
        line_inst_prob.set_xdata(times)
        line_inst_prob.set_ydata(inst_probs)
        line_inst_true.set_xdata(times)
        line_inst_true.set_ydata(inst_label)
        line_inst_pred.set_xdata(times)
        ax.set_xlim(times[0], times[-1] if len(times) > 0 else 1.0)
        title_text.set_text(f"{stem}")
        ann.set_visible(False)
        recompute_and_update()

    ax_prev = plt.axes([0.15, 0.02, 0.08, 0.04])
    ax_next = plt.axes([0.25, 0.02, 0.08, 0.04])
    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")

    def on_prev(event):
        nonlocal current_idx
        if not candidates:
            return
        current_idx = (current_idx - 1) % len(candidates)
        update_plot_data(candidates[current_idx])

    def on_next(event):
        nonlocal current_idx
        if not candidates:
            return
        current_idx = (current_idx + 1) % len(candidates)
        update_plot_data(candidates[current_idx])

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    ax_export = plt.axes([0.36, 0.02, 0.10, 0.04])
    btn_export = Button(ax_export, "Export")

    def on_export(event):
        uh = chk.get_status()[0] if hasattr(chk, "get_status") else use_hyst
        if not uh or not (s_on.val > s_off.val):
            print("请启用 hysteresis 并确保 th_on > th_off 后再导出。")
            return
        try:
            pred_hyst = apply_hysteresis_seq(inst_probs, float(s_on.val), float(s_off.val)).astype(np.float32)
            export_path = Path(args.export_json) if args.export_json else (Path(".") / "exports" / f"{stem}_inst_pred_hyst.json")
            export_path.parent.mkdir(parents=True, exist_ok=True)
            frame_dur = compute_frame_dur(times, HOP_SEC)
            segments = build_hyst_segments(pred_hyst, times, frame_dur)
            payload = make_hyst_payload(stem, audio_path, mode, s_on.val, s_off.val, SR, HOP_LENGTH, HOP_SEC, times, pred_hyst, segments)
            export_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            print(f"已导出 inst_pred_hyst 到：{export_path}")
        except Exception as e:
            print(f"导出 JSON 失败：{e}")

    btn_export.on_clicked(on_export)

    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8),
    )
    ann.set_visible(False)

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            ann.set_visible(False)
            fig.canvas.draw_idle()
            return
        x = float(event.xdata)
        idx = int(np.clip(round(x / HOP_SEC), 0, T - 1))
        t = times[idx]
        val = float(inst_probs[idx])
        lab = int(inst_label[idx]) if has_label else 0
        bin_pred = int(current_pred[idx])
        ann.xy = (t, val)
        ann.set_text(f"{format_time_ms(t)} | pred={val:.3f} | label={lab} | bin={bin_pred}")
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.show()


mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei"
]
mpl.rcParams["axes.unicode_minus"] = False

if __name__ == "__main__":
    main()
