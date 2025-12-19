from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import matplotlib as mpl
from matplotlib.widgets import Button, TextBox
from matplotlib.ticker import FuncFormatter
from common.config import SR, N_MELS, N_FFT, HOP_LENGTH, HOP_SEC
from common.data import load_mel, labels_to_frame_targets
from common.postproc import compute_metrics_from_arrays, apply_hysteresis_seq
from common.export_utils import compute_frame_dur, build_hyst_segments, make_hyst_payload
try:
    import onnxruntime as ort
except Exception:
    ort = None


# 与训练保持一致的特征参数从 common.config 读取

 # 统一使用 common.export_utils 中的导出工具

def format_time_ms(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:06.3f}"


def _compute_boundaries(labels: Optional[List[Dict]]) -> List[float]:
    """Collect unique boundary times (segment starts/ends) in seconds."""
    if not labels:
        return []
    boundaries: List[float] = []
    for it in labels:
        sm, ss = it.get("start", [0, 0])[0], float(it.get("start", [0, 0])[1])
        em, es = it.get("end", [0, 0])[0], float(it.get("end", [0, 0])[1])
        ssec = sm * 60.0 + ss
        esec = em * 60.0 + es
        boundaries.append(ssec)
        boundaries.append(esec)
    boundaries.sort()
    # Deduplicate with a small tolerance
    uniq: List[float] = []
    for b in boundaries:
        if not uniq or abs(b - uniq[-1]) > 1e-6:
            uniq.append(b)
    return uniq


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


def _load_onnx_session(onnx_path: Path):
    if ort is None:
        raise RuntimeError("未安装 onnxruntime，无法加载 ONNX 模型")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = {}
    try:
        m = sess.get_modelmeta()
        for kv in getattr(m, "custom_metadata_map", {}) or {}:
            meta[kv] = m.custom_metadata_map[kv]
        for kv in getattr(m, "metadata_props", []) or []:
            if hasattr(kv, "key") and hasattr(kv, "value"):
                meta[str(kv.key)] = str(kv.value)
    except Exception:
        meta = {}
    config = {}
    try:
        if "config" in meta:
            config = json.loads(meta["config"])
    except Exception:
        config = {}
    return sess, config

def load_model(ckpt_path: Path):
    if ckpt_path.suffix.lower() == ".onnx":
        sess, config = _load_onnx_session(ckpt_path)
        return sess, config
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    config = ckpt.get("config", {})
    bilstm = bool(config.get("bilstm") or (str(config.get("rnn_type", "")).lower() == "bilstm") or config.get("bidirectional"))
    state = ckpt.get("model_state", ckpt)
    model = CNNLSTM2Head(input_dim=N_MELS, cnn_channels=64, hidden=128, bidirectional=bilstm)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, config


def _sigmoid_np(a: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-a))).astype(np.float32)

def _run_infer_onnx(sess, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = X.astype(np.float32)[None, :, :]
    ins = sess.get_inputs()
    input_name = ins[0].name
    outputs = sess.run(None, {input_name: x})
    probs_cls = None
    probs_reg = None
    if len(outputs) == 2:
        o0 = np.array(outputs[0])
        o1 = np.array(outputs[1])
        o0 = np.squeeze(o0)
        o1 = np.squeeze(o1)
        if o0.ndim == 2 and o0.shape[-1] == 1:
            o0 = o0[..., 0]
        if o1.ndim == 2 and o1.shape[-1] == 1:
            o1 = o1[..., 0]
        probs_cls = o0
        probs_reg = o1
    # elif len(outputs) == 1:
    #     o = np.array(outputs[0])
    #     o = np.squeeze(o)
    #     if o.ndim == 2 and o.shape[1] == 2:
    #         probs_cls = o[:, 0]
    #         probs_reg = o[:, 1]
    #     else:
    #         probs_cls = o
    #         probs_reg = np.zeros_like(probs_cls, dtype=np.float32)
    else:
        raise RuntimeError("ONNX 模型输出不符合预期")

    if np.nanmin(probs_cls) < 0.0 or np.nanmax(probs_cls) > 1.0:
        probs_cls = _sigmoid_np(probs_cls.astype(np.float32))
    if np.nanmin(probs_reg) < 0.0 or np.nanmax(probs_reg) > 1.0:
        probs_reg = _sigmoid_np(probs_reg.astype(np.float32))
    # probs_cls = (1.0 - probs_cls).astype(np.float32)
    probs_reg = probs_reg.astype(np.float32)
    return probs_cls, probs_reg

def run_infer(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ort is not None and hasattr(model, "run"):
        return _run_infer_onnx(model, X)
    with torch.no_grad():
        logits_cls, logits_reg = model(torch.from_numpy(X).unsqueeze(0))
        probs_cls = torch.sigmoid(logits_cls).squeeze(0).cpu().numpy()
        probs_reg = torch.sigmoid(logits_reg).squeeze(0).cpu().numpy()
    return probs_cls.astype(np.float32), probs_reg.astype(np.float32)


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
    parser = argparse.ArgumentParser(description="CNN+LSTM 推理可视化：不在 labels/perfect 下的音频，与 labels 对照")
    parser.add_argument("--music-root", default=r"D:\meliris\music", help="音乐目录")
    parser.add_argument("--labels-root", default=str(Path(__file__).resolve().parent.parent / "labels_qrc"), help="标签目录")
    parser.add_argument("--with-perfect", action="store_true", help="包含 labels/perfect 下的音频")
    parser.add_argument("--stem", default=None, help="指定文件名 stem（不含扩展名）")
    parser.add_argument("--audio-path", default=None, help="指定音频绝对路径，优先于 stem/candidates")
    parser.add_argument("--ckpt", default=str(Path(__file__).resolve().parent.parent / "cnn_lstm_2task_inst.pt"), help="模型权重路径")
    parser.add_argument("--export-json", default=None, help="导出初始 inst_pred_hyst 为 JSON（需启用迟滞）")
    args = parser.parse_args()

    music_root = Path(args.music_root)
    labels_root = Path(args.labels_root)
    ckpt_path = Path(args.ckpt)

    candidates = find_candidates(music_root, labels_root, with_perfect=args.with_perfect)
    audio_path: Optional[Path] = None
    # 优先使用绝对路径
    if args.audio_path:
        ap = Path(args.audio_path)
        if not ap.exists() or not ap.is_file():
            print(f"指定音频不存在或不是文件：{ap}")
            return
        audio_path = ap
        # 若指定了绝对路径，则候选列表仅该文件，Prev/Next 仍可用但无效
        candidates = [ap]
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
    model, config = load_model(ckpt_path)
    X = load_mel(audio_path)
    T = X.shape[0]
    times = np.arange(T) * HOP_SEC

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

    probs_cls, probs_ratio = run_infer(model, X)
    mode = str(config.get("mode", "instrumental"))
    inst_probs = probs_cls if mode == "instrumental" else (1.0 - probs_cls)
    inst_label = (1.0 - y_true) if has_label else np.zeros((T,), dtype=np.float32)

    inst_pred = (inst_probs >= 0.5).astype(np.float32)
    m0 = compute_metrics_from_arrays(inst_pred, inst_label)
    p, r, f1 = m0["precision"], m0["recall"], m0["f1"]

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.39)
    label_prob = "inst_prob (model p)" if mode == "instrumental" else "inst_prob (1 - vocal_p)"
    line_inst_prob, = ax.plot(times, inst_probs, label=label_prob, zorder=2)
    line_inst_true, = ax.plot(times, inst_label, label="inst_label", zorder=2)
    line_inst_pred, = ax.plot(times, inst_pred, label="inst_pred", zorder=2)
    # ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instrumental Probability / Binary")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, pos: format_time_ms(t)))
    ax.format_coord = lambda x, y: f"t={format_time_ms(x)}, value={y:.3f}"

    txt = ax.text(0.01, 1.02, f"Precision: {p:.3f}  |  Recall: {r:.3f}  |  F1: {f1:.3f}", transform=ax.transAxes)

    pos = ax.get_position()
    ax_ratio = plt.axes([pos.x0, 0.21, pos.width, 0.17], sharex=ax)
    line_ratio_pred, = ax_ratio.plot(times, probs_ratio, color="tab:orange")
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_ylim(-0.05, 1.05)
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.xaxis.set_major_formatter(FuncFormatter(lambda t, pos: format_time_ms(t)))
    # ax_ratio.legend(loc="upper right")
    ax_ratio.set_xlim(ax.get_xlim())

    ax_th = plt.axes([0.15, 0.12, 0.7, 0.03])
    s_th = Slider(ax_th, "threshold", 0.0, 1.0, valinit=0.5, valstep=0.01)

    on_init = config.get("eval_hyst_on", None)
    off_init = config.get("eval_hyst_off", None)
    postproc = str(config.get("eval_postproc", "threshold"))
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

    # 若指定导出路径，按初始迟滞参数导出 inst_pred_hyst
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
        if use_hyst and (s_on.val > s_off.val):
            pred = apply_hysteresis_seq(inst_probs, float(s_on.val), float(s_off.val))
            line_inst_pred.set_label("inst_pred_hyst")
        else:
            pred = (inst_probs >= float(s_th.val)).astype(np.float32)
            line_inst_pred.set_label("inst_pred")
        if filter_short:
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
        nonlocal use_hyst, filter_short
        if label == "hysteresis":
            use_hyst = not use_hyst
        elif label == "filter<500ms":
            filter_short = not filter_short
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
        nonlocal audio_path, stem, X, T, times, labels, y_true, has_label, inst_probs, inst_label, boundary_spans
        audio_path = new_audio
        stem = audio_path.stem
        print(f"切换音频: {audio_path}")
        X = load_mel(audio_path)
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
        probs2_cls, probs2_ratio = run_infer(model, X)
        inst_probs = probs2_cls if mode == "instrumental" else (1.0 - probs2_cls)
        line_ratio_pred.set_xdata(times)
        line_ratio_pred.set_ydata(probs2_ratio)
        inst_label = (1.0 - y_true) if has_label else np.zeros((T,), dtype=np.float32)
        line_inst_prob.set_xdata(times)
        line_inst_prob.set_ydata(inst_probs)
        line_inst_true.set_xdata(times)
        line_inst_true.set_ydata(inst_label)
        line_inst_pred.set_xdata(times)
        ax.set_xlim(times[0], times[-1] if len(times) > 0 else 1.0)
        title_text.set_text(f"{stem}")
        ann.set_visible(False)
        # 刷新分隔带
        for sp in boundary_spans:
            try:
                sp.remove()
            except Exception:
                pass
        boundary_spans = []
        _add_boundary_spans()
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

    # 在标签交界处绘制红色分隔带（宽度 = 一个横轴单位 HOP_SEC）
    boundary_spans: List[Any] = []

    def _add_boundary_spans():
        if len(times) == 0:
            return
        bounds = _compute_boundaries(labels)
        if not bounds:
            return
        x_min = times[0]
        x_max = times[-1]
        for b in bounds:
            left = max(x_min, b - HOP_SEC / 2)
            right = min(x_max, b + HOP_SEC / 2)
            if right <= left:
                continue
            sp = ax.axvspan(left, right, color="red", alpha=0.50, zorder=1)
            boundary_spans.append(sp)

    _add_boundary_spans()

    # 导出按钮：将当前迟滞二值预测导出为 JSON
    ax_export = plt.axes([0.36, 0.02, 0.10, 0.04])
    btn_export = Button(ax_export, "Export")

    def on_export(event):
        # 仅在启用迟滞且参数合法时导出
        if not use_hyst or not (s_on.val > s_off.val):
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


# 设置系统常见字体，避免使用 DejaVu Sans（按可用字体自动回退）
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei"
]
mpl.rcParams["axes.unicode_minus"] = False

if __name__ == "__main__":
    main()
