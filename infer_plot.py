from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import matplotlib as mpl
from matplotlib.widgets import Button, TextBox
from matplotlib.ticker import FuncFormatter


# 与训练保持一致的特征参数
SR = 22050
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
HOP_SEC = HOP_LENGTH / SR

# 支持的音频扩展名（用于浏览/按 stem 查找）
AUDIO_EXTS_ORDER = [
    ".flac",
    ".wav",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
]

def format_time_ms(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    # 显示为 mm:ss.mmm（三位毫秒）
    return f"{m:02d}:{s:06.3f}"


class LSTMSVD(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, sequential_head: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        # 兼容训练时的定义：有的保存为 head.0.*（Sequential 包裹），有的为 head.*（Linear）
        self.head = nn.Sequential(nn.Linear(hidden, 1)) if sequential_head else nn.Linear(hidden, 1)

    def forward(self, X: torch.Tensor):
        out, _ = self.lstm(X)
        logits = self.head(out).squeeze(-1)
        return logits


def load_model(ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    keys = list(state.keys())
    # 检测 head 键名风格
    seq_head = any(k.startswith("head.0.") for k in keys)
    model = LSTMSVD(input_dim=N_MELS, hidden=128, sequential_head=seq_head)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # 尝试重映射键名并切换 head 风格
        remapped = {}
        if seq_head:
            # 训练保存为 head.0.*，当前模型为 Sequential 仍报错，则尝试映射为 head.*
            for k, v in state.items():
                if k.startswith("head.0."):
                    remapped[k.replace("head.0.", "head.")] = v
                else:
                    remapped[k] = v
            model = LSTMSVD(input_dim=N_MELS, hidden=128, sequential_head=False)
        else:
            # 训练保存为 head.*，当前模型为 Linear 仍报错，则尝试映射为 head.0.*
            for k, v in state.items():
                if k.startswith("head."):
                    remapped[k.replace("head.", "head.0.")] = v
                else:
                    remapped[k] = v
            model = LSTMSVD(input_dim=N_MELS, hidden=128, sequential_head=True)
        model.load_state_dict(remapped, strict=False)
    model.eval()
    config = ckpt.get("config", {})
    return model, config


def load_mel(audio_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm.T.astype(np.float32)  # [T, n_mels]


def labels_to_frame_targets(labels: List[Dict], T: int) -> np.ndarray:
    y = np.zeros((T,), dtype=np.float32)
    if not labels:
        return y
    for item in labels:
        start_m, start_s = item["start"][0], item["start"][1]
        end_m, end_s = item["end"][0], item["end"][1]
        start_sec = start_m * 60.0 + float(start_s)
        end_sec = end_m * 60.0 + float(end_s)
        s_idx = max(0, int(math.floor(start_sec / HOP_SEC)))
        e_idx = min(T, int(math.ceil(end_sec / HOP_SEC)))
        if e_idx > s_idx:
            y[s_idx:e_idx] = 1.0
    return y


def run_infer(model: LSTMSVD, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(X).unsqueeze(0))  # [1, T, F] -> [1, T]
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return probs.astype(np.float32)


def find_candidates(music_root: Path, labels_root: Path) -> List[Path]:
    perfect_root = labels_root / "perfect"
    candidates: List[Path] = []
    for p in music_root.glob("*"):
        if p.is_file():
            stem = p.stem
            if not (perfect_root / f"{stem}.json").exists():
                candidates.append(p)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Matplotlib 推理可视化：不在 labels/perfect 下的音频，与 labels 对照")
    parser.add_argument("--music-root", default=r"D:\meliris\music", help="音乐目录")
    parser.add_argument("--labels-root", default=str(Path(__file__).resolve().parent / "labels"), help="标签目录")
    parser.add_argument("--stem", default=None, help="指定文件名 stem（不含扩展名）")
    parser.add_argument("--ckpt", default=str(Path(__file__).resolve().parent / "lstm_svd_inst.pt"), help="模型权重路径")
    args = parser.parse_args()

    music_root = Path(args.music_root)
    labels_root = Path(args.labels_root)
    ckpt_path = Path(args.ckpt)

    candidates = find_candidates(music_root, labels_root)
    if not candidates:
        print("未找到候选音频（不在 labels/perfect 下的同名文件）。")
        return

    # 选择文件
    audio_path: Optional[Path] = None
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

    # 加载模型与特征
    if not ckpt_path.exists():
        print(f"未找到模型权重：{ckpt_path}")
        return
    model, config = load_model(ckpt_path)
    X = load_mel(audio_path)  # [T, F]
    T = X.shape[0]
    times = np.arange(T) * HOP_SEC

    # 读取标签（labels/<stem>.json，用于对照）
    label_json_path = labels_root / f"{stem}.json"
    labels = None
    if label_json_path.exists():
        try:
            labels = json.loads(label_json_path.read_text(encoding="utf-8"))
        except Exception:
            labels = None
    y_true = labels_to_frame_targets(labels or [], T)
    has_label = labels is not None

    # 推理：根据训练模式决定概率解释
    probs = run_infer(model, X)
    mode = str(config.get("mode", "vocal"))
    inst_probs = probs if mode == "instrumental" else (1.0 - probs)
    inst_label = (1.0 - y_true) if has_label else np.zeros((T,), dtype=np.float32)
    # 默认阈值：若权重包含 eval_threshold，则使用该值作为初始阈值
    # default_th = float(config.get("eval_threshold", 0.5))
    # default_th = float(np.clip(default_th, 0.0, 1.0))
    inst_pred = (inst_probs >= 0.5).astype(np.float32)

    # 计算指标
    def metrics(pred: np.ndarray, true: np.ndarray):
        tp = int(((pred == 1) & (true == 1)).sum())
        fp = int(((pred == 1) & (true == 0)).sum())
        fn = int(((pred == 0) & (true == 1)).sum())
        precision = tp / (tp + fp + 1e-8) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn + 1e-8) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0
        return precision, recall, f1

    p, r, f1 = metrics(inst_pred, inst_label)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 5))
    # 为多个控件与下方标题预留空间
    plt.subplots_adjust(bottom=0.39)
    label_prob = "inst_prob (model p)" if mode == "instrumental" else "inst_prob (1 - vocal_p)"
    line_inst_prob, = ax.plot(times, inst_probs, label=label_prob)
    line_inst_true, = ax.plot(times, inst_label, label="inst_label")
    line_inst_pred, = ax.plot(times, inst_pred, label="inst_pred")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instrumental Probability / Binary")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # 横坐标按 mm:ss.mmm 展示
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, pos: format_time_ms(t)))
    ax.format_coord = lambda x, y: f"t={format_time_ms(x)}, value={y:.3f}"

    txt = ax.text(0.01, 1.02, f"Precision: {p:.3f}  |  Recall: {r:.3f}  |  F1: {f1:.3f}", transform=ax.transAxes)

    # 阈值滑块
    # 阈值滑块（单阈值）
    ax_th = plt.axes([0.15, 0.12, 0.7, 0.03])
    s_th = Slider(ax_th, "threshold", 0.0, 1.0, valinit=0.5, valstep=0.01)

    # 迟滞（th_on/th_off）初始值与开关（若权重包含则读取），并依据 eval_postproc 默认启用
    on_init = config.get("eval_hyst_on", None)
    off_init = config.get("eval_hyst_off", None)
    postproc = str(config.get("eval_postproc", "threshold"))
    use_hyst = False
    if postproc == "hysteresis" and isinstance(on_init, (int, float)) and isinstance(off_init, (int, float)) and (float(on_init) > float(off_init)):
        use_hyst = True
    th_on0 = float(on_init) if isinstance(on_init, (int, float)) else 0.5
    th_off0 = float(off_init) if isinstance(off_init, (int, float)) else max(0.0, 0.5 - 0.2)

    # 迟滞滑块与开关
    ax_on = plt.axes([0.15, 0.07, 0.32, 0.03])
    ax_off = plt.axes([0.53, 0.07, 0.32, 0.03])
    s_on = Slider(ax_on, "th_on", 0.0, 1.0, valinit=th_on0, valstep=0.01)
    s_off = Slider(ax_off, "th_off", 0.0, 1.0, valinit=th_off0, valstep=0.01)
    ax_chk = plt.axes([0.88, 0.07, 0.1, 0.08])
    chk = CheckButtons(ax_chk, ["hysteresis"], [use_hyst])

    def apply_hysteresis(seq_probs: np.ndarray, th_on: float, th_off: float) -> np.ndarray:
        pred = np.zeros_like(seq_probs, dtype=np.float32)
        state = 0.0
        for i, pval in enumerate(seq_probs):
            if state == 0.0 and pval >= th_on:
                state = 1.0
            elif state == 1.0 and pval < th_off:
                state = 0.0
            pred[i] = state
        return pred

    # 统一更新函数：根据当前模式（单阈值或迟滞）更新曲线与指标
    current_pred = inst_pred.copy()
    def recompute_and_update():
        nonlocal current_pred
        if use_hyst and (s_on.val > s_off.val):
            pred = apply_hysteresis(inst_probs, float(s_on.val), float(s_off.val))
            # 曲线标签提示迟滞已启用
            line_inst_pred.set_label("inst_pred_hyst")
        else:
            pred = (inst_probs >= float(s_th.val)).astype(np.float32)
            line_inst_pred.set_label("inst_pred")
        current_pred = pred.astype(np.float32)
        line_inst_pred.set_ydata(current_pred)
        p2, r2, f2 = metrics(current_pred, inst_label)
        txt.set_text(f"Precision: {p2:.3f}  |  Recall: {r2:.3f}  |  F1: {f2:.3f}")
        ax.legend(loc="upper right")
        fig.canvas.draw_idle()

    def on_th_change(val):
        # 单阈值模式下生效；迟滞模式下忽略
        recompute_and_update()

    def on_on_change(val):
        recompute_and_update()

    def on_off_change(val):
        recompute_and_update()

    def on_chk_clicked(label):
        nonlocal use_hyst
        use_hyst = not use_hyst
        recompute_and_update()

    s_th.on_changed(on_th_change)
    s_on.on_changed(on_on_change)
    s_off.on_changed(on_off_change)
    chk.on_clicked(on_chk_clicked)

    # 图表下方显示音乐名（居中），替代顶部标题
    title_text = fig.text(0.5, 0.25, f"{stem}", ha="center", va="top")

    # 初始化一次
    recompute_and_update()

    # 交互：选择音频（Prev/Next/Browse/Stem 输入）
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
        # 重新加载特征
        X = load_mel(audio_path)
        T = X.shape[0]
        times = np.arange(T) * HOP_SEC
        # 重新加载标签
        label_json_path2 = labels_root / f"{stem}.json"
        labels = None
        if label_json_path2.exists():
            try:
                labels = json.loads(label_json_path2.read_text(encoding="utf-8"))
            except Exception:
                labels = None
        y_true = labels_to_frame_targets(labels or [], T)
        has_label = labels is not None
        # 推理并解释为间奏概率
        probs2 = run_infer(model, X)
        inst_probs = probs2 if mode == "instrumental" else (1.0 - probs2)
        inst_label = (1.0 - y_true) if has_label else np.zeros((T,), dtype=np.float32)
        # 更新曲线与坐标范围
        line_inst_prob.set_xdata(times)
        line_inst_prob.set_ydata(inst_probs)
        line_inst_true.set_xdata(times)
        line_inst_true.set_ydata(inst_label)
        line_inst_pred.set_xdata(times)
        ax.set_xlim(times[0], times[-1] if len(times) > 0 else 1.0)
        title_text.set_text(f"{stem}")
        ann.set_visible(False)
        # 依当前模式/阈值重新计算预测与指标
        recompute_and_update()

    # Prev / Next 按钮
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

    # 鼠标移动显示时间与预测值（最近帧）
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
        # 使用当前模式下的二值结果
        bin_pred = int(current_pred[idx])
        ann.xy = (t, val)
        ann.set_text(f"{format_time_ms(t)} | pred={val:.3f} | label={lab} | bin={bin_pred}")
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # 顶部标题移除，使用底部标题
    plt.show()


# 设置系统常见字体，避免使用 DejaVu Sans（按可用字体自动回退）
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei"
]
mpl.rcParams["axes.unicode_minus"] = False

if __name__ == "__main__":
    main()