from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button, TextBox
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

from common.config import SR, N_MELS, HOP_LENGTH, HOP_SEC
from common.data import load_mel, labels_to_frame_targets
from common.postproc import compute_metrics_from_arrays, apply_hysteresis_seq


def format_time_ms(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:06.3f}"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        T = x.shape[1]
        device = x.device
        pe = torch.zeros(T, self.d_model, device=device)
        position = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x = x + pe.unsqueeze(0)
        return self.dropout(x)


class CNNTransformer(nn.Module):
    def __init__(self, input_dim: int, cnn_channels: int = 128, nhead: int = 4, num_layers: int = 2, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.pos = PositionalEncoding(d_model=cnn_channels, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=cnn_channels, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(cnn_channels, 1)

    def forward(self, X: torch.Tensor):
        B, T, F = X.shape
        x = X.reshape(B * T, 1, F)
        x = self.cnn(x).squeeze(-1)
        x = x.reshape(B, T, -1)
        x = self.pos(x)
        out = self.encoder(x)
        logits = self.head(out).squeeze(-1)
        return logits

@torch.no_grad()
def predict_chunked(model: CNNTransformer, X_t: torch.Tensor, max_seq_len: Optional[int] = None) -> np.ndarray:
    """在推理阶段对超长序列分段前向，避免注意力 OOM。

    - X_t: [1, T, F]
    - 返回: [T] 概率数组
    """
    if (max_seq_len is None) or (X_t.shape[1] <= max_seq_len):
        logits = model(X_t)
        return torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)
    T = X_t.shape[1]
    outs: list[torch.Tensor] = []
    for st in range(0, T, int(max_seq_len)):
        ed = min(st + int(max_seq_len), T)
        logits_chunk = model(X_t[:, st:ed, :])
        outs.append(logits_chunk)
    logits = torch.cat(outs, dim=1)
    return torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)


def load_model(ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model = CNNTransformer(input_dim=N_MELS, cnn_channels=128, nhead=4, num_layers=2, ff_dim=256, dropout=0.0)
    try:
        model.load_state_dict(state, strict=False)
    except RuntimeError:
        # 允许 head 或层命名上的轻微差异
        model.load_state_dict(state, strict=False)
    model.eval()
    config = ckpt.get("config", {})
    return model, config


def find_candidates(music_root: Path, labels_root: Path):
    perfect_root = labels_root / "perfect"
    candidates = []
    for p in music_root.glob("*"):
        if p.is_file():
            stem = p.stem
            if not (perfect_root / f"{stem}.json").exists():
                candidates.append(p)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="CNN+Transformer 推理可视化：不在 labels/perfect 下的音频，与 labels 对照")
    parser.add_argument("--music-root", default=r"D:\meliris\music", help="音乐目录")
    parser.add_argument("--labels-root", default=str(Path(__file__).resolve().parent.parent / "labels"), help="标签目录")
    parser.add_argument("--stem", default=None, help="指定文件名 stem（不含扩展名）")
    parser.add_argument("--ckpt", default=str(Path(__file__).resolve().parent.parent / "cnn_transformer_inst.pt"), help="模型权重路径")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Transformer 前向分段长度，避免 OOM")
    args = parser.parse_args()

    music_root = Path(args.music_root)
    labels_root = Path(args.labels_root)
    ckpt_path = Path(args.ckpt)

    candidates = find_candidates(music_root, labels_root)
    if not candidates:
        print("未找到候选音频（不在 labels/perfect 下的同名文件）。")
        return

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

    if not ckpt_path.exists():
        print(f"未找到模型权重：{ckpt_path}")
        return
    model, config = load_model(ckpt_path)
    X = load_mel(audio_path)
    time_stride = int(config.get("time_stride", 1))
    if time_stride > 1:
        X_t = torch.from_numpy(X).unsqueeze(0)  # [1, T, F]
        Xd = F.avg_pool1d(X_t.permute(0, 2, 1), kernel_size=time_stride, stride=time_stride).permute(0, 2, 1).squeeze(0)
        X = Xd.cpu().numpy().astype(np.float32)
    T = X.shape[0]
    times = np.arange(T) * (HOP_SEC * time_stride)

    label_json_path = labels_root / f"{stem}.json"
    labels = None
    if label_json_path.exists():
        try:
            labels = json.loads(label_json_path.read_text(encoding="utf-8"))
        except Exception:
            labels = None
    y_true = labels_to_frame_targets(labels or [], int(np.ceil(T * time_stride))) if labels is not None else np.zeros((int(np.ceil(T * time_stride)),), dtype=np.float32)
    if time_stride > 1:
        yt = torch.from_numpy(y_true).unsqueeze(0).unsqueeze(0)
        yd = F.max_pool1d(yt, kernel_size=time_stride, stride=time_stride).squeeze(0).squeeze(0)
        y_true = yd.cpu().numpy().astype(np.float32)
        y_true = y_true[:T]
    has_label = labels is not None

    with torch.no_grad():
        probs = predict_chunked(model, torch.from_numpy(X).unsqueeze(0), max_seq_len=int(args.max_seq_len))

    mode = str(config.get("mode", "vocal"))
    inst_probs = probs if mode == "instrumental" else (1.0 - probs)
    inst_label = (1.0 - y_true) if has_label else np.zeros((T,), dtype=np.float32)
    inst_pred = (inst_probs >= 0.5).astype(np.float32)
    m0 = compute_metrics_from_arrays(inst_pred, inst_label)
    p, r, f1 = m0["precision"], m0["recall"], m0["f1"]

    fig, ax = plt.subplots(figsize=(12, 5))
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

    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, pos: format_time_ms(t)))
    ax.format_coord = lambda x, y: f"t={format_time_ms(x)}, value={y:.3f}"

    txt = ax.text(0.01, 1.02, f"Precision: {p:.3f}  |  Recall: {r:.3f}  |  F1: {f1:.3f}", transform=ax.transAxes)

    ax_th = plt.axes([0.15, 0.12, 0.7, 0.03])
    s_th = Slider(ax_th, "threshold", 0.0, 1.0, valinit=0.5, valstep=0.01)

    on_init = config.get("eval_hyst_on", None)
    off_init = config.get("eval_hyst_off", None)
    postproc = str(config.get("eval_postproc", "threshold"))
    use_hyst = False
    if postproc == "hysteresis" and isinstance(on_init, (int, float)) and isinstance(off_init, (int, float)) and (float(on_init) > float(off_init)):
        use_hyst = True
    th_on0 = float(on_init) if isinstance(on_init, (int, float)) else 0.5
    th_off0 = float(off_init) if isinstance(off_init, (int, float)) else max(0.0, 0.5 - 0.2)

    ax_on = plt.axes([0.15, 0.07, 0.32, 0.03])
    ax_off = plt.axes([0.53, 0.07, 0.32, 0.03])
    s_on = Slider(ax_on, "th_on", 0.0, 1.0, valinit=th_on0, valstep=0.01)
    s_off = Slider(ax_off, "th_off", 0.0, 1.0, valinit=th_off0, valstep=0.01)
    ax_chk = plt.axes([0.88, 0.07, 0.1, 0.08])
    chk = CheckButtons(ax_chk, ["hysteresis"], [use_hyst])

    current_pred = inst_pred.copy()
    def recompute_and_update():
        nonlocal current_pred
        if use_hyst and (s_on.val > s_off.val):
            pred = apply_hysteresis_seq(inst_probs, float(s_on.val), float(s_off.val))
            line_inst_pred.set_label("inst_pred_hyst")
        else:
            pred = (inst_probs >= float(s_th.val)).astype(np.float32)
            line_inst_pred.set_label("inst_pred")
        current_pred = pred.astype(np.float32)
        line_inst_pred.set_ydata(current_pred)
        m2 = compute_metrics_from_arrays(current_pred, inst_label)
        txt.set_text(f"Precision: {m2['precision']:.3f}  |  Recall: {m2['recall']:.3f}  |  F1: {m2['f1']:.3f}")
        ax.legend(loc="upper right")
        fig.canvas.draw_idle()

    def on_th_change(val):
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

    title_text = fig.text(0.5, 0.25, f"{stem}", ha="center", va="top")
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
        X = load_mel(audio_path)
        if time_stride > 1:
            X_t = torch.from_numpy(X).unsqueeze(0)
            Xd = F.avg_pool1d(X_t.permute(0, 2, 1), kernel_size=time_stride, stride=time_stride).permute(0, 2, 1).squeeze(0)
            X = Xd.cpu().numpy().astype(np.float32)
        T = X.shape[0]
        times = np.arange(T) * (HOP_SEC * time_stride)
        label_json_path2 = labels_root / f"{stem}.json"
        labels = None
        if label_json_path2.exists():
            try:
                labels = json.loads(label_json_path2.read_text(encoding="utf-8"))
            except Exception:
                labels = None
        y_true = labels_to_frame_targets(labels or [], int(np.ceil(T * time_stride))) if labels is not None else np.zeros((int(np.ceil(T * time_stride)),), dtype=np.float32)
        if time_stride > 1:
            yt2 = torch.from_numpy(y_true).unsqueeze(0).unsqueeze(0)
            yd2 = F.max_pool1d(yt2, kernel_size=time_stride, stride=time_stride).squeeze(0).squeeze(0)
            y_true = yd2.cpu().numpy().astype(np.float32)
            y_true = y_true[:T]
        has_label = labels is not None
        with torch.no_grad():
            probs2 = predict_chunked(model, torch.from_numpy(X).unsqueeze(0), max_seq_len=int(args.max_seq_len))
        inst_probs = probs2 if mode == "instrumental" else (1.0 - probs2)
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
        idx = int(np.clip(round(x / (HOP_SEC * time_stride)), 0, T - 1))
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