from __future__ import annotations

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from pathlib import Path

from common import HOP_SEC, load_features, labels_to_frame_targets, apply_hysteresis_seq, compute_metrics_from_arrays


class CNNLSTM(nn.Module):
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
            bidirectional=bidirectional
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, X: torch.Tensor):
        B, T, F = X.shape
        x = X.reshape(B * T, 1, F)
        x = self.cnn(x).squeeze(-1)
        x = x.reshape(B, T, -1)
        out, _ = self.lstm(x)
        logits = self.head(out).squeeze(-1)
        return logits


def load_model(ckpt_path: Path):
    ck = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ck.get("config", {})
    bilstm = bool(cfg.get("bilstm"))
    feat_dim = int(cfg.get("feature_dim", 0))
    state = ck.get("model_state", ck)
    model = CNNLSTM(input_dim=feat_dim or 256, cnn_channels=64, hidden=128, bidirectional=bilstm)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def run_infer(model: nn.Module, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(X.astype(np.float32)).unsqueeze(0)
        probs = torch.sigmoid(model(t)).squeeze(0).detach().cpu().numpy()
        return probs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="CNN+LSTM 推理可视化（多特征输入）")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--labels-root", type=str, default=str(Path(".") / "labels_qrc"))
    ap.add_argument("--mode", choices=["vocal", "instrumental"], default="instrumental")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    audio_path = Path(args.audio)
    labels_root = Path(args.labels_root)
    mode = str(args.mode)

    model, cfg = load_model(ckpt_path)
    X = load_features(audio_path)
    T = X.shape[0]
    times = np.arange(T) * HOP_SEC
    probs = run_infer(model, X)
    inst_probs = probs if mode == "instrumental" else (1.0 - probs)

    label_json_path = labels_root / (audio_path.stem + ".json")
    perfect_json_path = labels_root / "perfect" / (audio_path.stem + ".json")
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

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, inst_probs, label="prob", color="#268bd2")
    ax.plot(times, inst_label, label="true", color="#2aa198", alpha=0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title(audio_path.stem)

    ax_th = plt.axes([0.15, 0.07, 0.32, 0.03])
    ax_chk = plt.axes([0.53, 0.07, 0.20, 0.10])
    s_th = Slider(ax_th, "threshold", 0.0, 1.0, valinit=0.5, valstep=0.01)
    chk = CheckButtons(ax_chk, ["hysteresis"], [False])
    line_pred, = ax.plot(times, (inst_probs >= 0.5).astype(np.float32), label="pred", color="#b58900")

    def recompute(val=None):
        use_hyst = chk.get_status()[0]
        th = float(s_th.val)
        pred = apply_hysteresis_seq(inst_probs, th, max(0.0, th - 0.2)) if use_hyst else (inst_probs >= th).astype(np.float32)
        line_pred.set_ydata(pred.astype(np.float32))
        m2 = compute_metrics_from_arrays(pred.astype(np.float32), inst_label.astype(np.float32))
        ax.legend(loc="lower left")
        fig.canvas.draw_idle()

    s_th.on_changed(recompute)
    chk.on_clicked(lambda _: recompute())
    recompute()
    plt.show()


if __name__ == "__main__":
    main()
