from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa 未安装，请在环境中安装后再运行训练脚本。") from e


# 配置参数
SR = 22050
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
HOP_SEC = HOP_LENGTH / SR

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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mel(audio_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )  # [n_mels, T]
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm.T.astype(np.float32)  # [T, n_mels]


def labels_to_frame_targets(labels: List[Dict], T: int) -> np.ndarray:
    y = np.zeros((T,), dtype=np.float32)
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


@dataclass
class Sample:
    features: np.ndarray  # [T, n_mels]
    targets: np.ndarray   # [T]
    name: str


class SVDDataset(Dataset):
    def __init__(self, labels_root: Path, music_root: Path, instrumental: bool = False):
        self.samples: List[Sample] = []
        perfect_root = labels_root / "perfect"
        for p in perfect_root.glob("*.json"):
            stem = p.stem
            # 查找音频文件
            audio_path: Optional[Path] = None
            for ext in AUDIO_EXTS_ORDER:
                q = music_root / f"{stem}{ext}"
                if q.exists():
                    audio_path = q
                    break
            if audio_path is None:
                continue
            try:
                labels: List[Dict] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                labels = []
            X = load_mel(audio_path)
            y = labels_to_frame_targets(labels, T=X.shape[0])
            if instrumental:
                y = 1.0 - y
            self.samples.append(Sample(features=X, targets=y.astype(np.float32), name=stem))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return torch.from_numpy(s.features.copy()), torch.from_numpy(s.targets.copy()), s.name


class LazySVDDataset(Dataset):
    def __init__(self, labels_root: Path, music_root: Path, cache_dir: Optional[Path] = None, instrumental: bool = False):
        self.items: List[Dict] = []
        self.cache_dir = cache_dir
        self.instrumental = instrumental
        perfect_root = labels_root / "perfect"
        for p in perfect_root.glob("*.json"):
            stem = p.stem
            audio_path: Optional[Path] = None
            for ext in AUDIO_EXTS_ORDER:
                q = music_root / f"{stem}{ext}"
                if q.exists():
                    audio_path = q
                    break
            if audio_path is None:
                continue
            try:
                labels = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                labels = []
            self.items.append({"name": stem, "audio_path": audio_path, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        stem = it["name"]
        audio_path: Path = it["audio_path"]
        labels: List[Dict] = it["labels"]

        X: np.ndarray
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{stem}.npy"
            if cache_file.exists():
                X = np.load(cache_file, mmap_mode="r")
            else:
                X = load_mel(audio_path)
                try:
                    np.save(cache_file, X)
                except Exception:
                    pass
        else:
            X = load_mel(audio_path)

        y = labels_to_frame_targets(labels, T=X.shape[0])
        if self.instrumental:
            y = 1.0 - y
        return torch.from_numpy(X.copy()), torch.from_numpy(y.copy()), stem


def collate_pad(batch):
    # batch: List[(X[T,F], y[T], name)]
    names = [b[2] for b in batch]
    lengths = [b[0].shape[0] for b in batch]
    F = batch[0][0].shape[1]
    T_max = max(lengths)
    X_pad = torch.zeros((len(batch), T_max, F), dtype=torch.float32)
    y_pad = torch.zeros((len(batch), T_max), dtype=torch.float32)
    mask = torch.zeros((len(batch), T_max), dtype=torch.float32)
    for i, (X, y, _) in enumerate(batch):
        t = X.shape[0]
        X_pad[i, :t, :] = X
        y_pad[i, :t] = y
        mask[i, :t] = 1.0
    return X_pad, y_pad, mask, names, lengths


class CNNLSTM(nn.Module):
    """在频率维上做 1D CNN，之后接单层 LSTM 做时序分类。"""
    def __init__(self, input_dim: int, cnn_channels: int = 64, hidden: int = 128):
        super().__init__()
        # 对每个时间帧的频谱列做 1D 卷积提取局部频率模式
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # 汇聚到频率通道的全局特征
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, X: torch.Tensor):
        # X: [B, T, F]
        B, T, F = X.shape
        x = X.reshape(B * T, 1, F)          # [B*T, 1, F]
        x = self.cnn(x).squeeze(-1)         # [B*T, C]
        x = x.reshape(B, T, -1)             # [B, T, C]
        out, _ = self.lstm(x)               # [B, T, H]
        logits = self.head(out).squeeze(-1) # [B, T]
        return logits


def train_one_epoch(model, loader, optimizer, device, pos_weight: Optional[float] = None):
    model.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=(torch.tensor([pos_weight], dtype=torch.float32, device=device) if pos_weight is not None else None),
    )
    for X, y, mask, _, _ in tqdm(loader, desc="train", leave=False):
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits = model(X)
        loss = bce(logits, y)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def collect_val_outputs(model, loader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    probs_flat: List[np.ndarray] = []
    y_flat: List[np.ndarray] = []
    probs_seqs: List[np.ndarray] = []
    y_seqs: List[np.ndarray] = []
    for X, y, mask, _, _ in tqdm(loader, desc="collect", leave=False):
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)
        logits = model(X)
        loss = bce(logits, y)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        m = (mask == 1)
        probs_flat.append(probs[m].detach().cpu().numpy())
        y_flat.append(y[m].detach().cpu().numpy())
        B = X.shape[0]
        for i in range(B):
            t_len = int(mask[i].sum().item())
            probs_seqs.append(probs[i, :t_len].detach().cpu().numpy())
            y_seqs.append(y[i, :t_len].detach().cpu().numpy())
    probs_flat_all = np.concatenate(probs_flat) if probs_flat else np.zeros((0,), dtype=np.float32)
    y_flat_all = np.concatenate(y_flat) if y_flat else np.zeros((0,), dtype=np.float32)
    return probs_flat_all, y_flat_all, probs_seqs, y_seqs, (total_loss / max(1, len(loader)))


def compute_metrics_from_arrays(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = tp / (tp + fp + 1e-8) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def apply_hysteresis_seq(prob_seq: np.ndarray, th_on: float, th_off: float) -> np.ndarray:
    pred = np.zeros_like(prob_seq, dtype=np.float32)
    state = 0.0
    for i, pval in enumerate(prob_seq):
        if state == 0.0 and pval >= th_on:
            state = 1.0
        elif state == 1.0 and pval < th_off:
            state = 0.0
        pred[i] = state
    return pred


def compute_hysteresis_metrics(probs_seqs: List[np.ndarray], y_seqs: List[np.ndarray], th_on: float, th_off: float) -> Dict[str, float]:
    tp = fp = fn = 0
    for prob_seq, y_seq in zip(probs_seqs, y_seqs):
        pred_seq = apply_hysteresis_seq(prob_seq, th_on, th_off)
        tp += int(((pred_seq == 1) & (y_seq == 1)).sum())
        fp += int(((pred_seq == 1) & (y_seq == 0)).sum())
        fn += int(((pred_seq == 0) & (y_seq == 1)).sum())
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def grid_search_hysteresis(probs_seqs: List[np.ndarray], y_seqs: List[np.ndarray], th_on_grid: List[float], th_off_grid: List[float]) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    best_on: Optional[float] = None
    best_off: Optional[float] = None
    best_metrics: Dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for th_on in th_on_grid:
        for th_off in th_off_grid:
            if th_on <= th_off:
                continue
            m = compute_hysteresis_metrics(probs_seqs, y_seqs, th_on, th_off)
            if m["f1"] > best_metrics["f1"]:
                best_metrics = m
                best_on, best_off = th_on, th_off
    return best_on, best_off, best_metrics


def compute_pos_weight_for_subset(ds: Dataset, indices: List[int], instrumental: bool) -> float:
    pos = 0.0
    total = 0.0
    if isinstance(ds, LazySVDDataset):
        for i in indices:
            it = ds.items[i]
            audio_path: Path = it["audio_path"]
            labels: List[Dict] = it["labels"]
            try:
                duration = librosa.get_duration(filename=str(audio_path))
            except Exception:
                max_end = 0.0
                for item in labels:
                    end_m, end_s = item["end"][0], item["end"][1]
                    max_end = max(max_end, end_m * 60.0 + float(end_s))
                duration = max_end
            T = int(math.ceil(duration / HOP_SEC))
            y = labels_to_frame_targets(labels, T=T)
            if instrumental:
                y = 1.0 - y
            pos += float(y.sum())
            total += float(T)
    else:
        for i in indices:
            s = ds.samples[i]
            y = s.targets
            pos += float(y.sum())
            total += float(y.size)
    neg = max(0.0, total - pos)
    pos = max(pos, 1.0)
    raw_pw = (neg + 1e-8) / (pos + 1e-8)
    pos_weight = float(np.clip(raw_pw, 1.0, 50.0))
    return pos_weight


def run_training(
    labels_root: Path,
    music_root: Path,
    epochs: int = 5,
    batch_size: int = 2,
    lazy: bool = True,
    cache_dir: Optional[Path] = Path("./features_cache"),
    mode: str = "vocal",
    save_path: Optional[Path] = None,
    eval_hyst_on: Optional[float] = None,
    eval_hyst_off: Optional[float] = None,
):
    set_seed(42)
    assert mode in ("vocal", "instrumental"), "mode 仅支持 'vocal' 或 'instrumental'"
    instrumental = (mode == "instrumental")
    ds = LazySVDDataset(labels_root=labels_root, music_root=music_root, cache_dir=cache_dir, instrumental=instrumental) if lazy else SVDDataset(labels_root=labels_root, music_root=music_root, instrumental=instrumental)
    if len(ds) == 0:
        raise RuntimeError("数据集为空：请确认 labels/perfect 与 music 目录下存在同名有效数据。")

    indices = list(range(len(ds)))
    random.shuffle(indices)
    n_train = int(len(ds) * 0.9)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_subset = torch.utils.data.Subset(ds, train_idx)
    val_subset = torch.utils.data.Subset(ds, val_idx)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pad,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pad,
        pin_memory=use_cuda,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM(input_dim=N_MELS, cnn_channels=64, hidden=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    pos_weight = compute_pos_weight_for_subset(ds, train_idx, instrumental)
    print(f"pos_weight={pos_weight:.2f} (mode={mode})")

    best_f1 = -1.0
    if save_path is None:
        default_name = "cnn_lstm_inst.pt" if instrumental else "cnn_lstm.pt"
        best_path = Path(".") / default_name
    else:
        best_path = Path(save_path)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight)

        probs_flat, y_flat, probs_seqs, y_seqs, val_loss = collect_val_outputs(model, val_loader, device)
        preds_05 = (probs_flat >= 0.5).astype(np.float32)
        m05 = compute_metrics_from_arrays(preds_05, y_flat)

        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} f1@0.5={m05['f1']:.4f} precision@0.5={m05['precision']:.4f} recall@0.5={m05['recall']:.4f}")
        if (eval_hyst_on is not None) and (eval_hyst_off is not None) and (eval_hyst_on > eval_hyst_off):
            mh = compute_hysteresis_metrics(probs_seqs, y_seqs, float(eval_hyst_on), float(eval_hyst_off))
            print(f"  hyst(on={float(eval_hyst_on):.2f},off={float(eval_hyst_off):.2f}) f1={mh['f1']:.4f} precision={mh['precision']:.4f} recall={mh['recall']:.4f}")

        th_on_grid = [round(t, 2) for t in np.linspace(0.4, 0.9, num=11).tolist()]
        th_off_grid = [round(t, 2) for t in np.linspace(0.1, 0.6, num=11).tolist()]
        best_on, best_off, best_hyst = grid_search_hysteresis(probs_seqs, y_seqs, th_on_grid, th_off_grid)

        print(f"  best_hyst on={best_on} off={best_off} f1={best_hyst['f1']:.4f} precision={best_hyst['precision']:.4f} recall={best_hyst['recall']:.4f}")

        # 仅迟滞用于选优与保存
        if best_hyst["f1"] > best_f1:
            best_f1 = best_hyst["f1"]
            cfg = {
                "sr": SR,
                "n_mels": N_MELS,
                "hop_length": HOP_LENGTH,
                "mode": mode,
                "eval_postproc": "hysteresis",
                "eval_threshold": None,
                "eval_hyst_on": best_on,
                "eval_hyst_off": best_off,
            }
            torch.save({"model_state": model.state_dict(), "config": cfg}, best_path)
            print(f"saved best to {best_path} (f1={best_f1:.4f}, postproc=hysteresis, on={best_on}, off={best_off})")

    return {"best_f1": best_f1, "ckpt": str(best_path)}


def main():
    parser = argparse.ArgumentParser(description="训练 CNN+LSTM（支持人声或间奏模式）")
    parser.add_argument("--labels-root", default=str(Path(".") / "labels"))
    parser.add_argument("--music-root", default=r"D:\meliris\music")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lazy", action="store_true", default=True)
    parser.add_argument("--no-lazy", dest="lazy", action="store_false")
    parser.set_defaults(lazy=True)
    parser.add_argument("--mode", choices=["vocal", "instrumental"], default="instrumental")
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--eval-hyst-on", type=float, default=0.6)
    parser.add_argument("--eval-hyst-off", type=float, default=0.3)
    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    music_root = Path(args.music_root)
    stats = run_training(
        labels_root=labels_root,
        music_root=music_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lazy=args.lazy,
        cache_dir=Path("./features_cache"),
        mode=args.mode,
        save_path=(Path(args.save_path) if args.save_path else None),
        eval_hyst_on=args.eval_hyst_on,
        eval_hyst_off=args.eval_hyst_off,
    )
    print("training done:", stats)


if __name__ == "__main__":
    main()