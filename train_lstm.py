from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
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


def find_audio_by_stem(music_root: Path, stem: str) -> Optional[Path]:
    for ext in AUDIO_EXTS_ORDER:
        p = music_root / f"{stem}{ext}"
        if p.exists():
            return p
    # 兜底：遍历查找（若命名包含特殊字符导致 stem 不一致）
    for p in music_root.glob("*"):
        if p.is_file() and p.stem == stem:
            return p
    return None


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
    # 归一化到 [0,1]
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
    def __init__(self, labels_root: Path, music_root: Path):
        self.samples: List[Sample] = []
        for json_path in (labels_root / "perfect").glob("*.json"):
            stem = json_path.stem
            # 读取标签
            try:
                labels = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not labels:
                continue
            # 匹配音频
            audio_path = find_audio_by_stem(music_root, stem)
            if audio_path is None:
                continue
            # 提取特征
            try:
                X = load_mel(audio_path)  # [T, n_mels]
            except Exception:
                continue
            # 生成帧级目标
            y = labels_to_frame_targets(labels, T=X.shape[0])
            self.samples.append(Sample(features=X, targets=y, name=stem))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        X = torch.from_numpy(s.features)  # [T, F]
        y = torch.from_numpy(s.targets)   # [T]
        return X, y, s.name


class LazySVDDataset(Dataset):
    """
    惰性加载版本：不在构造时把所有歌曲的特征加载到内存；
    仅保存 (audio_path, labels) 元数据，在 __getitem__ 时按需计算或从磁盘缓存读取。
    更省内存，适合几十/上百首歌曲的训练。
    """
    def __init__(self, labels_root: Path, music_root: Path, cache_dir: Optional[Path] = None):
        self.items: List[Dict] = []
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        for json_path in (labels_root / "perfect").glob("*.json"):
            stem = json_path.stem
            # 读取标签
            try:
                labels = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not labels:
                continue
            # 匹配音频
            audio_path = find_audio_by_stem(music_root, stem)
            if audio_path is None:
                continue
            self.items.append({
                "name": stem,
                "audio_path": audio_path,
                "labels": labels,
            })

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
                # 使用内存映射读取，避免一次性读入巨大数组
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
        return torch.from_numpy(X), torch.from_numpy(y), stem


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


class LSTMSVD(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor):
        # X: [B, T, F]
        out, _ = self.lstm(X)
        logits = self.head(out).squeeze(-1)  # [B, T]
        return logits


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss(reduction="none")
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
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    tp = fp = fn = 0
    for X, y, mask, _, _ in tqdm(loader, desc="eval", leave=False):
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)
        logits = model(X)
        loss = bce(logits, y)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        tp += ((preds == 1) & (y == 1) & (mask == 1)).sum().item()
        fp += ((preds == 1) & (y == 0) & (mask == 1)).sum().item()
        fn += ((preds == 0) & (y == 1) & (mask == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "loss": total_loss / max(1, len(loader)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_training(
    labels_root: Path,
    music_root: Path,
    epochs: int = 5,
    batch_size: int = 2,
    lazy: bool = True,
    cache_dir: Optional[Path] = Path("./features_cache"),
):
    set_seed(42)
    # 更省内存的默认：使用惰性加载数据集
    ds = LazySVDDataset(labels_root=labels_root, music_root=music_root, cache_dir=cache_dir) if lazy else SVDDataset(labels_root=labels_root, music_root=music_root)
    if len(ds) == 0:
        raise RuntimeError("数据集为空：请确认 labels/perfect 与 music 目录下存在同名有效数据。")

    # 简单划分 9:1
    indices = list(range(len(ds)))
    random.shuffle(indices)
    n_train = int(len(ds) * 0.9)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_subset = torch.utils.data.Subset(ds, train_idx)
    val_subset = torch.utils.data.Subset(ds, val_idx)

    # num_workers=0 更省内存（Windows 下多进程会复制主进程状态）；
    # 若使用 GPU，可开启 pin_memory=True。
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

    input_dim = N_MELS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSVD(input_dim=input_dim, hidden=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1 = -1.0
    best_path = Path(".") / "lstm_svd.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({"model_state": model.state_dict(), "config": {"sr": SR, "n_mels": N_MELS, "hop_length": HOP_LENGTH}}, best_path)
            print(f"saved best to {best_path} (f1={best_f1:.4f})")

    return {"best_f1": best_f1, "ckpt": str(best_path)}


def main():
    labels_root = Path(".") / "labels"
    music_root = Path(r"D:\meliris\music")
    stats = run_training(labels_root=labels_root, music_root=music_root, epochs=5, batch_size=2, lazy=True, cache_dir=Path("./features_cache"))
    print("training done:", stats)


if __name__ == "__main__":
    main()