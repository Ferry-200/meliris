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
    def __init__(self, labels_root: Path, music_root: Path, instrumental: bool = False):
        self.samples: List[Sample] = []
        self.instrumental = instrumental
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
            if self.instrumental:
                y = 1.0 - y
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
    def __init__(self, labels_root: Path, music_root: Path, cache_dir: Optional[Path] = None, instrumental: bool = False):
        self.items: List[Dict] = []
        self.cache_dir = cache_dir
        self.instrumental = instrumental
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


def train_one_epoch(model, loader, optimizer, device, pos_weight: Optional[float] = None):
    model.train()
    total_loss = 0.0
    # 在类别不平衡场景下对正类加权
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
def evaluate(model, loader, device, threshold: float = 0.5, hysteresis: Optional[Tuple[float, float]] = None):
    """评估：支持普通阈值与迟滞开关（th_on, th_off）。

    当提供 hysteresis=(th_on, th_off) 时，使用迟滞规则进行二值化；否则使用单阈值。
    """
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    tp = fp = fn = 0

    def apply_hysteresis(seq_probs: np.ndarray, th_on: float, th_off: float) -> np.ndarray:
        pred = np.zeros_like(seq_probs, dtype=np.float32)
        state = 0.0
        for i, p in enumerate(seq_probs):
            if state == 0.0 and p >= th_on:
                state = 1.0
            elif state == 1.0 and p < th_off:
                state = 0.0
            pred[i] = state
        return pred

    use_hyst = hysteresis is not None
    th_on, th_off = (hysteresis if use_hyst else (threshold, threshold))
    if use_hyst:
        assert th_on > th_off, "迟滞阈值需满足 th_on > th_off"

    for X, y, mask, _, _ in tqdm(loader, desc="eval", leave=False):
        X = X.to(device)
        y = y.to(device)
        mask = mask.to(device)
        logits = model(X)
        loss = bce(logits, y)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        if not use_hyst:
            preds = (probs >= threshold).float()
            tp += ((preds == 1) & (y == 1) & (mask == 1)).sum().item()
            fp += ((preds == 1) & (y == 0) & (mask == 1)).sum().item()
            fn += ((preds == 0) & (y == 1) & (mask == 1)).sum().item()
        else:
            # 对每个样本按长度逐序应用迟滞逻辑
            B = X.shape[0]
            for i in range(B):
                t_len = int(mask[i].sum().item())
                prob_seq = probs[i, :t_len].detach().cpu().numpy()
                y_seq = y[i, :t_len].detach().cpu().numpy()
                pred_seq = apply_hysteresis(prob_seq, th_on, th_off)
                tp += int(((pred_seq == 1) & (y_seq == 1)).sum())
                fp += int(((pred_seq == 1) & (y_seq == 0)).sum())
                fn += int(((pred_seq == 0) & (y_seq == 1)).sum())

    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    return {
        "loss": total_loss / max(1, len(loader)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def collect_val_outputs(model, loader, device):
    """收集验证集的概率与标签（按样本序列与展平两种形式），用于阈值搜索。"""
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
        # 展平（按 mask）
        m = (mask == 1)
        probs_flat.append(probs[m].detach().cpu().numpy())
        y_flat.append(y[m].detach().cpu().numpy())
        # 序列形式
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
    """对单个概率序列应用迟滞(th_on/th_off)，返回二值序列。"""
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
    """对序列列表应用迟滞，统计整体精度/召回/F1。"""
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
    """二维网格搜索迟滞参数，返回最佳 on/off 与指标。"""
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
    """按数据集级别统计正负样本帧数，计算 pos_weight=neg/pos。"""
    pos = 0.0
    total = 0.0
    # SVDDataset 已含特征，可直接使用 targets；LazySVDDataset 用音频时长估计 T 再生成目标。
    if isinstance(ds, LazySVDDataset):
        for i in indices:
            it = ds.items[i]
            audio_path: Path = it["audio_path"]
            labels: List[Dict] = it["labels"]
            try:
                duration = librosa.get_duration(filename=str(audio_path))
            except Exception:
                # 若失败，退化为按标签最大边界估计
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
        # SVDDataset
        for i in indices:
            s = ds.samples[i]
            y = s.targets  # 已按 instrumental 处理
            pos += float(y.sum())
            total += float(y.size)

    neg = max(0.0, total - pos)
    # 防止除零与极端值，限制到合理范围
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
    mode: str = "vocal",  # "vocal" 或 "instrumental"
    save_path: Optional[Path] = None,
    eval_hyst_on: Optional[float] = None,
    eval_hyst_off: Optional[float] = None,
):
    set_seed(42)
    assert mode in ("vocal", "instrumental"), "mode 仅支持 'vocal' 或 'instrumental'"
    instrumental = (mode == "instrumental")
    # 更省内存的默认：使用惰性加载数据集
    ds = LazySVDDataset(labels_root=labels_root, music_root=music_root, cache_dir=cache_dir, instrumental=instrumental) if lazy else SVDDataset(labels_root=labels_root, music_root=music_root, instrumental=instrumental)
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

    # 计算数据集级 pos_weight（正类为标签为 1 的帧；instrumental 模式中已反转）
    pos_weight = compute_pos_weight_for_subset(ds, train_idx, instrumental)
    print(f"pos_weight={pos_weight:.2f} (mode={mode})")

    best_f1 = -1.0
    if save_path is None:
        default_name = "lstm_svd_inst.pt" if instrumental else "lstm_svd.pt"
        best_path = Path(".") / default_name
    else:
        best_path = Path(save_path)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight)

        # 收集验证输出一次（用于迟滞评估/搜索；保留默认阈值 0.5 指标仅作参考）
        probs_flat, y_flat, probs_seqs, y_seqs, val_loss = collect_val_outputs(model, val_loader, device)
        preds_05 = (probs_flat >= 0.5).astype(np.float32)
        m05 = compute_metrics_from_arrays(preds_05, y_flat)

        # 迟滞评估：若命令行提供了 on/off 则先报告该对参数的效果；同时进行网格搜索选优
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} f1@0.5={m05['f1']:.4f} precision@0.5={m05['precision']:.4f} recall@0.5={m05['recall']:.4f}")
        if (eval_hyst_on is not None) and (eval_hyst_off is not None) and (eval_hyst_on > eval_hyst_off):
            mh = compute_hysteresis_metrics(probs_seqs, y_seqs, float(eval_hyst_on), float(eval_hyst_off))
            print(f"  hyst(on={float(eval_hyst_on):.2f},off={float(eval_hyst_off):.2f}) f1={mh['f1']:.4f} precision={mh['precision']:.4f} recall={mh['recall']:.4f}")

        # 迟滞网格搜索（二维）：选出最佳 (th_on, th_off)
        th_on_grid = [round(t, 2) for t in np.linspace(0.4, 0.9, num=11).tolist()]
        th_off_grid = [round(t, 2) for t in np.linspace(0.1, 0.6, num=11).tolist()]
        best_on, best_off, best_hyst = grid_search_hysteresis(probs_seqs, y_seqs, th_on_grid, th_off_grid)

        print(f"  best_hyst on={best_on} off={best_off} f1={best_hyst['f1']:.4f} precision={best_hyst['precision']:.4f} recall={best_hyst['recall']:.4f}")

        # 仅使用迟滞作为选优与保存逻辑
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
    parser = argparse.ArgumentParser(description="训练单层 LSTM（支持人声或间奏模式）")
    parser.add_argument("--labels-root", default=str(Path(".") / "labels"))
    parser.add_argument("--music-root", default=r"D:\meliris\music")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lazy", action="store_true", default=True)
    parser.add_argument("--no-lazy", dest="lazy", action="store_false")
    parser.set_defaults(lazy=True)
    parser.add_argument("--mode", choices=["vocal", "instrumental"], default="instrumental", help="训练模式：人声(vocal)或间奏(instrumental)")
    parser.add_argument("--save-path", default=None, help="自定义权重保存路径，默认 vocal 为 lstm_svd.pt，instrumental 为 lstm_svd_inst.pt")
    parser.add_argument("--eval-hyst-on", type=float, default=0.6, help="评估时迟滞开启阈值（可选）。")
    parser.add_argument("--eval-hyst-off", type=float, default=0.3, help="评估时迟滞关闭阈值（可选）。")
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