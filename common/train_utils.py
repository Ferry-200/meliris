from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa 未安装，请在环境中安装后再运行脚本。") from e

from .config import HOP_SEC
from .data import LazySVDDataset, LazyMTLDataset, labels_to_frame_targets
from .windowed_data import LazyWindowedMTLDataset

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_val_outputs(model, loader, device):
    """Collect probabilities and labels from validation loader.

    Returns flattened arrays for threshold metrics and per-sequence lists
    for hysteresis computations, along with average loss.
    """
    model.eval()
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
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


def compute_pos_weight_for_subset(ds: Dataset, indices: List[int], instrumental: bool) -> float:
    """Estimate pos_weight=neg/pos at dataset level for a subset.

    Supports both eager `SVDDataset` and lazy `LazySVDDataset`.
    """
    pos = 0.0
    total = 0.0
    if isinstance(ds, (LazySVDDataset)):
        for i in indices:
            it = ds.items[i]
            audio_path: Path = it["audio_path"]
            labels: List[Dict] = it["labels"]
            try:
                duration = librosa.get_duration(path=str(audio_path))
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
    elif isinstance(ds, (LazyMTLDataset)):
        for i in indices:
            it = ds.items[i]
            audio_path: Path = it["audio_path"]
            labels: List[Dict] = it["labels"]
            try:
                duration = librosa.get_duration(path=str(audio_path))
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
    elif isinstance(ds, (LazyWindowedMTLDataset)):
        # Windowed dataset: compute per-window frames directly (fixed length)
        win_frames = max(1, int(round(float(ds.window_sec) / HOP_SEC)))
        for i in indices:
            it = ds.items[i]
            labels: List[Dict] = it["labels"]
            w0_sec = float(it["w0_sec"])
            start_idx_raw = int(round(w0_sec / HOP_SEC))
            end_idx_raw = start_idx_raw + win_frames

            if end_idx_raw <= 0:
                y = np.zeros((win_frames,), dtype=np.float32)
            else:
                left_pad = max(0, -start_idx_raw)
                start_idx = max(0, start_idx_raw)
                end_idx = max(start_idx, end_idx_raw)

                # build y only up to end_idx (never negative)
                y_full = labels_to_frame_targets(labels, T=end_idx)
                y = y_full[start_idx:end_idx]
                if left_pad > 0:
                    y = np.pad(y, (left_pad, 0), mode="constant")
                if y.shape[0] < win_frames:
                    y = np.pad(y, (0, win_frames - y.shape[0]), mode="constant")
                elif y.shape[0] > win_frames:
                    y = y[:win_frames]
            if instrumental:
                y = 1.0 - y
            pos += float(y.sum())
            total += float(win_frames)
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
