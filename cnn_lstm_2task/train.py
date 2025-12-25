from __future__ import annotations

import json
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa 未安装，请在环境中安装后再运行训练脚本。") from e


# 共享配置与工具（抽取到 common 模块）
from cnn_lstm_2task.model import CNNLSTM2Head
from common import (
    SR, N_MELS, N_FFT, HOP_LENGTH, HOP_SEC,
    LazyWindowedMTLDataset, WindowTypeBatchSampler, collate_pad_mtl,
    set_seed, compute_pos_weight_for_subset,
    compute_metrics_from_arrays, grid_search_hysteresis, compute_hysteresis_metrics,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_ratios(s: Optional[str]) -> Optional[Dict[str, float]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: Dict[str, float] = {}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"无效 ratios 格式: {p} (应为 key=value)")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        out[k] = float(v)
    return out


def _split_train_val_by_stem(items: List[Dict], train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    stems = sorted({str(it.get("name", "")) for it in items if str(it.get("name", ""))})
    if not stems:
        return [], []
    rng = random.Random(int(seed))
    rng.shuffle(stems)
    n_train = max(1, int(len(stems) * float(train_ratio)))
    train_stems = set(stems[:n_train])
    train_idx: List[int] = []
    val_idx: List[int] = []
    for i, it in enumerate(items):
        stem = str(it.get("name", ""))
        if stem in train_stems:
            train_idx.append(i)
        else:
            val_idx.append(i)
    return train_idx, val_idx


class _IndexViewDataset(torch.utils.data.Dataset):
    """A lightweight view over a base dataset with selected indices.

    - Keeps an `items` attribute so WindowTypeBatchSampler can read window_type metadata.
    - Delegates __getitem__ to the base dataset.
    """

    def __init__(self, base: torch.utils.data.Dataset, indices: List[int], base_items: Optional[List[Dict]] = None):
        self.base = base
        self.indices = list(indices)
        if base_items is None:
            self.items = []
        else:
            self.items = [base_items[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


## mel 频谱提取已抽取至 common.data.load_mel


## 标签到帧级目标转换已抽取至 common.data.labels_to_frame_targets


## 数据集与 collate 函数已抽取至 common.data


def train_one_epoch(model, loader, optimizer, device, pos_weight: Optional[float], instrumental: bool, lambda1: float, lambda2: float, energy_thresh: float):
    model.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=(torch.tensor([pos_weight], dtype=torch.float32, device=device) if pos_weight is not None else None),
    )
    mse = nn.MSELoss(reduction="none")
    for X, y_vocal, y_ratio, energy, mask, _, _ in tqdm(loader, desc="train", leave=False):
        X = X.to(device)
        y_vocal = y_vocal.to(device)
        y_ratio = y_ratio.to(device)
        energy = energy.to(device)
        mask = mask.to(device)

        logits_cls, logits_reg = model(X)
        y_cls = (1.0 - y_vocal) if instrumental else y_vocal
        loss_cls = bce(logits_cls, y_cls)
        loss_cls = (loss_cls * mask).sum() / (mask.sum() + 1e-8)
        pred_ratio = torch.sigmoid(logits_reg)
        mask_reg = mask * y_vocal * (energy >= energy_thresh).float()
        loss_reg = mse(pred_ratio, y_ratio)
        loss_reg = (loss_reg * mask_reg).sum() / (mask_reg.sum() + 1e-8)
        loss = lambda1 * loss_cls + lambda2 * loss_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def collect_val_outputs(model, loader, device, instrumental: bool):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    probs_flat: List[np.ndarray] = []
    y_flat: List[np.ndarray] = []
    probs_seqs: List[np.ndarray] = []
    y_seqs: List[np.ndarray] = []
    for X, y_vocal, y_ratio, energy, mask, _, _ in tqdm(loader, desc="collect", leave=False):
        X = X.to(device)
        y_vocal = y_vocal.to(device)
        mask = mask.to(device)
        logits_cls, _ = model(X)
        y_cls = (1.0 - y_vocal) if instrumental else y_vocal
        loss = bce(logits_cls, y_cls)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        total_loss += loss.item()
        probs = torch.sigmoid(logits_cls)
        m = (mask == 1)
        probs_flat.append(probs[m].detach().cpu().numpy())
        y_flat.append(y_cls[m].detach().cpu().numpy())
        B = X.shape[0]
        for i in range(B):
            t_len = int(mask[i].sum().item())
            probs_seqs.append(probs[i, :t_len].detach().cpu().numpy())
            y_seqs.append(y_cls[i, :t_len].detach().cpu().numpy())
    probs_flat_all = np.concatenate(probs_flat) if probs_flat else np.zeros((0,), dtype=np.float32)
    y_flat_all = np.concatenate(y_flat) if y_flat else np.zeros((0,), dtype=np.float32)
    return probs_flat_all, y_flat_all, probs_seqs, y_seqs, (total_loss / max(1, len(loader)))


## 指标与迟滞搜索已抽取至 common.postproc


## pos_weight 计算已抽取至 common.train_utils


def run_training(
    labels_root: Path,
    music_root: Path,
    demucs_root: Path,
    epochs: int = 5,
    batch_size: int = 2,
    lazy: bool = True,
    cache_dir: Optional[Path] = Path("./features_cache"),
    mode: str = "vocal",
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    bilstm: bool = False,
    save_path: Optional[Path] = None,
    eval_hyst_on: Optional[float] = None,
    eval_hyst_off: Optional[float] = None,
    energy_thresh: float = 0.0,
    window_sec: float = 5.0,
    hop_sec: float = 1.0,
    span: str = "first_to_last",
    window_mode: str = "all",
    tolerance_ms: float = 0.0,
    name_with_window: bool = False,
    song_cache_size: int = 1,
    batch_type_ratios: Optional[str] = None,
    transition_direction: str = "any",
    pos_weight_manual: Optional[float] = None,
    no_pos_weight: bool = True,
    precache_full: bool = False,
    precache_full_compress: bool = False,
):
    set_seed(42)
    assert mode in ("vocal", "instrumental"), "mode 仅支持 'vocal' 或 'instrumental'"
    instrumental = (mode == "instrumental")

    ds = LazyWindowedMTLDataset(
        labels_root=labels_root,
        music_root=music_root,
        demucs_root=demucs_root,
        window_sec=float(window_sec),
        hop_sec=float(hop_sec),
        span=str(span),
        window_mode=str(window_mode),
        tolerance_ms=float(tolerance_ms),
        cache_dir=cache_dir,
        name_with_window=bool(name_with_window),
        song_cache_size=int(song_cache_size),
        precache_full=bool(precache_full),
        precache_full_compress=bool(precache_full_compress),
    )
    if len(ds) == 0:
        raise RuntimeError("数据集为空：请确认 labels/perfect 与 music 目录下存在同名有效数据。")

    if precache_full:
        n = ds.precache_all()
        print(f"precache_full done: stems={n}")

    # IMPORTANT: split by stem to avoid leakage (windows from same song in both train/val)
    train_idx, val_idx = _split_train_val_by_stem(ds.items, train_ratio=0.9, seed=42)
    if not train_idx or not val_idx:
        raise RuntimeError(f"train/val 划分失败：train={len(train_idx)} val={len(val_idx)}")

    train_view = _IndexViewDataset(ds, train_idx, base_items=ds.items)
    val_view = _IndexViewDataset(ds, val_idx, base_items=ds.items)

    use_cuda = torch.cuda.is_available()

    ratios = _parse_ratios(batch_type_ratios)
    if ratios is not None:
        batch_sampler = WindowTypeBatchSampler(
            train_view,  # type: ignore[arg-type]
            batch_size=int(batch_size),
            ratios=ratios,
            transition_direction=str(transition_direction),
            seed=42,
            with_replacement=True,
        )
        train_loader = DataLoader(
            train_view,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=collate_pad_mtl,
            pin_memory=use_cuda,
        )
    else:
        train_loader = DataLoader(
            train_view,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_pad_mtl,
            pin_memory=use_cuda,
        )
    val_loader = DataLoader(
        val_view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pad_mtl,
        pin_memory=use_cuda,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM2Head(input_dim=N_MELS, cnn_channels=64, hidden=128, bidirectional=bilstm).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # pos_weight notes:
    # - windowed(first_to_last) excludes intro/outro interludes, so class ratio differs from full-song dataset
    # - if you already enforce per-batch window-type ratios, you may want to disable pos_weight to avoid double balancing
    if no_pos_weight:
        pos_weight = None
        print(f"pos_weight=None (disabled, mode={mode})")
    elif pos_weight_manual is not None:
        pos_weight = float(pos_weight_manual)
        print(f"pos_weight={pos_weight:.2f} (manual, mode={mode})")
    else:
        pos_weight = compute_pos_weight_for_subset(ds, train_idx, instrumental)
        print(f"pos_weight={pos_weight:.2f} (auto, mode={mode})")

    best_f1 = -1.0
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    last_saved_path: Optional[Path] = None
    base_dir = Path(".") if save_path is None else Path(save_path).parent

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pos_weight=pos_weight,
            instrumental=instrumental,
            lambda1=lambda1,
            lambda2=lambda2,
            energy_thresh=energy_thresh,
        )

        probs_flat, y_flat, probs_seqs, y_seqs, val_loss = collect_val_outputs(model, val_loader, device, instrumental)
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
                "lambda1": float(lambda1),
                "lambda2": float(lambda2),
                "bilstm": bool(bilstm),
                "eval_postproc": "hysteresis",
                "eval_threshold": None,
                "eval_hyst_on": best_on,
                "eval_hyst_off": best_off,
                "energy_thresh": float(energy_thresh),
            }
            if save_path is None:
                fname = (
                    f"cnn_lstm_2task{'_bilstm' if bilstm else ''}_{'inst' if instrumental else 'vocal'}_epoch-{epoch}_lambda2-{lambda2}_energy-{energy_thresh}_ts-{run_ts}.pt"
                )
                best_path = base_dir / fname
            else:
                best_path = Path(save_path)
            torch.save({"model_state": model.state_dict(), "config": cfg}, best_path)
            last_saved_path = best_path
            print(f"saved best to {best_path} (f1={best_f1:.4f}, postproc=hysteresis, on={best_on}, off={best_off})")

    final_ckpt = str(last_saved_path) if last_saved_path is not None else (str(Path(save_path)) if save_path is not None else "")
    return {"best_f1": best_f1, "ckpt": final_ckpt}


def main():
    parser = argparse.ArgumentParser(description="训练 CNN+LSTM（支持人声或间奏模式）")
    # 默认对齐 DALI 统计（labels_dali + 5s window + 500ms tolerance）
    parser.add_argument("--labels-root", default=r"D:\meliris\labels_dali")
    parser.add_argument("--music-root", default=r"D:\meliris\dali\DALI_v1.0_audios")
    parser.add_argument("--demucs-root", default=r"D:\ferry\Demucs-GUI_1.3.2_cuda_mkl\separated\htdemucs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lazy", action="store_true", default=True)
    parser.add_argument("--no-lazy", dest="lazy", action="store_false")
    parser.set_defaults(lazy=True)
    parser.add_argument("--mode", choices=["vocal", "instrumental"], default="instrumental")
    parser.add_argument("--bilstm", action="store_true", default=False)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.75)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--eval-hyst-on", type=float, default=0.6)
    parser.add_argument("--eval-hyst-off", type=float, default=0.3)
    parser.add_argument("--energy-thresh", type=float, default=0.0)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--span", choices=["first_to_last", "zero_to_last"], default="zero_to_last")
    parser.add_argument("--window-mode", choices=["all", "pure_vocal", "pure_interlude", "transition", "other"], default="all")
    parser.add_argument("--tolerance-ms", type=float, default=0.0)
    parser.add_argument("--name-with-window", action="store_true", default=False)
    # windowed 数据集按 stem 复用特征，适当增大可减少重复加载（每个 worker 各自缓存）
    parser.add_argument("--song-cache-size", type=int, default=2)
    parser.add_argument(
        "--batch-type-ratios",
        default="transition=0.3,pure_interlude=0.4,pure_vocal=0.3",
        help="按窗口类型组成 batch，例如: transition=0.3,pure_interlude=0.4,pure_vocal=0.3 (留空则普通 shuffle)",
    )
    parser.add_argument("--transition-direction", choices=["any", "v2i", "i2v"], default="any")
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="手动指定 BCE pos_weight（留空=自动估计；与 --no-pos-weight 互斥）",
    )
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        default=True,
        help="禁用 BCE pos_weight（适合已做 batch 比例采样的情况）",
    )
    parser.add_argument(
        "--precache-full",
        action="store_true",
        default=False,
        help="预缓存每首歌的 X/y_vocal/ratio/energy 到 cache_dir（生成 <stem>_full.npz，加速后续 epoch）",
    )
    parser.add_argument(
        "--precache-full-compress",
        action="store_true",
        default=False,
        help="预缓存使用压缩（更省空间但稍慢）",
    )
    args = parser.parse_args()

    labels_root = Path(args.labels_root)
    music_root = Path(args.music_root)
    demucs_root = Path(args.demucs_root)
    stats = run_training(
        labels_root=labels_root,
        music_root=music_root,
        demucs_root=demucs_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lazy=args.lazy,
        cache_dir=Path("./features_cache"),
        mode=args.mode,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        bilstm=bool(args.bilstm),
        save_path=(Path(args.save_path) if args.save_path else None),
        eval_hyst_on=args.eval_hyst_on,
        eval_hyst_off=args.eval_hyst_off,
        energy_thresh=args.energy_thresh,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        span=args.span,
        window_mode=args.window_mode,
        tolerance_ms=args.tolerance_ms,
        name_with_window=bool(args.name_with_window),
        song_cache_size=int(args.song_cache_size),
        batch_type_ratios=str(args.batch_type_ratios),
        transition_direction=str(args.transition_direction),
        pos_weight_manual=(float(args.pos_weight) if args.pos_weight is not None else None),
        no_pos_weight=bool(args.no_pos_weight),
        precache_full=bool(args.precache_full),
        precache_full_compress=bool(args.precache_full_compress),
    )
    print("training done:", stats)


if __name__ == "__main__":
    main()
