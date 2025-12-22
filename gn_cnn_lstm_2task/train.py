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
from gn_cnn_lstm_2task.model import GNCNNLSTM2Head
from common import (
    SR, N_MELS, N_FFT, HOP_LENGTH, HOP_SEC,
    LazyMTLDataset, collate_pad_mtl,
    set_seed, compute_pos_weight_for_subset,
    compute_metrics_from_arrays, grid_search_hysteresis, compute_hysteresis_metrics,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    gn_groups: int = 8,
    lr: float = 1e-3,
    seed: int = 42,
    save_path: Optional[Path] = None,
    eval_hyst_on: Optional[float] = None,
    eval_hyst_off: Optional[float] = None,
    energy_thresh: float = 0.0,
):
    set_seed(seed)
    assert mode in ("vocal", "instrumental"), "mode 仅支持 'vocal' 或 'instrumental'"
    instrumental = (mode == "instrumental")
    ds = LazyMTLDataset(labels_root=labels_root, music_root=music_root, demucs_root=demucs_root, cache_dir=cache_dir)
    print(f"数据集样本数：{len(ds)}")

    if len(ds) == 0:
        raise RuntimeError("数据集为空：请确认 labels/perfect 与 music 目录下存在同名有效数据。")

    # 使样本顺序稳定（避免文件系统遍历顺序影响划分）
    ds.items.sort(key=lambda it: str(it.get("name", "")))

    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
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
        collate_fn=collate_pad_mtl,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pad_mtl,
        pin_memory=use_cuda,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNCNNLSTM2Head(input_dim=N_MELS, cnn_channels=64, hidden=128, bidirectional=bilstm, gn_groups=gn_groups).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    pos_weight = compute_pos_weight_for_subset(ds, train_idx, instrumental)
    print(f"pos_weight={pos_weight:.2f} (mode={mode})")

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
                "norm": "gn",
                "gn_groups": int(gn_groups),
                "lr": float(lr),
                "seed": int(seed),
                "split": {"train_idx": train_idx, "val_idx": val_idx},
                "eval_postproc": "hysteresis",
                "eval_threshold": None,
                "eval_hyst_on": best_on,
                "eval_hyst_off": best_off,
                "energy_thresh": float(energy_thresh),
            }
            if save_path is None:
                fname = (
                    f"gn_cnn_lstm_2task{'_bilstm' if bilstm else ''}_{'inst' if instrumental else 'vocal'}_epoch-{epoch}_lambda2-{lambda2}_gn-{gn_groups}_lr-{lr}_energy-{energy_thresh}_ts-{run_ts}.pt"
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
    parser.add_argument("--labels-root", default=str(Path(".") / "labels_qrc"))
    parser.add_argument("--music-root", default=r"D:\meliris\music")
    parser.add_argument("--demucs-root", default=r"D:\ferry\Demucs-GUI_1.3.2_cuda_mkl\separated\htdemucs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lazy", action="store_true", default=True)
    parser.add_argument("--no-lazy", dest="lazy", action="store_false")
    parser.set_defaults(lazy=True)
    parser.add_argument("--mode", choices=["vocal", "instrumental"], default="instrumental")
    parser.add_argument("--bilstm", action="store_true", default=False)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.75)
    parser.add_argument("--gn-groups", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--eval-hyst-on", type=float, default=0.6)
    parser.add_argument("--eval-hyst-off", type=float, default=0.3)
    parser.add_argument("--energy-thresh", type=float, default=0.0)
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
        gn_groups=int(args.gn_groups),
        lr=float(args.lr),
        seed=int(args.seed),
        save_path=(Path(args.save_path) if args.save_path else None),
        eval_hyst_on=args.eval_hyst_on,
        eval_hyst_off=args.eval_hyst_off,
        energy_thresh=args.energy_thresh,
    )
    print("training done:", stats)


if __name__ == "__main__":
    main()
