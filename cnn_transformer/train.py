from __future__ import annotations

import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

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


from common import (
    SR, N_MELS, HOP_LENGTH,
    SVDDataset, LazySVDDataset, collate_pad,
    compute_pos_weight_for_subset,
    compute_metrics_from_arrays, grid_search_hysteresis, compute_hysteresis_metrics,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
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
    """在频率维上做 1D CNN，之后接 Transformer Encoder 做时序分类。"""
    def __init__(self, input_dim: int, cnn_channels: int = 128, nhead: int = 4, num_layers: int = 2, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = cnn_channels
        # 频率维卷积编码每帧谱，输出 d_model 嵌入
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

    def forward(self, X: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # X: [B, T, F], key_padding_mask: [B, T] (pad 为 True)
        B, T, F = X.shape
        x = X.reshape(B * T, 1, F)
        x = self.cnn(x).squeeze(-1)
        x = x.reshape(B, T, -1)
        x = self.pos(x)
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        logits = self.head(out).squeeze(-1)
        return logits


def forward_chunked_logits(
    model: CNNTransformer,
    X: torch.Tensor,
    mask: torch.Tensor,
    max_seq_len: Optional[int] = None,
):
    """对超长时序做分段前向，避免 Transformer 注意力的 O(T^2) 内存爆炸。

    - X: [B, T, F]
    - mask: [B, T]，1 表示有效帧，0 表示 pad
    - 返回: [B, T] 的 logits
    """
    if (max_seq_len is None) or (X.shape[1] <= max_seq_len):
        return model(X, key_padding_mask=(mask == 0))
    B, T, F = X.shape
    outs: List[torch.Tensor] = []
    for st in range(0, T, int(max_seq_len)):
        ed = min(st + int(max_seq_len), T)
        logits_chunk = model(
            X[:, st:ed, :],
            key_padding_mask=(mask[:, st:ed] == 0),
        )
        outs.append(logits_chunk)
    return torch.cat(outs, dim=1)


def train_one_epoch(model, loader, optimizer, device, pos_weight: Optional[float] = None, max_seq_len: Optional[int] = None):
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

        logits = forward_chunked_logits(model, X, mask, max_seq_len=max_seq_len)
        loss = bce(logits, y)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def collect_val_outputs(model, loader, device, max_seq_len: Optional[int] = None):
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
        logits = forward_chunked_logits(model, X, mask, max_seq_len=max_seq_len)
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
    max_seq_len: Optional[int] = 2048,
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
    model = CNNTransformer(input_dim=N_MELS, cnn_channels=128, nhead=4, num_layers=2, ff_dim=256, dropout=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    pos_weight = compute_pos_weight_for_subset(ds, train_idx, instrumental)
    print(f"pos_weight={pos_weight:.2f} (mode={mode})")

    best_f1 = -1.0
    if save_path is None:
        default_name = "cnn_transformer_inst.pt" if instrumental else "cnn_transformer.pt"
        best_path = Path(".") / default_name
    else:
        best_path = Path(save_path)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight, max_seq_len=max_seq_len)

        probs_flat, y_flat, probs_seqs, y_seqs, val_loss = collect_val_outputs(model, val_loader, device, max_seq_len=max_seq_len)
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
    parser = argparse.ArgumentParser(description="训练 CNN+Transformer（支持人声或间奏模式）")
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
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Transformer 前向分段长度，避免 OOM")
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
        max_seq_len=(int(args.max_seq_len) if args.max_seq_len else None),
    )
    print("training done:", stats)


if __name__ == "__main__":
    main()