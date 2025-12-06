from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa 未安装，请在环境中安装后再运行脚本。") from e

from .config import SR, N_MELS, N_FFT, HOP_LENGTH, HOP_SEC, AUDIO_EXTS_ORDER


def find_audio_by_stem(music_root: Path, stem: str) -> Optional[Path]:
    for ext in AUDIO_EXTS_ORDER:
        p = music_root / f"{stem}{ext}"
        if p.exists():
            return p
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
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm.T.astype(np.float32)  # [T, n_mels]


def load_features(audio_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    mel = S_norm.T.astype(np.float32)
    d1 = librosa.feature.delta(S_norm, order=1)
    d2 = librosa.feature.delta(S_norm, order=2)
    d1 = d1.T.astype(np.float32)
    d2 = d2.T.astype(np.float32)
    d1_max = float(np.max(np.abs(d1))) if d1.size > 0 else 1.0
    d2_max = float(np.max(np.abs(d2))) if d2.size > 0 else 1.0
    d1 = d1 / (d1_max + 1e-8)
    d2 = d2 / (d2_max + 1e-8)
    Xmag = np.abs(librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    flat = librosa.feature.spectral_flatness(S=Xmag).squeeze().astype(np.float32)
    harm, perc = librosa.decompose.hpss(Xmag)
    he = harm.sum(axis=0)
    pe = perc.sum(axis=0)
    ratio = (he / (he + pe + 1e-8)).astype(np.float32)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH).squeeze().astype(np.float32)
    T_min = int(min(mel.shape[0], d1.shape[0], d2.shape[0], flat.shape[0], ratio.shape[0], zcr.shape[0]))
    mel = mel[:T_min]
    d1 = d1[:T_min]
    d2 = d2[:T_min]
    flat = flat[:T_min]
    ratio = ratio[:T_min]
    zcr = zcr[:T_min]
    aux = np.stack([ratio, flat, zcr], axis=1).astype(np.float32)
    X = np.concatenate([mel, d1, d2, aux], axis=1).astype(np.float32)
    return X


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
        perfect_root = labels_root / "perfect"
        for json_path in perfect_root.glob("*.json"):
            stem = json_path.stem
            try:
                labels = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not labels:
                continue
            audio_path = find_audio_by_stem(music_root, stem)
            if audio_path is None:
                continue
            try:
                X = load_mel(audio_path)
            except Exception:
                continue
            y = labels_to_frame_targets(labels, T=X.shape[0])
            if self.instrumental:
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


class LazySVDDatasetMulFeatures(Dataset):
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

        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{stem}.npy"
            if cache_file.exists():
                X = np.load(cache_file, mmap_mode="r")
            else:
                X = load_features(audio_path)
                try:
                    np.save(cache_file, X)
                except Exception:
                    pass
        else:
            X = load_features(audio_path)

        y = labels_to_frame_targets(labels, T=X.shape[0])
        if self.instrumental:
            y = 1.0 - y
        return torch.from_numpy(X.copy()), torch.from_numpy(y.copy()), stem


def collate_pad(batch):
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


class LazyMTLDataset(Dataset):
    def __init__(self, labels_root: Path, music_root: Path, demucs_root: Path, cache_dir: Optional[Path] = None):
        self.items: List[Dict] = []
        self.cache_dir = cache_dir
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
            vocals_path = demucs_root / stem / "vocals.flac"
            inst_path = demucs_root / stem / "no_vocals.flac"
            if (not vocals_path.exists()) or (not inst_path.exists()):
                continue
            try:
                labels = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                labels = []
            self.items.append({
                "name": stem,
                "audio_path": audio_path,
                "labels": labels,
                "vocals_path": vocals_path,
                "inst_path": inst_path,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        stem = it["name"]
        audio_path: Path = it["audio_path"]
        labels: List[Dict] = it["labels"]
        vocals_path: Path = it["vocals_path"]
        inst_path: Path = it["inst_path"]

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

        yv, sr_v = librosa.load(str(vocals_path), sr=SR, mono=True)
        yi, sr_i = librosa.load(str(inst_path), sr=SR, mono=True)
        VS = librosa.feature.melspectrogram(y=yv, sr=sr_v, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
        IS = librosa.feature.melspectrogram(y=yi, sr=sr_i, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
        V_energy = VS.sum(axis=0)
        I_energy = IS.sum(axis=0)
        T_min = min(X.shape[0], V_energy.shape[0], I_energy.shape[0])
        X = X[:T_min]
        V_energy = V_energy[:T_min]
        I_energy = I_energy[:T_min]
        ratio = (V_energy / (V_energy + I_energy + 1e-8)).astype(np.float32)
        energy_total = (V_energy + I_energy).astype(np.float32)
        y_vocal = labels_to_frame_targets(labels, T=T_min)
        return (
            torch.from_numpy(X.copy()),
            torch.from_numpy(y_vocal.copy()),
            torch.from_numpy(ratio.copy()),
            torch.from_numpy(energy_total.copy()),
            stem,
        )


class LazyMTLDatasetMulFeatures(Dataset):
    def __init__(self, labels_root: Path, music_root: Path, demucs_root: Path, cache_dir: Optional[Path] = None):
        self.items: List[Dict] = []
        self.cache_dir = cache_dir
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
            vocals_path = demucs_root / stem / "vocals.flac"
            inst_path = demucs_root / stem / "no_vocals.flac"
            if (not vocals_path.exists()) or (not inst_path.exists()):
                continue
            try:
                labels = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                labels = []
            self.items.append({
                "name": stem,
                "audio_path": audio_path,
                "labels": labels,
                "vocals_path": vocals_path,
                "inst_path": inst_path,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        stem = it["name"]
        audio_path: Path = it["audio_path"]
        labels: List[Dict] = it["labels"]
        vocals_path: Path = it["vocals_path"]
        inst_path: Path = it["inst_path"]

        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{stem}.npy"
            if cache_file.exists():
                X = np.load(cache_file, mmap_mode="r")
            else:
                X = load_features(audio_path)
                try:
                    np.save(cache_file, X)
                except Exception:
                    pass
        else:
            X = load_features(audio_path)

        yv, sr_v = librosa.load(str(vocals_path), sr=SR, mono=True)
        yi, sr_i = librosa.load(str(inst_path), sr=SR, mono=True)
        VS = librosa.feature.melspectrogram(y=yv, sr=sr_v, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
        IS = librosa.feature.melspectrogram(y=yi, sr=sr_i, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
        V_energy = VS.sum(axis=0)
        I_energy = IS.sum(axis=0)
        T_min = min(X.shape[0], V_energy.shape[0], I_energy.shape[0])
        X = X[:T_min]
        V_energy = V_energy[:T_min]
        I_energy = I_energy[:T_min]
        ratio = (V_energy / (V_energy + I_energy + 1e-8)).astype(np.float32)
        energy_total = (V_energy + I_energy).astype(np.float32)
        y_vocal = labels_to_frame_targets(labels, T=T_min)
        return (
            torch.from_numpy(X.copy()),
            torch.from_numpy(y_vocal.copy()),
            torch.from_numpy(ratio.copy()),
            torch.from_numpy(energy_total.copy()),
            stem,
        )


def collate_pad_mtl(batch):
    names = [b[4] for b in batch]
    lengths = [b[0].shape[0] for b in batch]
    F = batch[0][0].shape[1]
    T_max = max(lengths)
    X_pad = torch.zeros((len(batch), T_max, F), dtype=torch.float32)
    y_vocal_pad = torch.zeros((len(batch), T_max), dtype=torch.float32)
    y_ratio_pad = torch.zeros((len(batch), T_max), dtype=torch.float32)
    energy_pad = torch.zeros((len(batch), T_max), dtype=torch.float32)
    mask = torch.zeros((len(batch), T_max), dtype=torch.float32)
    for i, (X, y_vocal, y_ratio, energy, _) in enumerate(batch):
        t = X.shape[0]
        X_pad[i, :t, :] = X
        y_vocal_pad[i, :t] = y_vocal
        y_ratio_pad[i, :t] = y_ratio
        energy_pad[i, :t] = energy
        mask[i, :t] = 1.0
    return X_pad, y_vocal_pad, y_ratio_pad, energy_pad, mask, names, lengths
