from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from collections import OrderedDict
from torch.utils.data import Sampler

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa 未安装，请在环境中安装后再运行脚本。") from e

from .config import SR, N_MELS, N_FFT, HOP_LENGTH, HOP_SEC, AUDIO_EXTS_ORDER
from .data import load_mel, labels_to_frame_targets

@dataclass(frozen=True)
class _Seg:
    start: float
    end: float


def _labels_to_segments(labels: List[Dict]) -> List[_Seg]:
    segs: List[_Seg] = []
    for item in labels:
        try:
            start_m, start_s = item["start"][0], item["start"][1]
            end_m, end_s = item["end"][0], item["end"][1]
            start_sec = float(start_m) * 60.0 + float(start_s)
            end_sec = float(end_m) * 60.0 + float(end_s)
        except Exception:
            continue
        if not (math.isfinite(start_sec) and math.isfinite(end_sec)):
            continue
        if end_sec <= start_sec:
            continue
        segs.append(_Seg(start=start_sec, end=end_sec))
    segs.sort(key=lambda s: (s.start, s.end))
    return segs


def _merge_union_segs(segs: List[_Seg]) -> List[_Seg]:
    merged: List[_Seg] = []
    for s in segs:
        if not merged:
            merged.append(s)
            continue
        last = merged[-1]
        if s.start <= last.end:
            merged[-1] = _Seg(start=last.start, end=max(last.end, s.end))
        else:
            merged.append(s)
    return merged


def _overlaps_any(segs: List[_Seg], a: float, b: float) -> bool:
    if b <= a:
        return False
    # linear scan is fine (segments per song is small)
    for s in segs:
        if s.start >= b:
            break
        if s.end > a and s.start < b:
            return True
    return False


def _contains_one(segs: List[_Seg], a: float, b: float) -> bool:
    if b <= a:
        return True
    for s in segs:
        if s.start > a:
            break
        if s.start <= a and s.end >= b:
            return True
    return False


def _classify_window_simple(segs_merged: List[_Seg], w0: float, w1: float, tol_sec: float) -> str:
    """Classify a window using the same semantics as tools/stats_dali_window_scan.py.

    Default tol_sec=0 keeps it strict; setting tol_sec>0 enables edge allowance.
    """
    if w1 <= w0:
        return "other"
    # pure vocal: window overlaps exactly one merged seg and edge interlude total <= 2*tol
    overlaps: List[_Seg] = []
    for s in segs_merged:
        if s.start >= w1:
            break
        if s.end > w0 and s.start < w1:
            overlaps.append(s)
            if len(overlaps) > 2:
                break
    if len(overlaps) == 1:
        s = overlaps[0]
        left_inter = max(0.0, s.start - w0)
        right_inter = max(0.0, w1 - s.end)
        if (left_inter + right_inter) <= (2.0 * tol_sec):
            return "pure_vocal"

    # pure interlude: vocal overlap total <=2*tol and must stick to edges
    if not overlaps:
        return "pure_interlude"
    allowed = 2.0 * tol_sec
    if len(overlaps) == 1:
        s = overlaps[0]
        os = max(w0, s.start)
        oe = min(w1, s.end)
        touches_left = (os == w0)
        touches_right = (oe == w1)
        if (touches_left or touches_right) and ((oe - os) <= allowed):
            return "pure_interlude"
    if len(overlaps) == 2:
        s0, s1 = overlaps[0], overlaps[1]
        os0, oe0 = max(w0, s0.start), min(w1, s0.end)
        os1, oe1 = max(w0, s1.start), min(w1, s1.end)
        if os0 == w0 and oe1 == w1 and oe0 <= os1 and ((oe0 - os0) + (oe1 - os1)) <= allowed:
            return "pure_interlude"

    # transition: boundary with tol on both sides
    trans_tol_sec = 0.5
    if trans_tol_sec > 0:
        left = w0 + trans_tol_sec
        right = w1 - trans_tol_sec
        if right > left:
            for s in segs_merged:
                # check start boundary
                t = s.start
                if left <= t <= right and (t - trans_tol_sec) >= w0 and (t + trans_tol_sec) <= w1:
                    if (not _overlaps_any(segs_merged, t - trans_tol_sec, t)) and _contains_one(segs_merged, t, t + trans_tol_sec):
                        return "transition"
                # check end boundary
                t = s.end
                if left <= t <= right and (t - trans_tol_sec) >= w0 and (t + trans_tol_sec) <= w1:
                    if _contains_one(segs_merged, t - trans_tol_sec, t) and (not _overlaps_any(segs_merged, t, t + trans_tol_sec)):
                        return "transition"
    else:
        # tol=0: any boundary inside window counts
        for s in segs_merged:
            if w0 <= s.start <= w1 or w0 <= s.end <= w1:
                return "transition"

    return "other"


def _split_transition_kind(segs_merged: List[_Seg], w0: float, w1: float, tol_sec: float) -> List[str]:
    """Return transition kinds within window.

    - i2v: interlude -> vocal (vocal start boundary)
    - v2i: vocal -> interlude (vocal end boundary)

    NOTE: this follows the same tolerance semantics as the window scan tool:
    boundary must have >= tol on both sides (when tol>0).
    """
    kinds: List[str] = []
    if w1 <= w0:
        return kinds

    if tol_sec <= 0:
        for s in segs_merged:
            if w0 <= s.start <= w1:
                kinds.append("i2v")
            if w0 <= s.end <= w1:
                kinds.append("v2i")
        return kinds

    left = w0 + tol_sec
    right = w1 - tol_sec
    if right <= left:
        return kinds

    for s in segs_merged:
        # i2v at t=start
        t = s.start
        if left <= t <= right and (t - tol_sec) >= w0 and (t + tol_sec) <= w1:
            if (not _overlaps_any(segs_merged, t - tol_sec, t)) and _contains_one(segs_merged, t, t + tol_sec):
                kinds.append("i2v")
        # v2i at t=end
        t = s.end
        if left <= t <= right and (t - tol_sec) >= w0 and (t + tol_sec) <= w1:
            if _contains_one(segs_merged, t - tol_sec, t) and (not _overlaps_any(segs_merged, t, t + tol_sec)):
                kinds.append("v2i")

    return kinds


class LazyWindowedMTLDataset(Dataset):
    """Windowed version of `LazyMTLDataset`.

    - Builds window samples by scanning time with (window_sec, hop_sec), similar to tools/dali_window_scan.
    - Returns the same 5-tuple as `LazyMTLDataset`: (X, y_vocal, y_ratio, energy_total, name)
      so it stays compatible with existing training + `collate_pad_mtl`.
    - Default tolerance is 0 (strict). You can enable tolerance to filter windows by type.
    """

    def __init__(
        self,
        labels_root: Path,
        music_root: Path,
        demucs_root: Path,
        window_sec: float,
        hop_sec: float = 1.0,
        span: str = "first_to_last",
        window_mode: str = "all",
        tolerance_ms: float = 0.0,
        cache_dir: Optional[Path] = None,
        name_with_window: bool = False,
        song_cache_size: int = 1,
        precache_full: bool = False,
        precache_full_compress: bool = False,
    ):
        assert span in ("first_to_last", "zero_to_last")
        assert window_mode in ("all", "pure_vocal", "pure_interlude", "transition", "other")
        self.items: List[Dict] = []
        self.cache_dir = cache_dir
        self.window_sec = float(window_sec)
        self.hop_sec = float(hop_sec)
        self.span = span
        self.window_mode = window_mode
        self.tolerance_ms = float(tolerance_ms)
        self.name_with_window = bool(name_with_window)
        self.song_cache_size = int(song_cache_size)
        self.precache_full = bool(precache_full)
        self.precache_full_compress = bool(precache_full_compress)

        if self.window_sec <= 0 or self.hop_sec <= 0:
            raise ValueError("window_sec and hop_sec must be > 0")
        if self.tolerance_ms < 0:
            raise ValueError("tolerance_ms must be >= 0")

        if self.song_cache_size < 1:
            raise ValueError("song_cache_size must be >= 1")

        if self.precache_full and self.cache_dir is None:
            raise ValueError("precache_full=True requires cache_dir to be set")

        tol_sec = self.tolerance_ms / 1000.0

        for p in labels_root.glob("*.json"):
            stem = p.stem
            audio_path: Optional[Path] = None
            for ext in AUDIO_EXTS_ORDER:
                q = music_root / f"{stem}{ext}"
                if q.exists():
                    audio_path = q
                    break
            if audio_path is None:
                continue

            vocals_path = demucs_root / stem / "vocals.mp3"
            inst_path = demucs_root / stem / "no_vocals.mp3"
            if (not vocals_path.exists()) or (not inst_path.exists()):
                continue

            try:
                labels = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not labels:
                continue

            segs = _labels_to_segments(labels)
            if not segs:
                continue
            merged = _merge_union_segs(segs)
            # Audio timeline starts at 0. If labels contain negative times, clamp to 0
            # to avoid generating negative windows.
            span_start_raw = float(merged[0].start)
            span_start = 0.0 if self.span == "zero_to_last" else max(0.0, span_start_raw)
            span_end = float(max(s.end for s in merged))
            if span_end <= span_start:
                continue

            # Scan windows in seconds (like dali_window_scan)
            n = int(math.floor((span_end - span_start - self.window_sec) / self.hop_sec)) + 1
            if n <= 0:
                continue

            for i in range(n):
                w0 = span_start + i * self.hop_sec
                w1 = w0 + self.window_sec
                lab = _classify_window_simple(merged, w0, w1, tol_sec)
                trans_kinds = _split_transition_kind(merged, w0, w1, tol_sec) if lab == "transition" else []
                if self.window_mode != "all" and lab != self.window_mode:
                    continue
                self.items.append(
                    {
                        "name": stem,
                        "audio_path": audio_path,
                        "labels": labels,
                        "vocals_path": vocals_path,
                        "inst_path": inst_path,
                        "w0_sec": float(w0),
                        "w1_sec": float(w1),
                        "window_type": lab,
                        "transition_kinds": trans_kinds,
                    }
                )

        # Per-worker song-level cache.
        # NOTE: DataLoader uses worker processes; each worker has its own dataset instance.
        # Default size=1 means "only caches the last accessed song".
        self._song_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def __len__(self):
        return len(self.items)

    def _load_full_for_stem(self, it: Dict):
        stem = it["name"]
        if stem in self._song_cache:
            # refresh LRU
            payload = self._song_cache.pop(stem)
            self._song_cache[stem] = payload
            return

        audio_path: Path = it["audio_path"]
        labels: List[Dict] = it["labels"]
        vocals_path: Path = it["vocals_path"]
        inst_path: Path = it["inst_path"]

        # Full-cache (X, y_vocal, ratio, energy) stored in cache_dir.
        # This avoids recomputing demucs mel-energy for every epoch.
        full_cache_path: Optional[Path] = None
        if self.cache_dir is not None:
            full_cache_path = self.cache_dir / f"{stem}_full.npz"
            if full_cache_path.exists():
                try:
                    z = np.load(full_cache_path, allow_pickle=False)
                    X2 = np.asarray(z["X"], dtype=np.float32)
                    y_vocal = np.asarray(z["y_vocal"], dtype=np.float32)
                    ratio = np.asarray(z["ratio"], dtype=np.float32)
                    energy_total = np.asarray(z["energy"], dtype=np.float32)

                    self._song_cache[stem] = {
                        "X": X2,
                        "y_vocal": y_vocal,
                        "ratio": ratio,
                        "energy": energy_total,
                    }
                    while len(self._song_cache) > self.song_cache_size:
                        self._song_cache.popitem(last=False)
                    return
                except Exception:
                    # fall back to recompute
                    pass

        # NOTE: We intentionally removed the legacy <stem>.npy (mel-only) cache.
        # With full cache in place, keeping two cache formats is unnecessary.
        X = load_mel(audio_path)

        yv, sr_v = librosa.load(str(vocals_path), sr=SR, mono=True)
        yi, sr_i = librosa.load(str(inst_path), sr=SR, mono=True)
        VS = librosa.feature.melspectrogram(y=yv, sr=sr_v, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
        IS = librosa.feature.melspectrogram(y=yi, sr=sr_i, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
        V_energy = VS.sum(axis=0)
        I_energy = IS.sum(axis=0)
        T_min = min(X.shape[0], V_energy.shape[0], I_energy.shape[0])
        X2 = np.asarray(X[:T_min], dtype=np.float32)
        V_energy = V_energy[:T_min]
        I_energy = I_energy[:T_min]
        ratio = (V_energy / (V_energy + I_energy + 1e-8)).astype(np.float32)
        energy_total = (V_energy + I_energy).astype(np.float32)
        y_vocal = labels_to_frame_targets(labels, T=T_min)

        # Write full cache on miss (if cache_dir is set). This makes caching effective
        # even without an explicit precache step.
        if full_cache_path is not None and (not full_cache_path.exists()):
            try:
                if self.precache_full_compress:
                    np.savez_compressed(full_cache_path, X=X2, y_vocal=y_vocal, ratio=ratio, energy=energy_total)
                else:
                    np.savez(full_cache_path, X=X2, y_vocal=y_vocal, ratio=ratio, energy=energy_total)
            except Exception:
                pass

        self._song_cache[stem] = {
            "X": X2,
            "y_vocal": y_vocal,
            "ratio": ratio,
            "energy": energy_total,
        }
        while len(self._song_cache) > self.song_cache_size:
            self._song_cache.popitem(last=False)

    def precache_all(self) -> int:
        """Precompute and write full-cache for all unique stems.

        Returns number of stems attempted.
        """
        if self.cache_dir is None:
            raise ValueError("cache_dir must be set to precache")
        stems_seen = set()
        count = 0
        # show progress
        for it in tqdm(self.items, desc="precaching"):
            stem = str(it.get("name", ""))
            if not stem or stem in stems_seen:
                continue
            stems_seen.add(stem)
            count += 1
            # Force load; if precache_full=True this will write file.
            self._load_full_for_stem(it)
        return count

    def __getitem__(self, idx: int):
        it = self.items[idx]
        self._load_full_for_stem(it)
        stem = str(it["name"])
        payload = self._song_cache.get(stem)
        if payload is None:
            # should not happen, but keep it safe
            self._load_full_for_stem(it)
            payload = self._song_cache.get(stem)
        assert payload is not None

        X_full = payload["X"]
        y_full = payload["y_vocal"]
        r_full = payload["ratio"]
        e_full = payload["energy"]

        w0_sec = float(it["w0_sec"])
        win_frames = max(1, int(round(self.window_sec / HOP_SEC)))
        start_idx_raw = int(round(w0_sec / HOP_SEC))
        end_idx_raw = start_idx_raw + win_frames

        # If window starts before 0, we left-pad with zeros.
        left_pad = max(0, -start_idx_raw)
        start_idx = max(0, start_idx_raw)
        end_idx = max(start_idx, end_idx_raw)

        T = int(X_full.shape[0])
        F = int(X_full.shape[1])

        # slice then pad to fixed length (keeps compatibility with collate_pad_mtl and fixed-window training)
        if end_idx_raw <= 0 or start_idx >= T:
            Xw = np.zeros((win_frames, F), dtype=np.float32)
            yw = np.zeros((win_frames,), dtype=np.float32)
            rw = np.zeros((win_frames,), dtype=np.float32)
            ew = np.zeros((win_frames,), dtype=np.float32)
        else:
            X_part = X_full[start_idx:min(end_idx, T)]
            y_part = y_full[start_idx:min(end_idx, T)]
            r_part = r_full[start_idx:min(end_idx, T)]
            e_part = e_full[start_idx:min(end_idx, T)]

            if left_pad > 0:
                X_part = np.pad(X_part, ((left_pad, 0), (0, 0)), mode="constant")
                y_part = np.pad(y_part, (left_pad, 0), mode="constant")
                r_part = np.pad(r_part, (left_pad, 0), mode="constant")
                e_part = np.pad(e_part, (left_pad, 0), mode="constant")

            pad = win_frames - int(X_part.shape[0])
            if pad > 0:
                Xw = np.pad(X_part, ((0, pad), (0, 0)), mode="constant")
                yw = np.pad(y_part, (0, pad), mode="constant")
                rw = np.pad(r_part, (0, pad), mode="constant")
                ew = np.pad(e_part, (0, pad), mode="constant")
            else:
                Xw = X_part[:win_frames]
                yw = y_part[:win_frames]
                rw = r_part[:win_frames]
                ew = e_part[:win_frames]

        name = stem
        if self.name_with_window:
            name = f"{stem}@{w0_sec:.3f}s"
        return (
            torch.from_numpy(np.asarray(Xw, dtype=np.float32).copy()),
            torch.from_numpy(np.asarray(yw, dtype=np.float32).copy()),
            torch.from_numpy(np.asarray(rw, dtype=np.float32).copy()),
            torch.from_numpy(np.asarray(ew, dtype=np.float32).copy()),
            name,
        )


def _split_counts(batch_size: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """Turn ratios into integer counts summing to batch_size."""
    keys = list(ratios.keys())
    raw = [max(0.0, float(ratios[k])) for k in keys]
    s = sum(raw)
    if s <= 0:
        raise ValueError("ratios sum must be > 0")
    raw = [x / s for x in raw]
    base = [int(math.floor(batch_size * x)) for x in raw]
    rem = batch_size - sum(base)
    frac = [(batch_size * raw[i] - base[i], i) for i in range(len(keys))]
    frac.sort(reverse=True)
    for j in range(rem):
        base[frac[j][1]] += 1
    return {keys[i]: base[i] for i in range(len(keys))}


class WindowTypeBatchSampler(Sampler[List[int]]):
    """BatchSampler that enforces per-batch window-type ratios.

    Typical target ratios:
      - 30%: transition (optionally v2i only)
      - 40%: pure_interlude
      - 30%: pure_vocal

    Notes:
    - This sampler runs in the main process (before workers), so it can control composition.
    - It samples WITH replacement by default to avoid exhausting rare types.
    """

    def __init__(
        self,
        dataset: LazyWindowedMTLDataset,
        batch_size: int,
        ratios: Dict[str, float],
        *,
        transition_direction: str = "any",  # any|v2i|i2v
        seed: int = 42,
        drop_last: bool = True,
        with_replacement: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if transition_direction not in ("any", "v2i", "i2v"):
            raise ValueError("transition_direction must be any|v2i|i2v")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.ratios = dict(ratios)
        self.transition_direction = transition_direction
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.with_replacement = bool(with_replacement)

        # Build pools
        self.pools: Dict[str, List[int]] = {k: [] for k in self.ratios.keys()}
        for i, it in enumerate(self.dataset.items):
            t = str(it.get("window_type", ""))
            if t == "transition" and transition_direction != "any":
                kinds = it.get("transition_kinds", []) or []
                if transition_direction not in kinds:
                    continue
            if t in self.pools:
                self.pools[t].append(i)

        missing = [k for k, v in self.pools.items() if len(v) == 0]
        if missing:
            raise RuntimeError(f"No samples for window types: {missing}. Try different tolerance_ms or window_sec.")

        self.counts = _split_counts(self.batch_size, self.ratios)

        # simple RNG
        import random
        self._rng = random.Random(self.seed)

    def __iter__(self):
        # Decide number of batches based on dominant pool
        if self.with_replacement:
            n_batches = int(math.ceil(len(self.dataset) / self.batch_size))
        else:
            n_batches = min(
                int(len(self.pools[k]) // max(1, c)) if c > 0 else 10**9
                for k, c in self.counts.items()
            )

        for _ in range(n_batches):
            batch: List[int] = []
            for k, c in self.counts.items():
                if c <= 0:
                    continue
                pool = self.pools[k]
                if self.with_replacement:
                    batch.extend(self._rng.choices(pool, k=c))
                else:
                    # without replacement (shuffle pool each epoch is recommended)
                    self._rng.shuffle(pool)
                    batch.extend(pool[:c])
                    del pool[:c]
            self._rng.shuffle(batch)
            if len(batch) == self.batch_size:
                yield batch

    def __len__(self) -> int:
        if self.with_replacement:
            return int(math.ceil(len(self.dataset) / self.batch_size))
        return min(
            int(len(self.pools[k]) // max(1, c)) if c > 0 else 10**9
            for k, c in self.counts.items()
        )