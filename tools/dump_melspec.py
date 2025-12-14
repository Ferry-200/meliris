import numpy as np
import librosa

from pathlib import Path
from common.config import SR, N_MELS, N_FFT, HOP_LENGTH, HOP_SEC, AUDIO_EXTS_ORDER

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

X = load_mel(Path("D:\\meliris\\music\\17さいのうた。.flac"))
np.save("17さいのうた。.npy", X)