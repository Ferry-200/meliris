from __future__ import annotations

# Audio and feature configuration
SR = 22050
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
HOP_SEC = HOP_LENGTH / SR

# Preferred audio extensions when matching by stem
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