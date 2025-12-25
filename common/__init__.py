"""
Common utilities shared by LSTM and CNN+LSTM modules.

Exposes configuration constants and reusable dataset, post-processing,
and training helper functions.
"""

from .config import SR, N_MELS, N_FFT, HOP_LENGTH, HOP_SEC, AUDIO_EXTS_ORDER
from .data import (
    find_audio_by_stem,
    load_mel,
    labels_to_frame_targets,
    SVDDataset,
    LazySVDDataset,
    collate_pad,
    LazyMTLDataset,
    collate_pad_mtl,
)
from .windowed_data import (
    LazyWindowedMTLDataset,
    WindowTypeBatchSampler,
)
from .postproc import (
    compute_metrics_from_arrays,
    apply_hysteresis_seq,
    compute_hysteresis_metrics,
    grid_search_hysteresis,
)
from .train_utils import (
    set_seed,
    collect_val_outputs,
    compute_pos_weight_for_subset,
)
