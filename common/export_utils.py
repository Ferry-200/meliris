from __future__ import annotations


def compute_frame_dur(times, fallback_sec: float) -> float:
    """Compute frame duration from consecutive time stamps; fallback if unavailable."""
    try:
        if len(times) > 1:
            return float(times[1] - times[0])
    except Exception:
        pass
    return float(fallback_sec)


def build_hyst_segments(pred_binary, times, frame_dur: float):
    """Merge consecutive 1s into [start_sec, end_sec) segments based on times and frame_dur."""
    segments = []
    in_seg = False
    start_t = 0.0
    # Ensure scalar ints
    seq = [int(x) for x in pred_binary.tolist()]
    for i, v in enumerate(seq):
        if v == 1 and not in_seg:
            in_seg = True
            start_t = float(times[i])
        elif v == 0 and in_seg:
            segments.append({"start_sec": start_t, "end_sec": float(times[i])})
            in_seg = False
    if in_seg:
        end_t = float((times[-1] if len(times) > 0 else 0.0) + frame_dur)
        segments.append({"start_sec": start_t, "end_sec": end_t})
    return segments


def make_hyst_payload(
    stem,
    audio_path,
    mode,
    th_on,
    th_off,
    sr,
    hop_length,
    hop_sec,
    times,
    pred_hyst,
    segments,
):
    """Build a unified JSON payload for hysteresis export."""
    return {
        "stem": stem,
        "audio_path": str(audio_path),
        "mode": mode,
        "postproc": "hysteresis",
        "th_on": float(th_on),
        "th_off": float(th_off),
        "sr": int(sr),
        "hop_length": int(hop_length),
        "hop_sec": float(hop_sec),
        # "times_sec": times.tolist(),
        # "inst_pred_hyst": [int(x) for x in pred_hyst.tolist()],
        "inst_pred_hyst_segments": segments,
    }

