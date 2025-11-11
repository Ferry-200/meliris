from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_metrics_from_arrays(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = tp / (tp + fp + 1e-8) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def apply_hysteresis_seq(prob_seq: np.ndarray, th_on: float, th_off: float) -> np.ndarray:
    pred = np.zeros_like(prob_seq, dtype=np.float32)
    state = 0.0
    for i, pval in enumerate(prob_seq):
        if state == 0.0 and pval >= th_on:
            state = 1.0
        elif state == 1.0 and pval < th_off:
            state = 0.0
        pred[i] = state
    return pred


def compute_hysteresis_metrics(probs_seqs: List[np.ndarray], y_seqs: List[np.ndarray], th_on: float, th_off: float) -> Dict[str, float]:
    tp = fp = fn = 0
    for prob_seq, y_seq in zip(probs_seqs, y_seqs):
        pred_seq = apply_hysteresis_seq(prob_seq, th_on, th_off)
        tp += int(((pred_seq == 1) & (y_seq == 1)).sum())
        fp += int(((pred_seq == 1) & (y_seq == 0)).sum())
        fn += int(((pred_seq == 0) & (y_seq == 1)).sum())
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def grid_search_hysteresis(
    probs_seqs: List[np.ndarray],
    y_seqs: List[np.ndarray],
    th_on_grid: List[float],
    th_off_grid: List[float],
) -> Tuple[Optional[float], Optional[float], Dict[str, float]]:
    best_on: Optional[float] = None
    best_off: Optional[float] = None
    best_metrics: Dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for th_on in th_on_grid:
        for th_off in th_off_grid:
            if th_on <= th_off:
                continue
            m = compute_hysteresis_metrics(probs_seqs, y_seqs, th_on, th_off)
            if m["f1"] > best_metrics["f1"]:
                best_metrics = m
                best_on, best_off = th_on, th_off
    return best_on, best_off, best_metrics