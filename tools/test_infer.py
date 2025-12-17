import json
import numpy as np
from pathlib import Path
from time import perf_counter

from cnn_lstm_2task.infer_plot import load_model, run_infer
from common.data import load_mel
from common.config import HOP_SEC, HOP_LENGTH, SR
from common.postproc import apply_hysteresis_seq
from common.export_utils import compute_frame_dur, build_hyst_segments, make_hyst_payload

_t0 = perf_counter()
model, config = load_model(Path(r"D:\meliris\dali636_cnn_lstm_2task_bilstm_inst_epoch-10_lambda2-0.75_energy-0.0_ts-20251211_065804.onnx"))
_t1 = perf_counter()

_t2 = perf_counter()
X = load_mel(Path(r"D:\meliris\music\17さいのうた。.flac"))
_t3 = perf_counter()
T = X.shape[0]
times = np.arange(T) * HOP_SEC

_t4 = perf_counter()
probs_cls, probs_ratio = run_infer(model, X)
_t5 = perf_counter()

print(f"load_model 用时: {(_t1 - _t0)*1000:.2f} ms")
print(f"load_mel 用时:   {(_t3 - _t2)*1000:.2f} ms")
print(f"run_infer 用时:  {(_t5 - _t4)*1000:.2f} ms")
inst_probs = probs_cls

s_on = 0.7
s_off = 0.5

pred_hyst = apply_hysteresis_seq(inst_probs, float(s_on), float(s_off)).astype(np.float32)
export_path = Path(".") / "exports" / "test_infer_inst_pred_hyst.json"
export_path.parent.mkdir(parents=True, exist_ok=True)
frame_dur = compute_frame_dur(times, HOP_SEC)
segments = build_hyst_segments(pred_hyst, times, frame_dur)
payload = make_hyst_payload("test_infer", "test_infer", "test_infer", s_on, s_off, SR, HOP_LENGTH, HOP_SEC, times, pred_hyst, segments)
export_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
print(f"已导出 inst_pred_hyst 到：{export_path}")

