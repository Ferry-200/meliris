from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from cnn_lstm_2task.model import CNNLSTM2Head
from common.config import N_MELS


def export_pt_to_onnx(ckpt: Path, out: Path | None, dummy_T: int, opset: int, dynamo: bool = False):
    ck = torch.load(str(ckpt), map_location="cpu")
    config = ck.get("config", {})
    bilstm = bool(config.get("bilstm") or (str(config.get("rnn_type", "")).lower() == "bilstm") or config.get("bidirectional"))
    state = ck.get("model_state", ck)
    model = CNNLSTM2Head(input_dim=N_MELS, cnn_channels=64, hidden=128, bidirectional=bilstm).to(torch.device("cpu"))
    model.load_state_dict(state, strict=False)
    model.eval()
    if out is None:
        out = ckpt.with_suffix(".onnx")
    n_mels = int(config.get("n_mels", N_MELS))
    dummy = torch.zeros((1, dummy_T, n_mels), dtype=torch.float32)
    try:
        if dynamo:
            torch.onnx.export(
                model,
                dummy,
                str(out),
                input_names=["input"],
                output_names=["logits_cls", "logits_reg"],
                dynamic_shapes={"input": {1: "T"}, "logits_cls": {1: "T"}, "logits_reg": {1: "T"}},
                opset_version=opset,
                dynamo=True,
            )
        else:
            torch.onnx.export(
                model,
                dummy,
                str(out),
                input_names=["input"],
                output_names=["logits_cls", "logits_reg"],
                dynamic_axes={"input": {1: "T"}, "logits_cls": {1: "T"}, "logits_reg": {1: "T"}},
                opset_version=opset,
                dynamo=False,
            )
        print(f"已导出 ONNX: {out}")

        json.dump(config, open(out.with_suffix(".json"), "w"), indent=4)
    except Exception as e:
        print(f"导出失败：{e}")
        print("若缺少依赖，请安装：python -m pip install onnx onnxscript")
        raise


def main():
    parser = argparse.ArgumentParser(description="将 PyTorch pt 转换为 ONNX")
    parser.add_argument("--ckpt", required=True, help="PyTorch 权重文件路径 (.pt)")
    parser.add_argument("--out", default=None, help="输出 ONNX 路径，默认为同名 .onnx")
    parser.add_argument("--dummy-T", type=int, default=128, help="导出时使用的时间步长度")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset 版本")
    parser.add_argument("--dynamo", action="store_true", help="使用 dynamo 导出")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out) if args.out else None
    export_pt_to_onnx(ckpt_path, out_path, args.dummy_T, args.opset, args.dynamo)


if __name__ == "__main__":
    main()
