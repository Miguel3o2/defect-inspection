"""Static int8 (QDQ) quantisation. CNNs need static quantisation with calibration data; dynamic quantisation
mostly covers MatMul/Gemm and does little for Conv."""
from __future__ import annotations

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


class ArrayReader(CalibrationDataReader):
    def __init__(self, arrays: np.ndarray, input_name: str, batch_size: int = 16):
        self._it = iter([{input_name: arrays[i:i + batch_size]} for i in range(0, len(arrays), batch_size)])

    def get_next(self):
        return next(self._it, None)


def quantize_onnx(fp32_path: str, int8_path: str, calib: np.ndarray, input_name: str = "input") -> None:
    """calib: float32 [N, 3, H, W] preprocessed TRAIN images (never val/test)."""
    quantize_static(fp32_path, int8_path, ArrayReader(calib, input_name), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
