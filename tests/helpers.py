"""Shared test helpers: tiny ONNX models built without torch."""
import numpy as np
from onnx import TensorProto, helper, numpy_helper

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SIZE = 32


def _gap_gemm(path, W: np.ndarray, B: np.ndarray, size: int) -> None:
    nodes = [helper.make_node("GlobalAveragePool", ["input"], ["gap"]),
             helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
             helper.make_node("Gemm", ["flat", "W", "B"], ["logits"], transB=1)]
    graph = helper.make_graph(
        nodes, "stub", [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(B)])],
        [numpy_helper.from_array(W.astype(np.float32), "W"), numpy_helper.from_array(B.astype(np.float32), "B")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


def make_stub_onnx(path, size: int = SIZE) -> None:
    """logit_i = k_i * (sum of channel means): dark images -> class 0, bright -> class 5."""
    k = np.array([-2, -1, 0, 1, 2, 3], np.float32)
    _gap_gemm(path, np.repeat(k[:, None], 3, axis=1), np.zeros(6), size)


def make_texture_stub_onnx(path, centers: np.ndarray, size: int = 200) -> None:
    """A brightness-based classifier: predicts the class whose centre is nearest to s = sum of channel means.
    logit_i = c_i * s - c_i^2 / 2 = -(s - c_i)^2 / 2 + const, so argmax is the nearest centre."""
    c = np.asarray(centers, np.float32)
    _gap_gemm(path, np.repeat(c[:, None], 3, axis=1), -(c ** 2) / 2, size)
