import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import numpy as np

from src.data.augment import build_transform
from src.models.classifier import build_classifier
from src.serve.preprocess import to_tensor


def test_serving_preprocess_matches_training_transform():
    from torchvision import tv_tensors

    gray = np.random.default_rng(0).integers(0, 256, (200, 200), dtype=np.uint8)
    train_side = build_transform("none", train=False)(tv_tensors.Image(torch.from_numpy(gray)[None])).numpy()
    assert np.allclose(train_side, to_tensor(gray), atol=1e-6)


def test_export_roundtrip(tmp_path):
    from src.deploy.export_onnx import export

    net = build_classifier("resnet18", 6, pretrained=False)
    ck = tmp_path / "ck.pt"
    torch.save({"state_dict": net.state_dict(), "arch": "resnet18", "classes": list("abcdef"), "image_size": 64}, ck)
    out = export(str(ck), str(tmp_path / "models"), quantize=False)
    assert (out / "model.onnx").exists() and (out / "meta.json").exists()
