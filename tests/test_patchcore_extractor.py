import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.anomaly.patchcore_extractor import PatchCoreExtractor


@pytest.mark.parametrize("backbone,dim", [("wide_resnet50_2", 1024), ("resnet18", 384)])
def test_extractor_output_shape(backbone, dim):
    ext = PatchCoreExtractor(backbone, out_dim=1024, pretrained=False)
    feats, hw = ext(torch.randn(2, 3, 224, 224))
    assert hw == (28, 28) and feats.shape == (2, 784, dim)
