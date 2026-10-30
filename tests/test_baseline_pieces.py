import numpy as np

from src.data.leakage import cross_split_duplicates
from src.data.splits import stratified_split
from src.features.handcrafted import extract


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (200, 200), dtype=np.uint8)


def test_feature_vector_is_fixed_length_and_finite():
    a, b = extract(_img(0)), extract(_img(1))
    assert a.shape == b.shape and np.isfinite(a).all()


def test_split_is_disjoint_and_stratified():
    y = np.repeat(np.arange(6), 100)
    tr, va, te = stratified_split(y, 0.15, 0.15, seed=0)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(y)
    assert np.bincount(y[te]).tolist() == [15] * 6


def test_leakage_detects_identical_images():
    import imagehash
    from PIL import Image
    im = Image.fromarray(_img(3))
    h = imagehash.phash(im)
    assert cross_split_duplicates([h], [h, imagehash.phash(Image.fromarray(_img(9)))])[0][:2] == (0, 0)
