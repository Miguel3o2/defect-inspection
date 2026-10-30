import json

import cv2
import numpy as np
import pytest
import yaml

from src.anomaly.patchcore_core import greedy_coreset, nn_distances, patch_scores_to_map
from src.data.mvtec import discover, load_image, load_mask
from src.evaluate.anomaly_metrics import (
    per_defect_auroc,
    pixel_auroc,
    threshold_metrics,
)
from src.train.train_patchcore import main


def test_nn_distances_matches_bruteforce():
    rng = np.random.default_rng(0)
    Q, B = rng.normal(size=(50, 16)).astype(np.float32), rng.normal(size=(200, 16)).astype(np.float32)
    brute = np.sqrt(((Q[:, None] - B[None]) ** 2).sum(-1)).min(1)
    assert np.allclose(nn_distances(Q, B, chunk=7), brute, atol=1e-3)


def test_coreset_beats_random_coverage():
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(0, 1, (900, 8)), rng.normal(8, 0.2, (20, 8))]).astype(np.float32)  # rare cluster
    sel = greedy_coreset(X, 50, proj_dim=8, seed=0)
    assert len(set(sel.tolist())) == 50
    cover = nn_distances(X, X[sel]).max()
    rand = nn_distances(X, X[rng.choice(len(X), 50, replace=False)]).max()
    assert cover < rand                      # greedy k-center keeps the rare cluster covered
    assert (np.abs(X[sel][:, 0] - 8) < 3).any()


def test_map_shape_and_peak_location():
    s = np.zeros((28, 28), np.float32)
    s[7, 21] = 5.0
    m = patch_scores_to_map(s, 224, sigma=4)
    assert m.shape == (224, 224)
    y, x = np.unravel_index(m.argmax(), m.shape)
    assert abs(y - 7 * 8) < 12 and abs(x - 21 * 8) < 12


def test_metrics():
    masks = np.zeros((2, 4, 4), bool)
    masks[1, :2] = True
    maps = np.where(masks, 1.0, 0.0)
    assert pixel_auroc(masks, maps) == 1.0
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.15])
    d = ["good", "good", "a", "a", "b"]
    assert per_defect_auroc(d, s) == {"a": 1.0, "b": 0.5}
    m = threshold_metrics(np.array([False, False, True, True, True]), s, 0.5)
    assert m["precision"] == 1.0 and m["fpr"] == 0.0 and abs(m["recall"] - 2 / 3) < 1e-9


# ---------- end-to-end on a synthetic MVTec-like dataset with a fake (numpy) feature extractor ----------
def _fake_extract(batch):
    b = len(batch)
    cells = batch.reshape(b, 3, 28, 8, 28, 8)
    f = np.concatenate([cells.mean((3, 5)), cells.std((3, 5))], axis=1)          # [B, 6, 28, 28]
    return f.transpose(0, 2, 3, 1).reshape(b, 784, 6).astype(np.float32), (28, 28)


def _normal(rng):
    yy, xx = np.mgrid[0:256, 0:256]
    base = np.stack([100 + yy * 0.1, 130 + xx * 0.05, 90 + 0 * yy], -1)
    return (base + rng.normal(0, 6, base.shape)).clip(0, 255).astype(np.uint8)


def _blob(img, mask, rng, boost):
    y, x = (int(v) for v in rng.integers(50, 180, 2))
    img[y:y + 24, x:x + 24] = np.clip(img[y:y + 24, x:x + 24].astype(int) + boost, 0, 255)
    mask[y:y + 24, x:x + 24] = 255


@pytest.fixture
def synthetic_root(tmp_path):
    rng = np.random.default_rng(0)
    base = tmp_path / "mvtec" / "widget"
    for d in ("train/good", "test/good", "test/scratch", "test/dent", "ground_truth/scratch", "ground_truth/dent"):
        (base / d).mkdir(parents=True)
    for i in range(20):
        cv2.imwrite(str(base / f"train/good/{i:03d}.png"), _normal(rng))
    for i in range(8):
        cv2.imwrite(str(base / f"test/good/{i:03d}.png"), _normal(rng))
    for name, boost, n in (("scratch", 90, 8), ("dent", 50, 6)):
        for i in range(n):
            img, mask = _normal(rng), np.zeros((256, 256), np.uint8)
            _blob(img, mask, rng, boost)
            cv2.imwrite(str(base / f"test/{name}/{i:03d}.png"), img)
            cv2.imwrite(str(base / f"ground_truth/{name}/{i:03d}_mask.png"), mask)
    return tmp_path / "mvtec"


def test_discover_and_loaders(synthetic_root):
    cat = discover(synthetic_root, "widget")
    assert len(cat.train) == 20 and len(cat.test) == 22
    assert set(cat.test_defect) == {"good", "scratch", "dent"}
    assert load_image(cat.train[0]).shape == (3, 224, 224)
    assert load_mask(None).sum() == 0 and load_mask(cat.test_mask[-1]).sum() > 0


def test_discover_fails_on_missing_mask(synthetic_root):
    (synthetic_root / "widget/ground_truth/dent/000_mask.png").unlink()
    with pytest.raises(FileNotFoundError):
        discover(synthetic_root, "widget")


def test_end_to_end_pipeline(synthetic_root, tmp_path):
    cfg = {"data_root": str(synthetic_root), "categories": ["widget"], "out_dir": str(tmp_path / "out"), "seed": 0,
           "backbone": "wide_resnet50_2", "pretrained": False, "out_dim": 1024, "resize": 256, "crop": 224,
           "batch_size": 8, "coreset_ratio": 0.1, "coreset_proj_dim": 128, "calib_fraction": 0.1, "blur_sigma": 4,
           "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}", "experiment": "test"}}
    cfg_path = tmp_path / "pc.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    (res,) = main(str(cfg_path), extract_fn=_fake_extract)
    assert res["image_auroc"] > 0.95 and res["pixel_auroc"] > 0.95
    assert set(res["per_defect_image_auroc"]) == {"scratch", "dent"}
    cdir = tmp_path / "out" / "widget"
    assert (cdir / "bank.npy").exists() and (cdir / "overlays.png").exists()
    meta = json.loads((cdir / "meta.json").read_text())
    assert meta["threshold"] == max(meta["cal_scores"]) and meta["hw"] == [28, 28]
    assert (tmp_path / "out" / "results.md").read_text().startswith("| Category")
