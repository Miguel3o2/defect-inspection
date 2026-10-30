import cv2
import numpy as np
import pytest
import yaml

from src.data.mvtec import discover
from src.evaluate.anomaly_metrics import paired_bootstrap_diff, tpr_at_fpr
from src.evaluate.unseen_defect_comparison import (
    load_patchcore_scores,
    make_splits,
    run_category,
)
from src.evaluate.unseen_defect_comparison import (
    main as compare_main,
)
from src.train.train_patchcore import main as patchcore_main


def _fake_extract(batch):
    b = len(batch)
    cells = batch.reshape(b, 3, 28, 8, 28, 8)
    f = np.concatenate([cells.mean((3, 5)), cells.std((3, 5))], axis=1)
    return f.transpose(0, 2, 3, 1).reshape(b, 784, 6).astype(np.float32), (28, 28)


def _normal(rng):
    yy, xx = np.mgrid[0:256, 0:256]
    base = np.stack([100 + yy * 0.1, 130 + xx * 0.05, 90 + 0 * yy], -1)
    return (base + rng.normal(0, 6, base.shape)).clip(0, 255).astype(np.uint8)


@pytest.fixture
def root(tmp_path):
    rng = np.random.default_rng(0)
    base = tmp_path / "mvtec" / "widget"
    types = {"scratch": 90, "dent": 50, "stain": -90}
    for d in ["train/good", "test/good", *[f"test/{t}" for t in types], *[f"ground_truth/{t}" for t in types]]:
        (base / d).mkdir(parents=True)
    for i in range(20):
        cv2.imwrite(str(base / f"train/good/{i:03d}.png"), _normal(rng))
    for i in range(10):
        cv2.imwrite(str(base / f"test/good/{i:03d}.png"), _normal(rng))
    for t, boost in types.items():
        for i in range(8):
            img, mask = _normal(rng), np.zeros((256, 256), np.uint8)
            y, x = (int(v) for v in rng.integers(50, 180, 2))
            img[y:y + 24, x:x + 24] = np.clip(img[y:y + 24, x:x + 24].astype(int) + boost, 0, 255)
            mask[y:y + 24, x:x + 24] = 255
            cv2.imwrite(str(base / f"test/{t}/{i:03d}.png"), img)
            cv2.imwrite(str(base / f"ground_truth/{t}/{i:03d}_mask.png"), mask)
    return tmp_path / "mvtec"


def _cfg(root, tmp_path):
    return {"data_root": str(root), "categories": ["widget"], "out_dir": str(tmp_path / "cmp"),
            "patchcore_dir": str(tmp_path / "pc"), "seed": 0, "resize": 256, "crop": 224,
            "supervised": {"models": ["handcrafted_lgbm"], "seeds": [0]},
            "bootstrap": {"n": 50, "fpr": 0.05},
            "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'm.db'}", "experiment": "t"}}


def test_make_splits_disjoint_and_complete(root):
    cat = discover(root, "widget")
    good, types, htr, hev = make_splits(cat, 0)
    assert types == ["dent", "scratch", "stain"] and len(good) == 10
    for t in types:
        assert not set(htr[t]) & set(hev[t])
        assert len(htr[t]) + len(hev[t]) == 8
    assert make_splits(cat, 0)[2]["dent"].tolist() == htr["dent"].tolist()


def test_no_evaluation_image_is_ever_a_supervised_training_image(root, tmp_path):
    cat = discover(root, "widget")
    rng = np.random.default_rng(0)
    pc = rng.random(len(cat.test))
    calls = []

    def spy(train_paths, y, all_paths, seed, cfg):
        calls.append(set(train_paths))
        return rng.random(len(all_paths))

    rows = run_category(_cfg(root, tmp_path), cat, pc, {"spy": spy})
    assert len(calls) == 3 and len(rows) == 6
    held = {t: {str(p) for p, d in zip(cat.test, cat.test_defect, strict=True) if d == t} for t in set(cat.test_defect)}
    good_test = held["good"]
    ordered = sorted(set(cat.test_defect) - {"good"})
    for train, t in zip(calls, ordered, strict=True):
        assert not train & held[t]
        assert not train & good_test
        assert all(str(p) in train for p in cat.train)


def test_metric_helpers():
    y = np.array([0] * 20 + [1] * 20, bool)
    perfect, noise = np.r_[np.zeros(20), np.ones(20)], np.random.default_rng(0).random(40)
    assert tpr_at_fpr(y, perfect, 0.05) == 1.0
    obs, lo, hi = paired_bootstrap_diff(y, perfect, noise, n=200)
    assert obs > 0.2 and lo < obs < hi and lo > 0


def test_end_to_end(root, tmp_path):
    pc_cfg = {"data_root": str(root), "categories": ["widget"], "out_dir": str(tmp_path / "pc"), "seed": 0,
              "backbone": "wide_resnet50_2", "pretrained": False, "out_dim": 1024, "resize": 256, "crop": 224,
              "batch_size": 8, "coreset_ratio": 0.1, "coreset_proj_dim": 128, "calib_fraction": 0.1, "blur_sigma": 4,
              "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'm.db'}", "experiment": "t"}}
    (tmp_path / "pc.yaml").write_text(yaml.safe_dump(pc_cfg))
    patchcore_main(str(tmp_path / "pc.yaml"), extract_fn=_fake_extract)
    assert len(load_patchcore_scores(tmp_path / "pc", discover(root, "widget"))) == 34

    (tmp_path / "cmp.yaml").write_text(yaml.safe_dump(_cfg(root, tmp_path)))
    rows = compare_main(str(tmp_path / "cmp.yaml"))
    unseen = [r for r in rows if r["kind"] == "unseen"]
    assert len(unseen) == 3 and all(r["patchcore_auroc"] > 0.9 for r in unseen)
    assert "Unseen defect types" in (tmp_path / "cmp" / "results.md").read_text()
