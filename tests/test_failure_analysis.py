import cv2
import numpy as np
import pandas as pd

from src.evaluate.failure_analysis import add_hints, overview


def _write(tmp_path, name, mean):
    p = tmp_path / name
    cv2.imwrite(str(p), np.full((200, 200), mean, dtype=np.uint8))
    return str(p)


def test_overview_counts_confusions(tmp_path):
    rows = [("a", "a")] * 8 + [("a", "b")] * 2 + [("b", "b")] * 10
    df = pd.DataFrame({"path": [_write(tmp_path, f"{i}.png", 100) for i in range(len(rows))],
                       "true": [r[0] for r in rows], "pred": [r[1] for r in rows], "confidence": 0.95})
    o = overview(df)
    assert o["n_err"] == 2 and o["recall"]["a"] == 0.8
    assert o["pairs"].iloc[0]["count"] == 2


def test_hints_flag_only_errors_and_high_confidence(tmp_path):
    rng = np.random.default_rng(0)
    paths, rows = [], []
    for i in range(12):
        p = tmp_path / f"ok{i}.png"
        cv2.imwrite(str(p), rng.normal(120, 10, (200, 200)).clip(0, 255).astype(np.uint8))
        paths.append(str(p))
        rows.append(("a", "a", 0.9))
    paths.append(_write(tmp_path, "err.png", 20))
    rows.append(("a", "b", 0.97))
    df = add_hints(pd.DataFrame({"path": paths, "true": [r[0] for r in rows],
                                 "pred": [r[1] for r in rows], "confidence": [r[2] for r in rows]}))
    assert (df.loc[df["correct"], "auto_hint"] == "").all()
    hint = df.loc[~df["correct"], "auto_hint"].iloc[0]
    assert "dark" in hint and "label_noise_suspect" in hint
