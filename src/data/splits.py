import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(y: np.ndarray, val: float, test: float, seed: int):
    """Return index arrays (train, val, test), stratified by label."""
    idx = np.arange(len(y))
    train_val, test_idx = train_test_split(idx, test_size=test, stratify=y, random_state=seed)
    rel_val = val / (1 - test)
    train_idx, val_idx = train_test_split(
        train_val, test_size=rel_val, stratify=y[train_val], random_state=seed
    )
    return train_idx, val_idx, test_idx
