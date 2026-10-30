from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_models(cfg: dict, seed: int) -> dict:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", random_state=seed, **cfg["logreg"]),
        ),
        "lightgbm": LGBMClassifier(
            class_weight="balanced", random_state=seed, subsample_freq=1, verbose=-1, **cfg["lightgbm"]
        ),
    }
