
from __future__ import annotations

import hashlib
import re
import sys
import warnings
from pathlib import Path

import numpy as np

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
PER_CLASS = 300


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


_ALIASES = {_norm(c): c for c in CLASSES}
_ALIASES.update({"cr": "crazing", "crack": "crazing", "cracks": "crazing", "in": "inclusion",
                 "pa": "patches", "patch": "patches", "ps": "pitted_surface", "pitted": "pitted_surface",
                 "rs": "rolled-in_scale", "sc": "scratches", "scratch": "scratches"})
_STEM = re.compile(r"^(.*?)[\s_\-]*\d+$")


def label_from_name(path: Path) -> str | None:
    m = _STEM.match(path.stem)
    for cand in ([m.group(1)] if m else []) + [path.parent.name]:
        label = _ALIASES.get(_norm(cand))
        if label:
            return label
    return None


def discover_images(data_dir: str | Path) -> tuple[list[Path], np.ndarray]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Download the NEU images (1800 grayscale images, 300 per class; the image-only "
            f"'NEU-CLS' version is enough) and put them under {root}. See RUNBOOK.md, Stage 2.")
    paths, labels = [], []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT and (lab := label_from_name(p)):
            paths.append(p)
            labels.append(CLASSES.index(lab))
    if not paths:
        raise RuntimeError(f"No recognisable NEU images under {root} (class must appear in the file or folder name)")
    y = np.array(labels)
    if np.bincount(y, minlength=len(CLASSES)).max() > PER_CLASS:
        warnings.warn(f"A class has more than {PER_CLASS} images: did you copy the dataset twice "
                      f"(e.g. NEU-CLS and NEU-DET together)? Duplicates across splits inflate every result.",
                      stacklevel=2)
    return paths, y


def check(data_dir: str | Path) -> int:
    """Print class counts and exact-duplicate files; return 0 if it looks like the standard 1800-image set."""
    paths, y = discover_images(data_dir)
    counts = np.bincount(y, minlength=len(CLASSES))
    print(f"{len(paths)} images (expected {PER_CLASS * len(CLASSES)})")
    for c, n in zip(CLASSES, counts, strict=True):
        print(f"  {c:16s} {n:4d}{'' if n == PER_CLASS else f'   <- expected {PER_CLASS}'}")
    seen: dict[str, Path] = {}
    dupes = []
    for p in paths:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            dupes.append((p, seen[h]))
        seen[h] = seen.get(h, p)
    print(f"exact duplicate files: {len(dupes)}")
    for a, b in dupes[:5]:
        print(f"  {a}  ==  {b}")
    ok = len(paths) == PER_CLASS * len(CLASSES) and bool((counts == PER_CLASS).all()) and not dupes
    print("OK" if ok else "CHECK THE ABOVE before training: counts or duplicates differ from the standard dataset")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check(sys.argv[1] if len(sys.argv) > 1 else "data/raw/NEU-DET"))