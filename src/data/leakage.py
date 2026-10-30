"""Near-duplicate check between splits using perceptual hashing."""
from __future__ import annotations

import imagehash
from PIL import Image


def phash_all(paths) -> list:
    return [imagehash.phash(Image.open(p).convert("L")) for p in paths]


def cross_split_duplicates(hashes_a, hashes_b, max_distance: int = 4) -> list[tuple[int, int, int]]:
    """Pairs (i in a, j in b, hamming distance) with distance <= max_distance. O(n*m) is fine at this scale."""
    hits = []
    for i, ha in enumerate(hashes_a):
        for j, hb in enumerate(hashes_b):
            d = ha - hb
            if d <= max_distance:
                hits.append((i, j, int(d)))
    return hits
