"""Region-of-interest masking.

The A1 camera sees the whole room, not just the bed. Restricting analysis to
the build area (a polygon in normalized image coords) removes background
objects (people, furniture) so they can't cause false positives. The ROI is
static per camera position — recalibrate only if you move the printer or the
print gets tall enough to leave the region.
"""
from __future__ import annotations

import numpy as np


def roi_mask(h: int, w: int, poly: list[list[float]] | None) -> np.ndarray:
    """Boolean mask (h,w), True inside the ROI. None/empty -> all True."""
    if not poly or len(poly) < 3:
        return np.ones((h, w), dtype=bool)
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx + 0.5) / w
    ny = (yy + 0.5) / h
    inside = np.zeros((h, w), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > ny) != (yj > ny)) & (
            nx < (xj - xi) * (ny - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= cond
        j = i
    return inside


def apply_roi(img: np.ndarray, poly: list[list[float]] | None) -> np.ndarray:
    """Zero out pixels outside the ROI (keep the rest). Returns a new array."""
    if img is None or not poly or len(poly) < 3:
        return img
    h, w = img.shape[:2]
    mask = roi_mask(h, w, poly)
    out = img.copy()
    out[~mask] = 0
    return out
