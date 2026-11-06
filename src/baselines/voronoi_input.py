"""Deterministic, modality-wise nearest-sensor completion of a sparse raster.

Only rasterised observations are read. Masks stay sparse; an absent modality
stays identically zero. Rasterisation resolves collisions before completion
(the existing callers use the last observation at a cell). Equal-distance ties
choose the lexicographically first (x, y) cell, independently of sensor order.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=128)
def _nearest_map(nx, ny, cells, lengths, periodic_x, cell_centred):
    cells = np.asarray(cells, dtype=np.int64)
    # Distances use cell centres for RB and endpoint grids for cases 1 and 2.
    # The common half-cell shift cancels in differences.
    sx = lengths[0] / (nx if cell_centred else max(nx - 1, 1))
    sy = lengths[1] / (ny if cell_centred else max(ny - 1, 1))
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    dx = np.abs(xx[..., None] - cells[:, 0]) * sx
    if periodic_x:
        dx = np.minimum(dx, lengths[0] - dx)
    dy = (yy[..., None] - cells[:, 1]) * sy
    nearest = np.argmin(dx * dx + dy * dy, axis=-1)
    return (cells[nearest, 0] * ny + cells[nearest, 1]).reshape(-1)


def complete_voronoi(image, n_modalities, *, lengths=(1.0, 1.0), periodic_x=False,
                     cell_centred=False):
    """Return a filled copy of (batch, value+mask channels, x, y).

    Every member of a batch must share its layout, as in the training scripts.
    Physical lengths matter: RB uses (4, 1) with horizontal periodicity.
    """
    image = np.asarray(image)
    if image.ndim != 4 or image.shape[1] != 2 * n_modalities:
        raise ValueError("expected batch x (2 * modalities) x x-grid x y-grid")
    out = image.copy()
    if not len(image):
        return out
    nx, ny = image.shape[-2:]
    lengths = tuple(float(x) for x in lengths)
    if len(lengths) != 2 or min(lengths) <= 0:
        raise ValueError("physical domain lengths must be positive")
    for modality in range(n_modalities):
        mask = image[:, n_modalities + modality] != 0
        if not np.all(mask == mask[0]):
            raise ValueError("all examples in a raster batch must share the layout")
        cells = tuple(map(tuple, np.argwhere(mask[0])))
        if not cells:
            out[:, modality] = 0
            out[:, n_modalities + modality] = 0
            continue
        nearest = _nearest_map(nx, ny, cells, lengths, bool(periodic_x), bool(cell_centred))
        out[:, modality] = image[:, modality].reshape(len(image), -1)[:, nearest].reshape(len(image), nx, ny)
    return out


def restore_pod(state, representation_type):
    """Restore saved POD statistics without fitting on evaluation data."""
    rep = representation_type(int(state["rank"]))
    for name, value in state.items():
        setattr(rep, name, value)
    return rep
