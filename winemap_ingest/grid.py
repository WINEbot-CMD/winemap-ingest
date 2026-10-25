"""KNMI radar grid (polar stereographic, 1 km, 700 x 765) -> the viewer's RD New 1 km grid.

Nearest neighbour ("honest blocks", like the thesis layers): every RD cell takes the radar cell under its centre.
The index arrays are computed once; resampling a field is then a plain numpy lookup.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import Transformer

# radar grid (the same in the real-time 5-minute and the climatological hourly files)
RADAR_PROJ = "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378137 +b=6356752 +x_0=0 +y_0=0 +units=m +no_defs"
RADAR_W, RADAR_H = 700, 765
RADAR_X0, RADAR_Y0 = 0.0, -3_650_000.0  # upper-left corner, m
RADAR_CELL = 1000.0

# output grid: RD New, 1 km, the Netherlands with a margin (x 0-300 km, y 290-630 km)
RD_CRS = "EPSG:28992"
CELL = 1000.0
X0, Y1 = 0.0, 630_000.0  # upper-left corner
WIDTH, HEIGHT = 300, 340
TRANSFORM = (CELL, 0.0, X0, 0.0, -CELL, Y1)  # GDAL order: a, b, c, d, e, f (rasterio Affine)


@lru_cache(maxsize=1)
def lookup() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radar row / column for every RD cell centre, and a mask of cells inside the radar image."""
    xs = X0 + (np.arange(WIDTH) + 0.5) * CELL
    ys = Y1 - (np.arange(HEIGHT) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    to_radar = Transformer.from_crs(RD_CRS, RADAR_PROJ, always_xy=True)
    rx, ry = to_radar.transform(gx, gy)
    col = np.floor((rx - RADAR_X0) / RADAR_CELL).astype(int)
    row = np.floor((RADAR_Y0 - ry) / RADAR_CELL).astype(int)
    inside = (col >= 0) & (col < RADAR_W) & (row >= 0) & (row < RADAR_H)
    return np.clip(row, 0, RADAR_H - 1), np.clip(col, 0, RADAR_W - 1), inside


def to_rd(field: np.ndarray) -> np.ndarray:
    """A radar field (765 x 700, float, NaN = no data) on the RD grid (340 x 300)."""
    if field.shape != (RADAR_H, RADAR_W):
        raise ValueError(f"radar field has shape {field.shape}, expected {(RADAR_H, RADAR_W)}")
    row, col, inside = lookup()
    out = field[row, col].astype(np.float32)
    out[~inside] = np.nan
    return out
