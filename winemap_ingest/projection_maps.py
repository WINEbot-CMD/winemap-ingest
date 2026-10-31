"""Maps of the climate projections for the viewer (Winemap docs/38, Jasper 31 Oct 2026).

In : projections/cckp/eu/<model>/<scenario>.nc   (winemap-climate; projections.py)
     bci/era5land/eu/<BCI>/<year>.tif           (winemap-wx; the measured ERA5-Land seasons, winemap-compute)
Out: bci/cckp/eu/<scenario>/<method>/<model>/   (winemap-wx, /wx/ on the site), the layout of the measured years:
         <BCI>/<year>.tif      the year's value, float32 COG on the ERA5-Land grid (0.1 degrees), NaN = sea
         <BCI>/cum/<year>.tif  running sums from 2025, for the mean of any period
         <BCI>/series.tif      every year as a band (the series at a point)
         index.json            years, ranges, region means

Only the years from 2026 (Jasper: "no years before 2026 for the projections, so it is clear they are projections").
Two methods, to compare (Jasper: "do both"; docs/38 §8):
    delta  the measured level (ERA5-Land 2019-2025) + the models' change against their own 2019-2025; rain
           (GSP) as a ratio, counts kept at or above 0. Keeps the 0.1 degree detail, no step after the measured years.
    raw    the projections as the models give them (0.25 degrees, drawn as blocks on the 0.1 degree grid).
Models: "ensemble" = the mean of the 8 within the IPCC likely climate sensitivity; "ec-earth3" on its own.

    python -m winemap_ingest.projection_maps [--scenario ssp245] [--only EHI]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path

import netCDF4
import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from .projections import COMPARE, ENSEMBLE, SCENARIOS
from .store import R2Store, open_store

BCIS = ["EHI", "HI", "GST", "GSP", "EFD", "EHD", "TN"]
COUNTS = {"EFD", "EHD", "TN"}  # days: never below 0
RATIO = {"GSP"}  # rain: the change as a ratio
BASE = range(2019, 2026)  # the measured years the delta method starts from
FIRST = 2026  # the first year shown
DECIMALS = {"GST": 2}  # the rest: 1 (as winemap-compute)
COG = dict(compress="DEFLATE", predictor=3, blocksize=256, overview_resampling="nearest", BIGTIFF="NO")
METHODS = ["delta", "raw"]
MODELS = {"ensemble": ENSEMBLE, "ec-earth3": COMPARE}
TITLES = {
    "EHI": ("Huglin index, extended (Apr–Oct)", "°C·d", "sum"),
    "HI": ("Huglin index, standard (Apr–Sep)", "°C·d", "sum"),
    "GST": ("Growing season temperature", "°C", "mean"),
    "GSP": ("Growing season precipitation", "mm", "sum"),
    "EFD": ("Early frost days (15 Apr – 31 May)", "days", "count"),
    "EHD": ("Extreme heat days", "days", "count"),
    "TN": ("Tropical nights", "days", "count"),
}


class Grid:
    def __init__(self, crs, transform, width, height):
        self.crs, self.transform, self.width, self.height = crs, transform, width, height


def cog_bytes(a: np.ndarray, grid: Grid, tags: dict | None = None, count: int = 1, series: bool = False) -> bytes:
    """float32 COG, written in one copy step (the COG driver is copy-only); `series`: all bands in 16 x 16 blocks."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "a.tif", Path(tmp) / "b.tif"
        prof = dict(driver="GTiff", width=grid.width, height=grid.height, count=count, dtype="float32", crs=grid.crs,
                    transform=grid.transform, nodata=np.nan, interleave="pixel")  # fmt: skip
        with rasterio.open(src, "w", **prof) as o:
            o.write(a.astype("float32") if count > 1 else a.astype("float32")[None])
            if tags:
                o.update_tags(**{k: str(v) for k, v in tags.items()})
        if series:
            rio_copy(src, dst, driver="COG", **{**COG, "blocksize": 16, "overviews": "NONE"})
        else:
            rio_copy(src, dst, driver="COG", **COG)
        return dst.read_bytes()


def read_tif(raw: bytes) -> tuple[np.ndarray, Grid]:
    with rasterio.MemoryFile(raw) as m, m.open() as r:
        return r.read(1).astype("f8"), Grid(r.crs.to_string(), r.transform, r.width, r.height)


def model_years(store, model: str, scenario: str, bci: str) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    """(years, lat, lon) of one model run, the years, and its axes (lat south to north)."""
    with netCDF4.Dataset("p.nc", memory=store.get(f"projections/cckp/eu/{model}/{scenario}.nc")) as d:
        years = [int(y) for y in d["year"][:]]
        return np.ma.filled(d[bci][:].astype("f8"), np.nan), years, d["lat"][:].astype("f8"), d["lon"][:].astype("f8")


def to_grid(a: np.ndarray, lat: np.ndarray, lon: np.ndarray, grid: Grid, how: Resampling) -> np.ndarray:
    """A (…, lat, lon) stack at 0.25 degrees onto the 0.1 degree grid of the measured maps."""
    res = float(lon[1] - lon[0])
    src_t = from_origin(lon[0] - res / 2, lat[-1] + res / 2, res, res)  # north-up: flip the rows
    flat = a.reshape(-1, len(lat), len(lon))[:, ::-1]
    out = np.full((flat.shape[0], grid.height, grid.width), np.nan)
    for i in range(flat.shape[0]):
        reproject(flat[i], out[i], src_transform=src_t, src_crs="EPSG:4326", dst_transform=grid.transform,
                  dst_crs=grid.crs, resampling=how, src_nodata=np.nan, dst_nodata=np.nan)  # fmt: skip
    return out.reshape(a.shape[:-2] + (grid.height, grid.width))


def delta(base_meas: np.ndarray, change: np.ndarray, bci: str) -> np.ndarray:
    """The measured level + the models' change (a ratio for rain); counts not below 0."""
    out = base_meas[None] * change if bci in RATIO else base_meas[None] + change
    return np.maximum(out, 0.0) if bci in COUNTS else out


def maps_for(store_clim, store_wx, scenario: str, bci: str, log=print) -> dict[tuple[str, str], tuple]:
    """(method, model) -> (years from 2026, lat, lon) on the measured grid."""
    meas = [read_tif(store_wx.get(f"bci/era5land/eu/{bci}/{y}.tif")) for y in BASE]
    grid = meas[0][1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_meas = np.nanmean([m[0] for m in meas], 0)
    land = np.isfinite(base_meas)
    out = {}
    for name, models in MODELS.items():
        runs = []
        for m in models:
            a, years, lat, lon = model_years(store_clim, m, scenario, bci)
            runs.append(a)
        stack = np.stack(runs)  # (models, years, lat, lon)
        yi = [years.index(y) for y in BASE]
        keep = [i for i, y in enumerate(years) if y >= FIRST]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            base_model = np.nanmean(stack[:, yi], 1, keepdims=True)  # each model's own 2019-2025
            if bci in RATIO:
                change = np.nanmean(stack[:, keep] / np.where(base_model > 0, base_model, np.nan), 0)
            else:
                change = np.nanmean(stack[:, keep] - base_model, 0)
            raw = np.nanmean(stack[:, keep], 0)
        d = delta(base_meas, to_grid(change, lat, lon, grid, Resampling.bilinear), bci)
        r = to_grid(raw, lat, lon, grid, Resampling.nearest)
        for method, v in (("delta", d), ("raw", r)):
            v = np.where(land[None], v, np.nan)
            out[(method, name)] = (np.round(v, DECIMALS.get(bci, 1)), grid)
        log(f"{scenario} {bci} {name}: {len(keep)} years")
    return out


def write(store_wx, scenario: str, method: str, model: str, bci: str, maps: np.ndarray, grid: Grid, years: list[int]):
    """Year files, running sums and the series, like winemap-compute's measured layers. Returns the index entry."""
    base = f"bci/cckp/eu/{scenario}/{method}/{model}/{bci}"
    cum = np.where(np.isfinite(maps[0]), 0.0, np.nan)
    store_wx.put(f"{base}/cum/{years[0] - 1}.tif", cog_bytes(cum, grid, {"bci": bci}), "image/tiff")
    for i, y in enumerate(years):
        store_wx.put(f"{base}/{y}.tif", cog_bytes(maps[i], grid, {"bci": bci, "year": y}), "image/tiff")
        cum = cum + np.nan_to_num(maps[i])
        store_wx.put(f"{base}/cum/{y}.tif", cog_bytes(cum, grid, {"bci": bci, "sum_to": y}), "image/tiff")
    series = cog_bytes(maps, grid, {"years": ",".join(map(str, years))}, len(years), series=True)
    store_wx.put(f"{base}/series.tif", series, "image/tiff")
    title, unit, kind = TITLES[bci]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            "title": title,
            "unit": unit,
            "kind": kind,
            "years": years,
            "range": [round(float(np.nanpercentile(maps, 2)), 2), round(float(np.nanpercentile(maps, 98)), 2)],
            "means": {str(y): round(float(np.nanmean(maps[i])), 2) for i, y in enumerate(years)},
            "counts": False,
        }


def run(store_clim, store_wx, scenarios: list[str], bcis: list[str], log=print) -> int:
    n = 0
    for scenario in scenarios:
        index = {(method, model): {} for method in METHODS for model in MODELS}
        grid = None
        for bci in bcis:
            maps = maps_for(store_clim, store_wx, scenario, bci, log)
            for (method, model), (v, grid) in maps.items():
                years = list(range(FIRST, FIRST + v.shape[0]))
                index[(method, model)][bci] = write(store_wx, scenario, method, model, bci, v, grid, years)
                n += 1
        for (method, model), entries in index.items():
            doc = {
                "version": 1,
                "source": "cckp",
                "region": "eu",
                "scenario": scenario,
                "method": method,
                "model": model,
                "models": MODELS[model],
                "base": [BASE[0], BASE[-1]],
                "formulas": "winemap-nl docs/12_bci_formulas.md, from monthly values (winemap-ingest projections.py)",
                "period": [2041, 2060],
                "generated": datetime.now(UTC).isoformat(timespec="seconds"),
                "grid": {
                    "crs": grid.crs,
                    "x0": grid.transform.c,
                    "y0": grid.transform.f,
                    "res": grid.transform.a,
                    "width": grid.width,
                    "height": grid.height,
                },  # fmt: skip
                "bcis": entries,
            }
            key = f"bci/cckp/eu/{scenario}/{method}/{model}/index.json"
            old = json.loads(store_wx.get(key) or b"{}") if set(entries) != set(BCIS) else {}
            doc["bcis"] = {**old.get("bcis", {}), **entries}  # --only: keep the other BCIs
            store_wx.put(key, json.dumps(doc, indent=1, ensure_ascii=False).encode(), "application/json")
    return n


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--scenario", choices=SCENARIOS, help="one scenario (default: both)")
    a.add_argument("--only", choices=BCIS, help="one BCI (default: all)")
    args = a.parse_args()
    where = os.environ.get("WX_STORE", "r2")  # a folder for a local try-out: both stores in it
    if where != "r2":
        clim = wx = open_store(where)
    else:
        clim, wx = R2Store(bucket="winemap-climate", creds="R2_CLIMATE"), R2Store(bucket="winemap-wx")
    n = run(clim, wx, [args.scenario] if args.scenario else SCENARIOS, [args.only] if args.only else BCIS)
    print(f"{n} layers written")


if __name__ == "__main__":
    main()
