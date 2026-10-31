"""Climate projections (Winemap docs/38, Jasper 31 Oct 2026): the World Bank's CMIP6 climatologies, world + Europe.

Source: Climate Change Knowledge Portal (CCKP), CMIP6 0.25 deg, bias-adjusted and downscaled, per model: the monthly
mean of a 20-year period (s3://wbg-cckp, public; CC BY 4.0 - "The World Bank: CCKP, CMIP6 0.25 degree"). Ready-made
period means, so ~6 GB for the world instead of ~590 GB of daily files (NEX-GDDP-CMIP6, chunks of the whole world per
day). Jasper chose 8 models within the IPCC "likely" climate sensitivity (ECS 2.5-4 degC, TCR 1.4-2.2 degC) for the
ensemble, and EC-Earth3 (the model behind KNMI'23, slightly warmer) on its own for comparison.

    climate/cckp/world/<model>/<scenario>_<period>/<var>.nc   the source file as it is (whole world, lon 0..360)
    climate/cckp/eu/<model>/<scenario>_<period>/<var>.nc      Europe cut, float32 (12 months, lat, lon), zlib, ~1.4 MB

Jasper kept the world too (for the wine belts later; ~6.2 GB, R2 a few cents a month above the free 10 GB).
The status page shows the progress (monitor/climate.json "cckp", like the KNMI and ERA5 downloads).

Variables: tas, tasmax, tasmin (degC), pr (mm per month), and the monthly counts tr (Tmin > 20 degC), hd35
(Tmax > 35 degC), fd (Tmin < 0 degC) - enough for the season BCIs (winemap-compute projections step).
Box: Europe as in the Europe view (lon -25..45, lat 34..72).

    python -m winemap_ingest.projections [--only <model>]       # in this repo's workflow, or with WX_STORE=<folder>
"""

from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import netCDF4
import numpy as np

from .climate import report
from .store import R2Store, open_store

BASE = "https://wbg-cckp.s3.amazonaws.com/data/cmip6-x0.25"
ENSEMBLE = [
    "mpi-esm1-2-hr-r1i1p1f1",
    "mri-esm2-0-r1i1p1f1",
    "gfdl-esm4-r1i1p1f1",
    "miroc6-r1i1p1f1",
    "noresm2-lm-r1i1p1f1",
    "bcc-csm2-mr-r1i1p1f1",
    "access-esm1-5-r1i1p1f1",
    "cmcc-esm2-r1i1p1f1",
]
COMPARE = ["ec-earth3-r1i1p1f1"]
RUNS = [("historical", "1995-2014")] + [(s, p) for s in ("ssp245", "ssp585") for p in ("2040-2059", "2080-2099")]
VARS = ["tas", "tasmax", "tasmin", "pr", "tr", "hd35", "fd"]
LON0, LAT0, LON1, LAT1 = -25.0, 34.0, 45.0, 72.0
FILL = np.float32(1e20)


def url(var: str, model: str, scenario: str, period: str) -> str:
    run = f"{model}-{scenario}"
    return f"{BASE}/{var}/{run}/climatology-{var}-monthly-mean_cmip6-x0.25_{run}_climatology_mean_{period}.nc"


def europe(src: Path, var: str) -> bytes:
    """The Europe box of one global file (longitudes 0..360 there) as a small NetCDF, west to east."""
    with netCDF4.Dataset(src) as d:
        lat, lon = d["lat"][:], d["lon"][:]
        name = next(k for k in d.variables if k.startswith("climatology-"))
        lon180 = np.where(lon > 180, lon - 360, lon)
        order = np.argsort(lon180)
        lon_sorted = lon180[order]
        cols = order[(lon_sorted >= LON0) & (lon_sorted <= LON1)]
        rows = np.where((lat >= LAT0) & (lat <= LAT1))[0]
        data = d[name][:, rows, :][:, :, cols].astype("float32")
        units = getattr(d[name], "units", "")
        out_lat, out_lon = lat[rows], lon180[cols]
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "eu.nc"
        with netCDF4.Dataset(path, "w") as o:
            o.createDimension("month", 12)
            o.createDimension("lat", len(out_lat))
            o.createDimension("lon", len(out_lon))
            o.createVariable("lat", "f8", ("lat",))[:] = out_lat
            o.createVariable("lon", "f8", ("lon",))[:] = out_lon
            v = o.createVariable(var, "f4", ("month", "lat", "lon"), zlib=True, complevel=6, fill_value=FILL)
            v[:] = np.ma.filled(data, FILL)
            v.units = units
            o.source = "World Bank Climate Change Knowledge Portal (CCKP), CMIP6 0.25 degree; CC BY 4.0"
        return path.read_bytes()


def keys(model: str, scenario: str, period: str, var: str) -> tuple[str, str]:
    part = f"{model}/{scenario}_{period}/{var}.nc"
    return f"climate/cckp/world/{part}", f"climate/cckp/eu/{part}"


def progress(store, models: list[str], result: str) -> dict:
    """For the status page: files in the bucket against all files of this download (world + Europe)."""
    have = set(store.list("climate/cckp/"))
    want = [k for m in models for s, p in RUNS for v in VARS for k in keys(m, s, p, v)]
    world = sum(k in have for k in want if "/world/" in k)
    eu = sum(k in have for k in want if "/eu/" in k)
    return {
        "org": "World Bank (CCKP)",
        "title": f"CMIP6 projections 0.25°, {len(models)} models, world + Europe",
        "files": world + eu,
        "total": len(want),
        "unit": "files",
        "detail": f"world {world} / {len(want) // 2} · Europe {eu} / {len(want) // 2}",
        "manual": True,  # started by hand: no 'stale' warning when it has not run for a day
        "lastRun": datetime.now(UTC).isoformat(timespec="seconds"),
        "lastResult": result,
    }


def run(store, models: list[str], log=print, every: int = 10) -> int:
    """Download what is missing (skips what is there); reports progress every `every` files."""
    done = set(store.list("climate/cckp/"))
    todo = [(m, s, p, v) for m in models for s, p in RUNS for v in VARS if not set(keys(m, s, p, v)) <= done]
    n = 0

    def status(result: str) -> None:
        report(lambda: ("cckp", progress(store, models, result)), log=log)

    status(f"running: 0 of {len(todo)} done")
    for model, scenario, period, var in todo:
        world, eu = keys(model, scenario, period, var)
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "global.nc"
            try:
                urllib.request.urlretrieve(url(var, model, scenario, period), src)
            except Exception as e:  # noqa: BLE001
                log(f"missing at the source: {model} {scenario} {period} {var} ({e})")
                continue
            if world not in done:
                store.put(world, src.read_bytes(), "application/x-netcdf")
            if eu not in done:
                store.put(eu, europe(src, var), "application/x-netcdf")
        n += 1
        log(eu)
        if n % every == 0:
            status(f"running: {n} of {len(todo)} done")
    status(f"{n} written" if n == len(todo) else f"{n} of {len(todo)} written")
    return n


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--only", help="one model id (default: the ensemble and EC-Earth3)")
    args = a.parse_args()
    where = os.environ.get("WX_STORE", "r2")  # a folder for a local try-out
    store = open_store(where) if where != "r2" else R2Store(bucket="winemap-climate", creds="R2_CLIMATE")
    models = [args.only] if args.only else ENSEMBLE + COMPARE
    print(f"{run(store, models)} files written")


if __name__ == "__main__":
    main()
