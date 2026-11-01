"""Climate projections per year (Winemap docs/38, Jasper 31 Oct 2026): the World Bank's CMIP6 monthly time series.

Source: Climate Change Knowledge Portal (CCKP), CMIP6 0.25 deg, downscaled and bias-corrected to ERA5 (CC BY 4.0 -
"The World Bank: CCKP, CMIP6 0.25 degree"), one 4.3 GB file per variable, model and scenario for the world, monthly
2015-2100. Only the future (Jasper: the measured years come from ERA5); 2015-2025 are kept to check the hand-over to
the measured years. Only the rows of the region and the season months are read (remote_nc.py), and only the BCIs per
year are stored:

    projections/cckp/<region>/<model>/<scenario>.nc     int16 (year, lat, lon) per BCI, scaled, zlib

Models: 8 within the IPCC "likely" climate sensitivity (ECS 2.5-4 degC, TCR 1.4-2.2 degC) + EC-Earth3 (the model
behind KNMI'23) for comparison; all 9 kept apart (Jasper). Scenarios SSP2-4.5 and SSP5-8.5. Regions: Europe, and the
world's wine belts with a wide margin (Jasper, 1 Nov 2026), a northern band 20-62 N and a southern band 8-56 S, all
the way round. The Southern Hemisphere's season is October - April: its year is the vintage, the year the season ends
(Oct 2025 - Apr 2026 = 2026); Huglin October - April (standard October - March), EFD 15 October - 30 November.

BCIs from monthly values, the same definitions as winemap-compute bci.py (season 1 Apr - 31 Oct, Tgem = (Tmax +
Tmin) / 2, Huglin k per latitude): Huglin per month = days x max(0, (Tgem + Tmax) / 2 - 10) x k (measured on ERA5:
-0.9 % against daily values, docs/38); GST the day-weighted mean of Tgem; GSP the sum of the monthly rain; EHD and TN
the monthly counts hd35 (Tmax > 35) and tr (Tmin > 20); EFD (15 Apr - 31 May) only approximated: half of April's
frost days + May's.

    python -m winemap_ingest.projections [--model <id>] [--region eu]
"""

from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import netCDF4
import numpy as np

from .climate import report
from .remote_nc import RemoteGrid
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
MODELS = ENSEMBLE + COMPARE
SCENARIOS = ["ssp245", "ssp585"]
VARS = ["tasmax", "tasmin", "pr", "hd35", "tr", "fd"]
FIRST_YEAR, LAST_YEAR = 2015, 2100
# region -> ((lon0, lat0, lon1, lat1), hemisphere); Europe as in the Europe view, the world's wine belts with a margin
# (Okanagan 50 N and southern Sweden 58 N in the north; Brazil's São Francisco valley 9 S to Central Otago 45 S)
REGIONS = {
    "eu": ((-25.0, 34.0, 45.0, 72.0), "N"),
    "world_n": ((-180.0, 20.0, 180.0, 62.0), "N"),
    "world_s": ((-180.0, -56.0, 180.0, -8.0), "S"),
}
# the season per hemisphere: its months in order and their days (February as 28.25)
SEASONS = {
    "N": ([4, 5, 6, 7, 8, 9, 10], np.array([30, 31, 30, 31, 31, 30, 31], dtype="f8")),  # April - October
    "S": ([10, 11, 12, 1, 2, 3, 4], np.array([31, 30, 31, 31, 28.25, 31, 30], dtype="f8")),  # October - April
}
MONTHS, DAYS = SEASONS["N"]
# id -> the scale for int16 (the scales of winemap-compute's curves)
BCIS = {"EHI": 10, "HI": 10, "GST": 100, "GSP": 10, "EFD": 100, "EHD": 100, "TN": 100}
FILL = np.int16(-32768)


def url(var: str, model: str, scenario: str) -> str:
    run = f"{model}-{scenario}"
    return f"{BASE}/{var}/{run}/timeseries-{var}-monthly-mean_cmip6-x0.25_{run}_timeseries_mean_{FIRST_YEAR}-2100.nc"


def key(region: str, model: str, scenario: str) -> str:
    return f"projections/cckp/{region}/{model}/{scenario}.nc"


def huglin_k(lat: np.ndarray) -> np.ndarray:
    """winemap-compute bci.huglin_k: 1.02 below 42 degrees, +0.01 per 2 degrees, 1.06 from 48."""
    steps = np.clip(np.floor((np.abs(np.asarray(lat, dtype=float)) - 40) / 2), 0, 4)
    return 1.02 + 0.01 * steps


def window(lat: np.ndarray, lon: np.ndarray, box) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Rows r0..r1 and the columns (west to east; longitudes -180..180 or 0..360) of a box."""
    lon0, lat0, lon1, lat1 = box
    rows = np.where((lat >= lat0) & (lat <= lat1))[0]
    lon180 = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon180, kind="stable")
    cols = order[(lon180[order] >= lon0) & (lon180[order] <= lon1)]
    return int(rows[0]), int(rows[-1]) + 1, cols, lat[rows], lon180[cols]


def season_steps(hemisphere: str = "N") -> list[int]:
    """The time steps (months since January FIRST_YEAR) of every year's season; -1 where it starts before the data
    (the Southern Hemisphere's first season, October - December of the year before)."""
    months, _ = SEASONS[hemisphere]
    out = []
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        for m in months:
            year = y - 1 if hemisphere == "S" and m >= 10 else y  # the vintage: the season ends in its year
            out.append((year - FIRST_YEAR) * 12 + m - 1 if year >= FIRST_YEAR else -1)
    return out


def read_var(grid, box, threads: int = 16, hemisphere: str = "N") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(years, months, lat, lon) float32 of one variable: the season months of every year, the box only."""
    r0, r1, cols, lat, lon = window(grid.lat, grid.lon, box)
    steps = season_steps(hemisphere)
    n_months = len(SEASONS[hemisphere][0])

    def slab(t: int) -> np.ndarray:
        return grid.rows(t, r0, r1)[:, cols] if t >= 0 else np.full((r1 - r0, len(cols)), np.nan, "f4")

    with ThreadPoolExecutor(threads) as ex:
        slabs = list(ex.map(slab, steps))
    return np.stack(slabs).reshape(len(steps) // n_months, n_months, r1 - r0, len(cols)), lat, lon


def bcis(v: dict[str, np.ndarray], lat: np.ndarray, hemisphere: str = "N") -> dict[str, np.ndarray]:
    """The BCIs per year (year, lat, lon) from the season months (year, month, lat, lon). In the south the months run
    October - April, so the same positions hold: the first six for HI, the first two for EFD."""
    tx, tn = v["tasmax"], v["tasmin"]
    tg = (tx + tn) / 2
    season_days = SEASONS[hemisphere][1]
    days = season_days[None, :, None, None]
    k = huglin_k(lat)[None, :, None]
    hug = days * np.maximum(0.0, (tg + tx) / 2 - 10)
    return {
        "EHI": hug.sum(1) * k,
        "HI": hug[:, :6].sum(1) * k,  # April - September (south: October - March)
        "GST": (tg * days).sum(1) / season_days.sum(),
        "GSP": v["pr"].sum(1),
        "EFD": 0.5 * v["fd"][:, 0] + v["fd"][:, 1],
        "EHD": v["hd35"].sum(1),
        "TN": v["tr"].sum(1),
    }


def to_nc(
    out: dict[str, np.ndarray], lat: np.ndarray, lon: np.ndarray, model: str, scenario: str, hemisphere: str = "N"
) -> bytes:
    n_years = next(iter(out.values())).shape[0]
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "p.nc"
        with netCDF4.Dataset(path, "w") as o:
            o.createDimension("year", n_years)
            o.createDimension("lat", len(lat))
            o.createDimension("lon", len(lon))
            o.createVariable("year", "i2", ("year",))[:] = np.arange(FIRST_YEAR, FIRST_YEAR + n_years)
            o.createVariable("lat", "f8", ("lat",))[:] = lat
            o.createVariable("lon", "f8", ("lon",))[:] = lon
            for name, scale in BCIS.items():
                a = out[name] * scale
                q = np.where(np.isfinite(a), np.clip(np.round(a), -32767, 32767), FILL).astype("i2")
                v = o.createVariable(name, "i2", ("year", "lat", "lon"), zlib=True, complevel=6, fill_value=FILL)
                v.set_auto_maskandscale(False)
                v[:] = q
                v.scale_factor = 1 / scale
            o.EFD_note = "approximated from monthly frost-day counts: " + (
                "0.5 x October + November" if hemisphere == "S" else "0.5 x April + May"
            )
            o.season = "October - April, the year is the vintage" if hemisphere == "S" else "April - October"
            o.model, o.scenario = model, scenario
            o.source = "World Bank Climate Change Knowledge Portal (CCKP), CMIP6 0.25 degree time series; CC BY 4.0"
            o.created = datetime.now(UTC).isoformat(timespec="seconds")
        return path.read_bytes()


def progress(store, region: str, result: str) -> dict:
    """For the status page: model runs (model x scenario) done against all."""
    have = set(store.list(f"projections/cckp/{region}/"))
    want = [key(region, m, s) for m in MODELS for s in SCENARIOS]
    done = [k for k in want if k in have]
    by = {s: sum(k.endswith(f"/{s}.nc") for k in done) for s in SCENARIOS}
    return {
        "org": "World Bank (CCKP)",
        "title": f"CMIP6 projections per year 2015–2100, 0.25°, {len(MODELS)} models ({region})",
        "files": len(done),
        "total": len(want),
        "unit": "model runs",
        "detail": " · ".join(f"{s.upper()} {n} / {len(MODELS)}" for s, n in by.items()),
        "manual": True,  # started by hand: no 'stale' warning when it has not run for a day
        "lastRun": datetime.now(UTC).isoformat(timespec="seconds"),
        "lastResult": result,
    }


def run(store, models: list[str], region: str = "eu", log=print, open_grid=RemoteGrid) -> int:
    """Each missing model run: read the six variables, compute the BCIs, store one file. Returns the runs written."""
    box, hemisphere = REGIONS[region]
    have = set(store.list(f"projections/cckp/{region}/"))

    def status(result: str) -> None:
        report(lambda: (f"cckp-{region}", progress(store, region, result)), log=log)

    n = 0
    for model in models:
        for scenario in SCENARIOS:
            if key(region, model, scenario) in have:
                continue
            v = {}
            for i, var in enumerate(VARS, 1):
                status(f"running: {model.split('-r1')[0]} {scenario} {var} ({i}/{len(VARS)})")
                v[var], lat, lon = read_var(open_grid(url(var, model, scenario)), box, hemisphere=hemisphere)
                log(f"{model} {scenario} {var}: {v[var].shape}")
            raw = to_nc(bcis(v, lat, hemisphere), lat, lon, model, scenario, hemisphere)
            store.put(key(region, model, scenario), raw, "application/x-netcdf")
            n += 1
            log(f"written {key(region, model, scenario)}")
    status(f"{n} model runs written")
    return n


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--model", help="one model id (default: all 9)")
    a.add_argument("--region", default="eu", choices=sorted(REGIONS))
    args = a.parse_args()
    where = os.environ.get("WX_STORE", "r2")  # a folder for a local try-out
    store = open_store(where) if where != "r2" else R2Store(bucket="winemap-climate", creds="R2_CLIMATE")
    print(f"{run(store, [args.model] if args.model else MODELS, args.region)} model runs written")


if __name__ == "__main__":
    main()
