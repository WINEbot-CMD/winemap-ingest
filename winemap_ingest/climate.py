"""Climate data for the BCIs (Winemap docs/24): daily grids, repacked into one small NetCDF per variable per month.

Jasper (27 Oct 2026): everything in the cloud (no laptop), NetCDF per month, private bucket `winemap-climate`.

    climate/knmi/nl/{tg,tn,tx,rr}/<year>/<year>-<mm>.nc       KNMI daily 1 km grids (EPSG:28992), CC BY 4.0
    climate/era5land/eu/{tg,tn,tx,rr}/<year>/<year>-<mm>.nc   ERA5-Land daily, 0.1 deg (EPSG:4326), CC BY 4.0

int16 (time, y, x) with scale 0.01 (degC or mm), fill -32768, zlib + shuffle, chunks of the whole month x 64 x 64:
every reader (GDAL, xarray, netCDF4) opens them without plugins. The time axis holds the real days. For ERA5-Land
rain that is not the file's own month: its 00 UTC value is the total of the day before, so a month file holds the
last day of the previous month up to the second-last day of its month; the BCI step joins the months by date.

Runs (GitHub Actions in this repo, or on the laptop with a folder as the store):
    python -m winemap_ingest.climate knmi [--years 2024 1961] [--minutes 50]
    python -m winemap_ingest.climate era5 [--years 2025 1961] [--minutes 320]
    python -m winemap_ingest.climate era5-world [--years 2025 1991] [--minutes 320] [--parallel 3]   # wine belts
    python -m winemap_ingest.climate pack-knmi <raw dir> <year> [<year> ...]     # the laptop's KNMI files
    python -m winemap_ingest.climate pack-era5 <raw dir> <year> [<year> ...]     # the laptop's ERA5 files
Each run picks up what is missing in the store, newest first, until its time or the request limit is used up.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

from .knmi import Knmi, OutOfRequests
from .store import open_store

SCALE = 0.01
FILL = -32768
EPOCH = date(1950, 1, 1)
VARS = ("tg", "tn", "tx", "rr")
KNMI_DATASETS = {"tg": ("Tg1", "5"), "tn": ("Tn1", "2"), "tx": ("Tx1", "2"), "rr": ("Rd1", "5")}
KNMI_FIRST_YEAR = 1961
ERA5_FIRST_YEAR = 1961
ERA5_AREA = [72.0, -25.0, 27.0, 45.0]  # all of Europe, as pipeline/download_era5land.py (north, west, south, east)
ERA5_STATS = {"tg": "daily_mean", "tn": "daily_minimum", "tx": "daily_maximum"}
# The world's wine belts outside Europe's box, with a wide margin (Jasper, 1 Nov 2026: wine belts first, the rest of the
# world later; 1991-2025, newest first), [north, west, south, east]:
WORLD_AREAS = {
    "na": [62.0, -170.0, 20.0, -50.0],  # North America (Okanagan 50 N .. Baja California, Mexico)
    "as": [62.0, 45.0, 20.0, 150.0],  # Asia east of Europe's box: Central Asia, China, Japan, India from 20 N
    "sa": [-8.0, -82.0, -56.0, -34.0],  # South America (Brazil's São Francisco valley 9 S .. Patagonia)
    "saf": [-8.0, 10.0, -36.0, 42.0],  # southern Africa
    "oc": [-8.0, 110.0, -48.0, 180.0],  # Australia and New Zealand
}
WORLD_FIRST_YEAR = 1991
CDS_URL = "https://cds.climate.copernicus.eu/api"


def key_of(source: str, var: str, year: int, month: int) -> str:
    region = "nl" if source == "knmi" else "eu"
    return f"climate/{source}/{region}/{var}/{year}/{year}-{month:02d}.nc"


def to_int16(a: np.ndarray) -> np.ndarray:
    """Values (NaN = no data) -> int16 at SCALE; clipped to the int16 range (327 degC / mm a day)."""
    q = np.round(np.asarray(a, dtype=np.float64) / SCALE)
    return np.where(np.isfinite(q), np.clip(q, -32767, 32767), FILL).astype(np.int16)


def write_month(
    days: list[date],
    data: np.ndarray,
    var: str,
    coords: dict[str, np.ndarray],
    units: str,
    attrs: dict[str, str],
    grid: dict[str, str],
) -> bytes:
    """One month as NetCDF bytes. coords = {"y": ..., "x": ...} (metres) or {"latitude": ..., "longitude": ...}."""
    (ydim, yv), (xdim, xv) = coords.items()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4") as o:
            for k, v in attrs.items():
                o.setncattr(k, v)
            o.Conventions = "CF-1.8"
            o.createDimension("time", len(days))
            o.createDimension(ydim, len(yv))
            o.createDimension(xdim, len(xv))
            t = o.createVariable("time", "i4", ("time",))
            t.units = f"days since {EPOCH.isoformat()}"
            t.calendar = "standard"
            t[:] = [(d - EPOCH).days for d in days]
            for name, vals in ((ydim, yv), (xdim, xv)):
                c = o.createVariable(name, "f8", (name,))
                c.units = "m" if name in ("x", "y") else ("degrees_north" if name == "latitude" else "degrees_east")
                c[:] = vals
            p = o.createVariable("crs", "i4")
            for k, v in grid.items():
                p.setncattr(k, v)
            v = o.createVariable(
                var,
                "i2",
                ("time", ydim, xdim),
                zlib=True,
                complevel=4,
                shuffle=True,
                chunksizes=(len(days), min(64, len(yv)), min(64, len(xv))),
                fill_value=FILL,
            )
            v.set_auto_maskandscale(False)  # the int16 are written as they are
            v.scale_factor = SCALE
            v.add_offset = 0.0
            v.units = units
            v.grid_mapping = "crs"
            v[:] = to_int16(data)
        return path.read_bytes()


# ---------------------------------------------------------------- KNMI


def knmi_day(name: str) -> date:
    """The day a KNMI file holds, from its name (..._L3__20240101T000000_20240102T000000_0006.nc)."""
    stamp = name.split("__")[-1][:8]
    return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))


def latest_per_day(names: list[str]) -> list[str]:
    """KNMI sometimes has a day twice (an older and a newer version, ..._0002.nc and ..._0006.nc; seen in 2017-2018):
    keep the newest version of each day, in day order."""
    best: dict[date, str] = {}
    for n in sorted(names):  # same day: the names differ only in the version number at the end, so newest sorts last
        best[knmi_day(n)] = n
    return [best[d] for d in sorted(best)]


def knmi_read(raw: bytes) -> tuple[np.ndarray, dict]:
    """One KNMI daily file: the grid (NaN = no data) and its x, y."""
    with netCDF4.Dataset("knmi.nc", memory=raw) as d:
        a = d["prediction"][0]
        grid = np.ma.filled(a.astype(np.float64), np.nan)
        return grid, {"y": np.asarray(d["y"][:]), "x": np.asarray(d["x"][:])}


def knmi_pack(var: str, days: list[date], grids: list[np.ndarray], coords: dict) -> bytes:
    title = {"tg": "mean", "tn": "minimum", "tx": "maximum"}.get(var)
    return write_month(
        days,
        np.stack(grids),
        var,
        coords,
        "degC" if var != "rr" else "mm",
        {
            "title": f"KNMI daily {title} temperature" if title else "KNMI daily precipitation sum",
            "source": "KNMI Data Platform, daily 1 km grids (CC BY 4.0); repacked for Winemap",
        },
        {"epsg_code": "EPSG:28992", "grid_mapping_name": "oblique_stereographic"},
    )


def knmi_month_files(api: Knmi, var: str, year: int, month: int) -> list[str]:
    name, _ = KNMI_DATASETS[var]
    begin = f"INTER_OPER_R___{name.upper()}_____L3__{year}{month:02d}00"  # sorts just before the month's first file
    files = api.list_files(KNMI_DATASETS[var], begin=begin, limit=31)
    return latest_per_day([f for f in files if knmi_day(f).year == year and knmi_day(f).month == month])


def months_newest_first(first_year: int, last: date, years: tuple[int, int] | None):
    y1, y0 = years if years else (last.year, first_year)
    for y in range(y1, y0 - 1, -1):
        for m in range(12, 0, -1):
            if date(y, m, 1) <= last:
                yield y, m


def run_knmi(store, api: Knmi, minutes: float, years=None, log=print) -> int:
    """Pack missing KNMI months (newest first). The month of yesterday is redone until it is complete."""
    stop = time.monotonic() + minutes * 60
    have = set(store.list("climate/knmi/nl/"))
    done = 0
    yesterday = date.today() - timedelta(days=1)
    try:
        for year, month in months_newest_first(KNMI_FIRST_YEAR, yesterday, years):
            for var in VARS:
                key = key_of("knmi", var, year, month)
                recent = (yesterday - date(year, month, 1)).days < 62  # still filling (rain lags a few weeks)
                if key in have and not recent:
                    continue
                if time.monotonic() > stop:
                    return done
                files = knmi_month_files(api, var, year, month)
                if not files or (key in have and recent and len(files) == days_in(store, key)):
                    continue
                grids, coords = [], None
                for f in files:
                    g, coords = knmi_read(api.download(KNMI_DATASETS[var], f))
                    grids.append(g)
                store.put(key, knmi_pack(var, [knmi_day(f) for f in files], grids, coords), "application/x-netcdf")
                done += 1
                log(f"ok {key} ({len(files)} days)")
    except OutOfRequests as e:
        log(f"stopped: {e}")
    return done


def days_in(store, key: str) -> int:
    raw = store.get(key)
    if not raw:
        return 0
    with netCDF4.Dataset("m.nc", memory=raw) as d:
        return len(d.dimensions["time"])


# ---------------------------------------------------------------- ERA5-Land


def era5_requests(year: int, month: int | list[int], area: list[float] = ERA5_AREA):
    """(var, dataset, request) for one month (or several: the world asks a year at a time): three temperature
    statistics and the rain."""
    months = month if isinstance(month, list) else [month]
    base = {
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "area": area,
    }
    for var, stat in ERA5_STATS.items():
        yield (
            var,
            "derived-era5-land-daily-statistics",
            {
                **base,
                "variable": ["2m_temperature"],
                "daily_statistic": stat,
                "time_zone": "utc+00:00",
                "frequency": "1_hourly",
            },
        )
    yield (
        "rr",
        "reanalysis-era5-land",
        {
            **base,
            "variable": ["total_precipitation"],
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
    )


def era5_read(var: str, raw: bytes) -> tuple[np.ndarray, list[date], dict]:
    """A CDS file: the values (time, lat, lon), the day of each step, the coordinates."""
    with netCDF4.Dataset("era5.nc", memory=raw) as d:
        name = "tp" if var == "rr" else "t2m"
        a = np.ma.filled(d[name][:].astype(np.float64), np.nan)
        t = d["valid_time"]
        stamps = netCDF4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
        days = [date(s.year, s.month, s.day) for s in stamps]
        coords = {"latitude": np.asarray(d["latitude"][:]), "longitude": np.asarray(d["longitude"][:])}
    return a, days, coords


def era5_pack(var: str, raw: bytes) -> bytes:
    """One CDS month file -> our month file (degC / mm; rain moved to the day it fell)."""
    return era5_pack_steps(var, *era5_read(var, raw))


def era5_pack_months(var: str, raw: bytes) -> dict[tuple[int, int], bytes]:
    """A CDS file of several months (the world asks a year at a time) -> our month files, each packed exactly as a
    month asked on its own (split on the month of the time steps, before the rain moves a day back)."""
    a, days, coords = era5_read(var, raw)
    out = {}
    for ym in sorted({(d.year, d.month) for d in days}):
        idx = [i for i, d in enumerate(days) if (d.year, d.month) == ym]
        out[ym] = era5_pack_steps(var, a[idx], [days[i] for i in idx], coords)
    return out


def era5_pack_steps(var: str, a: np.ndarray, days: list[date], coords: dict) -> bytes:
    """Steps of one month -> our month file."""
    if var == "rr":
        a = a * 1000.0  # m -> mm
        days = [x - timedelta(days=1) for x in days]  # 00 UTC = the total of the day before
    else:
        a = a - 273.15  # K -> degC
    title = {"tg": "mean", "tn": "minimum", "tx": "maximum"}.get(var)
    return write_month(
        days,
        a,
        var,
        coords,
        "degC" if var != "rr" else "mm",
        {
            "title": f"ERA5-Land daily {title} 2 m temperature" if title else "ERA5-Land daily precipitation",
            "source": "Contains modified Copernicus Climate Change Service information (ERA5-Land, CC BY 4.0); "
            "repacked for Winemap",
        },
        {"epsg_code": "EPSG:4326", "grid_mapping_name": "latitude_longitude"},
    )


def run_era5(store, minutes: float, years=None, log=print) -> int:
    """Request missing ERA5-Land months one after the other (CDS rules: sequential), newest first."""
    import cdsapi

    client = cdsapi.Client(url=CDS_URL, key=os.environ["CDS_API_KEY"], quiet=True, progress=False)
    stop = time.monotonic() + minutes * 60
    have = set(store.list("climate/era5land/eu/"))
    today = date.today()
    last = date(today.year, today.month, 1) - timedelta(days=1)  # ERA5-Land runs a few days behind: last month
    done = failed = 0
    for year, month in months_newest_first(ERA5_FIRST_YEAR, last, years):
        for var, dataset, request in era5_requests(year, month):
            key = key_of("era5land", var, year, month)
            if key in have:
                continue
            if time.monotonic() > stop:
                return done
            with tempfile.TemporaryDirectory() as tmp:
                part = Path(tmp) / "cds.nc"
                try:
                    client.retrieve(dataset, request).download(str(part))
                    store.put(key, era5_pack(var, part.read_bytes()), "application/x-netcdf")
                    done += 1
                    log(f"ok {key}")
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    log(f"FAILED {key}: {e}")
                    if failed >= 5:
                        return done
                    time.sleep(60)
    return done


def world_key(region: str, var: str, year: int, month: int) -> str:
    return f"climate/era5land/{region}/{var}/{year}/{year}-{month:02d}.nc"


def europe_complete(store) -> bool:
    """Whether Europe's ERA5-Land history is in (it goes newest first, so its first month is the last to come)."""
    return all(store.get(key_of("era5land", v, ERA5_FIRST_YEAR, 1)) is not None for v in VARS)


def run_era5_world(store, minutes: float, years=None, parallel: int = 3, log=print, after_europe: bool = True) -> int:
    """The world's wine belts (WORLD_AREAS), newest month first, a few requests at a time (Jasper: in parallel to
    Europe's own job); each request with its own CDS client. Stops after 5 failures; the next run goes on."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    # Jasper, 1 Nov 2026: "Europe first": the world waits until Europe's history is in (one CDS queue for both)
    if after_europe and not europe_complete(store):
        log("Europe's ERA5-Land history is not complete yet: the world waits")
        return 0

    import cdsapi

    stop = time.monotonic() + minutes * 60
    have = set()
    for region in WORLD_AREAS:
        have |= set(store.list(f"climate/era5land/{region}/"))
    today = date.today()
    last = date(today.year, today.month, 1) - timedelta(days=1)
    # one request per area, variable and year (the CDS queue waits about as long for a month as for a year; 1 Nov
    # 2026: ~80 minutes), only the months still missing; stored per month as Europe's
    y1, y0 = years if years else (last.year, WORLD_FIRST_YEAR)
    todo = []
    for year in range(y1, y0 - 1, -1):
        for region in WORLD_AREAS:
            for var in VARS:
                months = [m for m in range(1, 13) if date(year, m, 1) <= last]
                months = [m for m in months if world_key(region, var, year, m) not in have]
                if months:
                    todo.append((region, var, year, months))
    log(f"{len(todo)} world requests to go (a year, area and variable each)")

    def fetch(job) -> str:
        region, var, year, months = job
        _, dataset, request = next(r for r in era5_requests(year, months, WORLD_AREAS[region]) if r[0] == var)
        client = cdsapi.Client(url=CDS_URL, key=os.environ["CDS_API_KEY"], quiet=True, progress=False)
        with tempfile.TemporaryDirectory() as tmp:
            part = Path(tmp) / "cds.nc"
            client.retrieve(dataset, request).download(str(part))
            for (y, m), data in era5_pack_months(var, part.read_bytes()).items():
                if y == year and m in months:
                    store.put(world_key(region, var, y, m), data, "application/x-netcdf")
        return f"{region} {var} {year} ({len(months)} months)"

    done = failed = 0
    jobs = iter(todo)
    with ThreadPoolExecutor(parallel) as ex:
        running = {}
        while True:
            while len(running) < parallel and time.monotonic() < stop and failed < 5:
                job = next(jobs, None)
                if job is None:
                    break
                running[ex.submit(fetch, job)] = f"{job[0]} {job[1]} {job[2]}"
            if not running:
                return done
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                key = running.pop(f)
                try:
                    f.result()
                    done += 1
                    log(f"ok {key}")
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    log(f"FAILED {key}: {e}")


# ---------------------------------------------------------------- the laptop's downloads


def pack_local_knmi(store, raw_dir: Path, years: list[int], log=print) -> int:
    """data/raw/knmi/<Tg1_v5 ...>/<year>/*.nc -> the store; only complete months (all days present)."""
    done = 0
    for year in years:
        for var, (name, version) in KNMI_DATASETS.items():
            folder = raw_dir / f"{name}_v{version}" / str(year)
            by_month: dict[int, list[Path]] = {}
            files = {f.name: f for f in folder.glob("*.nc")}
            for name in latest_per_day(list(files)):
                by_month.setdefault(knmi_day(name).month, []).append(files[name])
            for month, files in sorted(by_month.items()):
                full = (date(year + month // 12, month % 12 + 1, 1) - date(year, month, 1)).days
                if len(files) < full:
                    log(f"skip {var} {year}-{month:02d}: {len(files)}/{full} days")
                    continue
                grids, coords = [], None
                for f in files:
                    g, coords = knmi_read(f.read_bytes())
                    grids.append(g)
                key = key_of("knmi", var, year, month)
                store.put(key, knmi_pack(var, [knmi_day(f.name) for f in files], grids, coords), "application/x-netcdf")
                done += 1
                log(f"ok {key}")
    return done


def pack_local_era5(store, raw_dir: Path, years: list[int], log=print) -> int:
    """data/raw/era5land/eu/<var>/<year>/<var>_<year>-<mm>.nc -> the store."""
    done = 0
    for year in years:
        for var in VARS:
            for f in sorted((raw_dir / var / str(year)).glob(f"{var}_{year}-*.nc")):
                month = int(f.stem.rsplit("-", 1)[1])
                key = key_of("era5land", var, year, month)
                store.put(key, era5_pack(var, f.read_bytes()), "application/x-netcdf")
                done += 1
                log(f"ok {key}")
    return done


SOURCES = {
    "knmi": ("KNMI", "Daily grids NL, 1 km (tg, tn, tx, rr)", KNMI_FIRST_YEAR),
    "era5land": ("Copernicus (ECMWF)", "ERA5-Land Europe, 0.1° (tg, tn, tx, rr)", ERA5_FIRST_YEAR),
    "era5land-world": ("Copernicus (ECMWF)", "ERA5-Land world wine belts, 0.1° (tg, tn, tx, rr)", WORLD_FIRST_YEAR),
}


def progress(store, source: str, result: str, today: date | None = None) -> dict:
    """Months in the store against the months that exist (1961 to last month), for the status page."""
    today = today or date.today()
    org, title, first = SOURCES[source]
    if source == "era5land-world":
        keys = [k for r in WORLD_AREAS for k in store.list(f"climate/era5land/{r}/")]
        per_month = len(VARS) * len(WORLD_AREAS)
    else:
        region = "nl" if source == "knmi" else "eu"
        keys = store.list(f"climate/{source}/{region}/")
        per_month = len(VARS)
    months = sorted({k.rsplit("/", 1)[-1][:7] for k in keys})
    total = ((today.year - first) * 12 + today.month - 1) * per_month  # up to and including last month
    return {
        "org": org,
        "title": title,
        "files": len(keys),
        "total": total,
        "newest": months[-1] if months else None,
        "oldest": months[0] if months else None,
        "lastRun": datetime.now(UTC).isoformat(timespec="seconds"),
        "lastResult": result,
    }


def report(source: str | Callable[[], tuple[str, dict]], store=None, result: str = "", log=print) -> None:
    """monitor/climate.json in winemap-wx (read by /api/status). Never fails the run.

    `source` is "knmi"/"era5land", or for other downloads (projections.py) a function giving (key, entry)."""
    try:
        from .store import R2Store

        if os.environ.get("CLIMATE_STORE", "r2") != "r2" or not os.environ.get("R2_ACCESS_KEY_ID"):
            return
        key, entry = source() if callable(source) else (source, progress(store, source, result))
        wx = R2Store(bucket="winemap-wx")
        state = json.loads(wx.get("monitor/climate.json") or b"{}")
        state[key] = entry
        state["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
        wx.put("monitor/climate.json", json.dumps(state, indent=1).encode(), "application/json")
    except Exception as e:  # noqa: BLE001
        log(f"status not written: {e}")


def climate_store():
    """The private bucket (R2_CLIMATE_* credentials), or a folder for trial runs (CLIMATE_STORE=<folder>)."""
    where = os.environ.get("CLIMATE_STORE", "r2")
    if where != "r2":
        return open_store(where)
    from .store import R2Store

    return R2Store(bucket=os.environ.get("R2_CLIMATE_BUCKET", "winemap-climate"), creds="R2_CLIMATE")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="winemap_ingest.climate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, default in (("knmi", 50), ("era5", 320)):
        p = sub.add_parser(name)
        p.add_argument("--years", type=int, nargs=2, metavar=("NEWEST", "OLDEST"))
        p.add_argument("--minutes", type=float, default=default)
    p = sub.add_parser("era5-world")
    p.add_argument("--years", type=int, nargs=2, metavar=("NEWEST", "OLDEST"))
    p.add_argument("--minutes", type=float, default=320)
    p.add_argument("--parallel", type=int, default=3)
    for name in ("pack-knmi", "pack-era5"):
        p = sub.add_parser(name)
        p.add_argument("raw", type=Path)
        p.add_argument("years", type=int, nargs="+")
    a = ap.parse_args(argv)
    store = climate_store()
    if a.cmd == "knmi":
        api = Knmi(keep=50)  # KNMI_API_KEY; the workflow passes the download key, not the one of the rain jobs
        n = run_knmi(store, api, a.minutes, a.years)
        report("knmi", store, f"{n} months written")
    elif a.cmd == "era5":
        n = run_era5(store, a.minutes, a.years)
        report("era5land", store, f"{n} months written")
    elif a.cmd == "era5-world":
        n = run_era5_world(store, a.minutes, a.years, a.parallel)
        report("era5land-world", store, f"{n} months written")
    elif a.cmd == "pack-knmi":
        n = pack_local_knmi(store, a.raw, a.years)
    else:
        n = pack_local_era5(store, a.raw, a.years)
    print(f"{n} month files written to {store}")


if __name__ == "__main__":
    main()
