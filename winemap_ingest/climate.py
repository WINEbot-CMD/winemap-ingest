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
    return [f for f in files if knmi_day(f).year == year and knmi_day(f).month == month]


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


def era5_requests(year: int, month: int):
    """(var, dataset, request) for one month: three temperature statistics and the rain."""
    base = {"year": str(year), "month": [f"{month:02d}"], "day": [f"{d:02d}" for d in range(1, 32)], "area": ERA5_AREA}
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


def era5_pack(var: str, raw: bytes) -> bytes:
    """One CDS month file -> our month file (degC / mm; rain moved to the day it fell)."""
    with netCDF4.Dataset("era5.nc", memory=raw) as d:
        name = "tp" if var == "rr" else "t2m"
        a = np.ma.filled(d[name][:].astype(np.float64), np.nan)
        t = d["valid_time"]
        stamps = netCDF4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
        days = [date(s.year, s.month, s.day) for s in stamps]
        coords = {"latitude": np.asarray(d["latitude"][:]), "longitude": np.asarray(d["longitude"][:])}
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


# ---------------------------------------------------------------- the laptop's downloads


def pack_local_knmi(store, raw_dir: Path, years: list[int], log=print) -> int:
    """data/raw/knmi/<Tg1_v5 ...>/<year>/*.nc -> the store; only complete months (all days present)."""
    done = 0
    for year in years:
        for var, (name, version) in KNMI_DATASETS.items():
            folder = raw_dir / f"{name}_v{version}" / str(year)
            by_month: dict[int, list[Path]] = {}
            for f in sorted(folder.glob("*.nc")):
                by_month.setdefault(knmi_day(f.name).month, []).append(f)
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
}


def progress(store, source: str, result: str, today: date | None = None) -> dict:
    """Months in the store against the months that exist (1961 to last month), for the status page."""
    today = today or date.today()
    org, title, first = SOURCES[source]
    region = "nl" if source == "knmi" else "eu"
    keys = store.list(f"climate/{source}/{region}/")
    months = sorted({k.rsplit("/", 1)[-1][:7] for k in keys})
    total = ((today.year - first) * 12 + today.month - 1) * len(VARS)  # up to and including last month
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


def report(source: str, store, result: str, log=print) -> None:
    """monitor/climate.json in winemap-wx (read by /api/status). Never fails the run."""
    try:
        from .store import R2Store

        if os.environ.get("CLIMATE_STORE", "r2") != "r2" or not os.environ.get("R2_ACCESS_KEY_ID"):
            return
        wx = R2Store(bucket="winemap-wx")
        state = json.loads(wx.get("monitor/climate.json") or b"{}")
        state[source] = progress(store, source, result)
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
    elif a.cmd == "pack-knmi":
        n = pack_local_knmi(store, a.raw, a.years)
    else:
        n = pack_local_era5(store, a.raw, a.years)
    print(f"{n} month files written to {store}")


if __name__ == "__main__":
    main()
