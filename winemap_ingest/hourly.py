"""Hourly weather for the disease and frost models (winemap-nl docs/27 step 1, Jasper 28 Oct 2026), next to the daily
grids of climate.py, in the private bucket winemap-climate:

  ERA5-Land hourly (Copernicus, CC BY 4.0), the Netherlands, Belgium and Denmark, April-October from 2000:
    climate/era5land-hourly/nlbedk/<var>/<year>/<year>-<mm>.nc   var = t2m (degC), d2m (dewpoint, degC), rr (mm/h)
    int16 at 0.01, hours since 1950-01-01 UTC; the rain de-accumulated (ERA5-Land sums it from 00 UTC).
  KNMI hourly station observations (KNMI open data), all stations, whole years from 2000:
    climate/knmi-stations/hourly/<year>.csv.gz   stn, time (UTC, end of the hour), T, TD (degC), U (%), RH (mm), DR (h)
    climate/knmi-stations/stations.csv           stn, lon, lat, alt, name

    python -m winemap_ingest.hourly era5 [--minutes 150] [--years NEWEST OLDEST]
    python -m winemap_ingest.hourly stations [--years NEWEST OLDEST]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

from .climate import CDS_URL, FILL, SCALE, climate_store, to_int16

FIRST_YEAR = 2000
MONTHS = range(4, 11)  # April - October: the growing season
AREA = [58.0, 2.4, 49.4, 15.3]  # north, west, south, east: NL, BE and DK
REGION = "nlbedk"
VARS = {"t2m": "2m_temperature", "d2m": "2m_dewpoint_temperature", "rr": "total_precipitation"}
EPOCH = datetime(1950, 1, 1, tzinfo=UTC)
KNMI_HOURLY = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
STATION_VARS = ("T", "TD", "U", "RH", "DR")


def era5_key(var: str, year: int, month: int) -> str:
    return f"climate/era5land-hourly/{REGION}/{var}/{year}/{year}-{month:02d}.nc"


def months_newest_first(first: int, last: date, years: tuple[int, int] | None):
    newest, oldest = years if years else (last.year, first)
    for y in range(newest, oldest - 1, -1):
        for m in reversed(MONTHS):
            if date(y, m, 1) <= last:
                yield y, m


def era5_request(var: str, year: int, month: int) -> dict:
    return {
        "variable": [VARS[var]],
        "year": str(year),
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def deaccumulate(a: np.ndarray, hours: list[datetime]) -> np.ndarray:
    """ERA5-Land rain is a running sum from 00 UTC (the 00 UTC value is the whole day before): the amount per hour."""
    out = np.empty_like(a)
    for i, h in enumerate(hours):
        if h.hour == 1 or i == 0:
            out[i] = a[i] if h.hour == 1 else np.nan  # the first hour of a day is its own sum
        else:
            out[i] = a[i] - a[i - 1]
    return np.where(out < 0, 0, out)  # rounding in the source can give tiny negatives


def era5_pack(var: str, raw: bytes) -> bytes:
    """One CDS month (hourly) -> our int16 month file."""
    with netCDF4.Dataset("era5.nc", memory=raw) as d:
        name = {"t2m": "t2m", "d2m": "d2m", "rr": "tp"}[var]
        a = np.ma.filled(d[name][:].astype(np.float64), np.nan)
        t = d["valid_time"]
        stamps = netCDF4.num2date(t[:], t.units, only_use_cftime_datetimes=False)
        hours = [datetime(s.year, s.month, s.day, s.hour, tzinfo=UTC) for s in stamps]
        lat = np.asarray(d["latitude"][:])
        lon = np.asarray(d["longitude"][:])
    if var == "rr":
        a = deaccumulate(a * 1000.0, hours)  # m -> mm per hour
    else:
        a = a - 273.15
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4") as o:
            o.title = f"ERA5-Land hourly {VARS[var].replace('_', ' ')}"
            o.source = (
                "Contains modified Copernicus Climate Change Service information (ERA5-Land, CC BY 4.0); "
                "repacked for Winemap"
            )
            o.Conventions = "CF-1.8"
            o.createDimension("time", len(hours))
            o.createDimension("latitude", len(lat))
            o.createDimension("longitude", len(lon))
            tv = o.createVariable("time", "i4", ("time",))
            tv.units = "hours since 1950-01-01 00:00:00"
            tv.calendar = "standard"
            tv[:] = [int((h - EPOCH).total_seconds() // 3600) for h in hours]
            for nm, vals, units in (("latitude", lat, "degrees_north"), ("longitude", lon, "degrees_east")):
                c = o.createVariable(nm, "f8", (nm,))
                c.units = units
                c[:] = vals
            v = o.createVariable(
                var,
                "i2",
                ("time", "latitude", "longitude"),
                zlib=True,
                complevel=4,
                shuffle=True,
                chunksizes=(24, min(64, len(lat)), min(64, len(lon))),
                fill_value=FILL,
            )
            v.set_auto_maskandscale(False)
            v.scale_factor = SCALE
            v.add_offset = 0.0
            v.units = "mm" if var == "rr" else "degC"
            v[:] = to_int16(a)
        return path.read_bytes()


def run_era5(store, minutes: float, years=None, log=print) -> int:
    """Request the missing months one after the other (CDS: sequential), newest first."""
    import cdsapi

    client = cdsapi.Client(url=CDS_URL, key=os.environ["CDS_API_KEY"], quiet=True, progress=False)
    stop = time.monotonic() + minutes * 60
    have = set(store.list(f"climate/era5land-hourly/{REGION}/"))
    today = date.today()
    last = today - timedelta(days=7)  # ERA5-Land runs ~5 days behind; a month is fetched once it is complete
    last = date(last.year, last.month, 1) - timedelta(days=1)
    done = failed = 0
    for year, month in months_newest_first(FIRST_YEAR, last, years):
        for var in VARS:
            key = era5_key(var, year, month)
            if key in have:
                continue
            if time.monotonic() > stop:
                return done
            with tempfile.TemporaryDirectory() as tmp:
                part = Path(tmp) / "cds.nc"
                try:
                    client.retrieve("reanalysis-era5-land", era5_request(var, year, month)).download(str(part))
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


# ---------------------------------------------------------------- KNMI stations


def knmi_hourly(start: date, end: date, opener=urllib.request.urlopen) -> str:
    body = urllib.parse.urlencode(
        {"start": f"{start:%Y%m%d}01", "end": f"{end:%Y%m%d}24", "vars": ":".join(STATION_VARS), "stns": "ALL"}
    ).encode()
    req = urllib.request.Request(KNMI_HOURLY, data=body, headers={"User-Agent": "winemap-ingest"})
    with opener(req, timeout=300) as r:
        return r.read().decode("latin-1")


def _val(p: dict, k: str, scale: float = 0.1):
    """One value of a KNMI row: tenths to units; RH -1 means < 0.05 mm (counted as 0)."""
    v = p.get(k, "")
    if v == "":
        return ""
    n = int(v)
    return 0.0 if (k == "RH" and n == -1) else round(n * scale, 2)


def parse_hourly(text: str) -> tuple[list[list], list[list]]:
    """KNMI's text answer -> (stations [stn, lon, lat, alt, name], rows [stn, time, T, TD, U, RH, DR] in degC / mm / h).
    The values come in tenths; RH -1 means < 0.05 mm; the hour HH covers HH-1 .. HH UT."""
    stations, rows, header = [], [], None
    for line in text.splitlines():
        if line.startswith("# ") and line[2:5].strip().isdigit() and len(line.split()) >= 6:
            p = line[2:].split()
            stations.append([int(p[0]), float(p[1]), float(p[2]), float(p[3]), " ".join(p[4:])])
        elif line.startswith("# STN,"):
            header = [h.strip() for h in line[2:].split(",")]
        elif line and not line.startswith("#") and header:
            p = dict(zip(header, (x.strip() for x in line.split(",")), strict=False))
            day = datetime.strptime(p["YYYYMMDD"], "%Y%m%d").replace(tzinfo=UTC)
            t = day + timedelta(hours=int(p["HH"]))

            values = [_val(p, "T"), _val(p, "TD"), _val(p, "U", 1), _val(p, "RH"), _val(p, "DR")]
            if any(v != "" for v in values):  # wind and sea stations measure none of these: left out
                rows.append([int(p["STN"]), t.isoformat(timespec="minutes").replace("+00:00", "Z"), *values])
    return stations, rows


def run_stations(store, years=None, log=print, fetch=knmi_hourly) -> int:
    """Whole years (the running one again each time), month by month."""
    today = date.today()
    newest, oldest = years if years else (today.year, FIRST_YEAR)
    have = set(store.list("climate/knmi-stations/hourly/"))
    done = 0
    stations: list[list] = []
    for year in range(newest, oldest - 1, -1):
        key = f"climate/knmi-stations/hourly/{year}.csv.gz"
        if key in have and year < today.year:
            continue
        rows: list[list] = []
        for m in range(1, 13):
            start = date(year, m, 1)
            if start > today:
                break
            end = (date(year + (m == 12), m % 12 + 1, 1) - timedelta(days=1)) if m < 12 else date(year, 12, 31)
            s, r = parse_hourly(fetch(start, min(end, today)))
            stations = stations or s
            rows += r
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["stn", "time", "T", "TD", "U", "RH", "DR"])
        w.writerows(rows)
        store.put(key, gzip.compress(buf.getvalue().encode()), "application/gzip")
        done += 1
        log(f"ok {key}: {len(rows)} hours")
    if stations:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["stn", "lon", "lat", "alt", "name"])
        w.writerows(stations)
        store.put("climate/knmi-stations/stations.csv", buf.getvalue().encode(), "text/csv")
    return done


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="winemap_ingest.hourly")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("era5")
    e.add_argument("--minutes", type=float, default=150)
    e.add_argument("--years", type=int, nargs=2, metavar=("NEWEST", "OLDEST"))
    s = sub.add_parser("stations")
    s.add_argument("--years", type=int, nargs=2, metavar=("NEWEST", "OLDEST"))
    a = ap.parse_args(argv)
    store = climate_store()
    n = run_era5(store, a.minutes, a.years) if a.cmd == "era5" else run_stations(store, a.years)
    print(f"{n} files written to {store}")


if __name__ == "__main__":
    main()
