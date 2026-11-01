"""The weather forecast for Europe (Winemap backlog 1.6, docs/38 §10; Jasper 1 Nov 2026: ICON-EU only, frost nights,
temperature, rain): DWD's ICON-EU open data (6.5 km, 0-120 h; DWD open data, commercial use allowed with the source
named), copied into winemap-wx as COGs the viewer reads. The 00 and 12 UTC runs, twice a day.

    fc/eu/<run>/T2M/<YYYYMMDDTHHMM>.tif    2 m temperature (°C) at that time: hourly to +78 h, then every 3 hours
    fc/eu/<run>/RAIN/<YYYYMMDDTHHMM>.tif   rain (mm/h) averaged over the step that ends at that time
    fc/eu/<run>/TMIN/<YYYYMMDD>.tif        the night's lowest 2 m temperature (°C), 18 UTC the evening before to 12 UTC
                                           that morning (the dawn of all of Europe; DWD's minimum blocks fit it)
    fc/eu/index.json                       the run shown, its times and nights, the grid

A new run goes into its own folder and the index moves to it when it is complete; then the previous run's folder goes.
Each file: float32, rounded to 0.1, NaN = no data; DEFLATE COG; EPSG:4326, 0.0625°.

    python -m winemap_ingest.icon_eu          # in the workflow, ~4 hours after 00 and 12 UTC
"""

from __future__ import annotations

import bz2
import json
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin

from . import grib2
from .store import open_store

BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
RUN_HOURS = (0, 12)
STEPS = list(range(0, 79)) + list(range(81, 121, 3))
FIELDS = {"T2M": "t_2m", "PREC": "tot_prec", "TMIN": "tmin_2m"}
PREFIX = "fc/eu/"
INDEX = PREFIX + "index.json"
NIGHT = (18, 12)  # UTC: from 18 h the evening before to 12 h that morning
COG = dict(compress="DEFLATE", predictor=3, blocksize=512, overview_resampling="average", BIGTIFF="NO")
ATTRIBUTION = "Deutscher Wetterdienst, ICON-EU (DWD open data)"


def stamp(t: datetime) -> str:
    return t.strftime("%Y%m%dT%H%M")


def url(field: str, run: datetime, step: int) -> str:
    name = FIELDS[field]
    return (f"{BASE}/{run:%H}/{name}/icon-eu_europe_regular-lat-lon_single-level_{run:%Y%m%d%H}_{step:03d}_"
            f"{name.upper()}.grib2.bz2")  # fmt: skip


def exists(u: str) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=30):
            return True
    except urllib.error.URLError:
        return False


def latest_run(now: datetime) -> datetime | None:
    """The newest 00/12 UTC run whose last step is there for every field (DWD keeps about a day of runs)."""
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for back in range(0, 48, 12):
        run = day + timedelta(hours=12) - timedelta(hours=back)
        if run > now or run.hour not in RUN_HOURS:
            continue
        if all(exists(url(f, run, STEPS[-1])) for f in FIELDS):
            return run
    return None


def fetch(field: str, run: datetime, step: int) -> grib2.Field:
    with urllib.request.urlopen(url(field, run, step), timeout=120) as r:
        return grib2.read(bz2.decompress(r.read()))


def nights(run: datetime) -> list[datetime]:
    """The mornings (00 UTC of that date) whose whole night 18 -> 12 UTC lies inside the run."""
    end = run + timedelta(hours=STEPS[-1])
    out, day = [], run.replace(hour=0) + timedelta(days=1)
    while day + timedelta(hours=NIGHT[1]) <= end:
        if day - timedelta(hours=24 - NIGHT[0]) >= run:
            out.append(day)
        day += timedelta(days=1)
    return out


def night_min(run: datetime, morning: datetime, tmin: dict[int, np.ndarray]) -> np.ndarray:
    """The lowest of DWD's minimum blocks that end inside the night (they start on 00/06/12/18 UTC, so they fit it)."""
    t0, t1 = morning - timedelta(hours=24 - NIGHT[0]), morning + timedelta(hours=NIGHT[1])
    steps = [s for s in STEPS if s and t0 < run + timedelta(hours=s) <= t1 and s in tmin]
    return np.fmin.reduce([tmin[s] for s in steps])


def rain_rates(acc: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """mm/h over each step from the sums since the run started."""
    out, prev = {}, 0
    for s in STEPS[1:]:
        out[s] = np.clip(acc[s] - acc[prev], 0, None) / (s - prev)
        prev = s
    return out


def cog(a: np.ndarray, f: grib2.Field) -> bytes:
    with tempfile.TemporaryDirectory() as t:
        mid, dst = Path(t) / "mid.tif", Path(t) / "out.tif"
        h, w = a.shape
        prof = dict(driver="GTiff", width=w, height=h, count=1, dtype="float32", crs="EPSG:4326", nodata=np.nan,
                    transform=from_origin(f.lon0 - f.dlon / 2, f.lat0 + f.dlat / 2, f.dlon, f.dlat))  # fmt: skip
        with rasterio.open(mid, "w", **prof) as o:
            o.write(np.round(a, 1).astype("float32"), 1)
        rio_copy(mid, dst, driver="COG", **COG)
        return dst.read_bytes()


def run(store, now: datetime | None = None, log=print, get=fetch) -> str | None:
    """Copy the newest complete run if it is not the one shown yet. Returns its id (YYYYMMDDHH) or None."""
    now = now or datetime.now(UTC)
    index = json.loads(store.get(INDEX) or b"{}")
    r = latest_run(now)
    if r is None:
        log("no complete ICON-EU run yet")
        return None
    rid = f"{r:%Y%m%d%H}"
    if index.get("run") == rid:
        log(f"run {rid} is already there")
        return None
    base = f"{PREFIX}{rid}/"

    def load(field):
        with ThreadPoolExecutor(8) as ex:
            return dict(zip(STEPS, ex.map(lambda s: get(field, r, s), STEPS), strict=True))

    t2m = load("T2M")
    grid = t2m[0]
    times = {"T2M": [], "RAIN": [], "TMIN": []}
    for s, f in t2m.items():
        k = stamp(r + timedelta(hours=s))
        store.put(f"{base}T2M/{k}.tif", cog(f.values - 273.15, grid), "image/tiff")
        times["T2M"].append(k)
    del t2m
    for s, a in rain_rates({s: f.values for s, f in load("PREC").items()}).items():
        k = stamp(r + timedelta(hours=s))
        store.put(f"{base}RAIN/{k}.tif", cog(a, grid), "image/tiff")
        times["RAIN"].append(k)
    tmin = {s: f.values for s, f in load("TMIN").items() if s}
    for m in nights(r):
        k = f"{m:%Y%m%d}"
        store.put(f"{base}TMIN/{k}.tif", cog(night_min(r, m, tmin) - 273.15, grid), "image/tiff")
        times["TMIN"].append(k)

    old = index.get("run")
    store.put(
        INDEX,
        json.dumps(
            {
                "run": rid,
                **times,
                "night": {"from_utc": NIGHT[0], "to_utc": NIGHT[1]},
                "units": {"T2M": "°C", "RAIN": "mm/h", "TMIN": "°C"},
                "grid": {
                    "crs": "EPSG:4326",
                    "lon0": grid.lon0,
                    "lat0": grid.lat0,
                    "res": grid.dlon,
                    "width": grid.values.shape[1],
                    "height": grid.values.shape[0],
                },  # fmt: skip
                "source": ATTRIBUTION,
                "updated": now.isoformat(timespec="seconds"),
            },
            indent=1,
        ).encode(),
        "application/json",
    )
    if old and old != rid:
        for key in store.list(f"{PREFIX}{old}/"):
            store.delete(key)
    log(f"ICON-EU run {rid}: {len(times['T2M'])} temperature and {len(times['RAIN'])} rain steps, "
        f"{len(times['TMIN'])} nights")  # fmt: skip
    return rid


def main() -> None:
    run(open_store())


if __name__ == "__main__":
    main()
