"""Precipitation sums for Winemap Live / Season (phase 1): hours, days (08-08 UTC, as KNMI) and a running total.

Layout in the store (all COGs on the RD 1 km grid, float32 mm, NaN = no data):
    rain/hour/YYYY/YYYY-MM-DD.tif   24 bands: band i = the hour starting at i:00 UTC
    rain/day/YYYY/YYYY-MM-DD.tif    the day from 08:00 UTC on that date to 08:00 UTC the next day
    rain/cum/YYYY/YYYY-MM-DD.tif    running total from `cum_start` to the end of that day
    rain/mask.tif                   1 = cells with data (the Netherlands, as in KNMI's final data); NaN elsewhere
    rain/index.json                 what is there (see `new_index`)
A period sum (this season, or any chosen days) is cum[last day] - cum[day before the first]: two files.

Sources: KNMI's climatological (final, gauge-adjusted) hourly sums for the backfill, then the real-time 5-minute
sums, added up per hour, by the hourly job. `final_until` in the index says up to when the data is final.

    python -m winemap_ingest.rain backfill --zips DIR --start 2019-10-01T08 --until 2026-09-30T08
    python -m winemap_ingest.rain live [--max-hours 48]
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from . import grid
from .knmi import FINAL_HOURLY, REALTIME, Knmi, OutOfRequests
from .radar import end_time, read_mm
from .store import open_store

HOUR = timedelta(hours=1)
DAY_START = 8  # UTC hour where a KNMI precipitation day starts
INDEX = "rain/index.json"
MASK = "rain/mask.tif"


# ---------- small helpers ----------
def hkey(d: date) -> str:
    return f"rain/hour/{d.year}/{d.isoformat()}.tif"


def dkey(d: date) -> str:
    return f"rain/day/{d.year}/{d.isoformat()}.tif"


def ckey(d: date) -> str:
    return f"rain/cum/{d.year}/{d.isoformat()}.tif"


def parse_hour(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H").replace(tzinfo=UTC)


def fmt_hour(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H")


def rain_day(t: datetime) -> date:
    """The KNMI precipitation day an hour (by its start time) belongs to."""
    return (t - timedelta(hours=DAY_START)).date()


def cog(arr: np.ndarray, descriptions: list[str] | None = None) -> bytes:
    """A (bands, H, W) or (H, W) float array as COG bytes (via a temporary GTiff: never write into a COG)."""
    a = arr[None] if arr.ndim == 2 else arr
    a = np.round(a.astype(np.float32), 2)  # 0.01 mm is plenty and compresses much better
    profile = dict(
        driver="GTiff",
        width=grid.WIDTH,
        height=grid.HEIGHT,
        count=a.shape[0],
        dtype="float32",
        crs=grid.RD_CRS,
        transform=Affine(*grid.TRANSFORM),
        nodata=np.nan,
    )
    with MemoryFile() as tmp:
        with tmp.open(**profile) as ds:
            ds.write(a)
            for i, d in enumerate(descriptions or []):
                ds.set_band_description(i + 1, d)
        with MemoryFile() as out:
            rasterio.shutil.copy(
                tmp.name,
                out.name,
                driver="COG",
                COMPRESS="DEFLATE",
                PREDICTOR="3",
                BLOCKSIZE="512",
                INTERLEAVE="BAND",  # the viewer reads one band at a time
                OVERVIEWS="NONE",
            )
            return bytes(out.read())


def read_cog(data: bytes) -> np.ndarray:
    with MemoryFile(data) as mem, mem.open() as ds:
        return ds.read().astype(np.float32)


def new_index(start: datetime) -> dict:
    return {
        "version": 1,
        "units": "mm",
        "crs": grid.RD_CRS,
        "transform": list(grid.TRANSFORM),
        "width": grid.WIDTH,
        "height": grid.HEIGHT,
        "day_start_utc": DAY_START,
        "cum_start": fmt_hour(start),
        "hour": None,  # {"first": "YYYY-MM-DDTHH", "last": ...} by the start time of the hour
        "day": None,  # {"first": "YYYY-MM-DD", "last": ...}
        "final_until": None,  # hours before this are KNMI final (gauge-adjusted), later ones real-time
        "missing_hours": {},  # day -> number of hours without radar data
        "updated": None,
        "source": "KNMI radar precipitation (CC BY 4.0): RAD_NL25_RAC_MFBS_01H (final), "
        "nl_rdr_data_rtcor_5m (real-time)",
    }


def load_index(store) -> dict | None:
    raw = store.get(INDEX)
    return json.loads(raw) if raw else None


def save_index(store, index: dict) -> None:
    index["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
    store.put(INDEX, json.dumps(index, indent=1).encode(), "application/json")


def load_mask(store) -> np.ndarray:
    """Cells with data. The final data covers the Netherlands only; the real-time data is cut to the same area so
    that sums over both stay comparable."""
    raw = store.get(MASK)
    if raw is None:
        return grid.lookup()[2]
    return ~np.isnan(read_cog(raw)[0])


# ---------- hours ----------
class Hours:
    """Hour files per UTC date, kept in memory while they are being filled and written on flush."""

    def __init__(self, store, keep: int = 3):
        self.store = store
        self.keep = keep
        self.days: dict[date, np.ndarray] = {}
        self.dirty: set[date] = set()

    def _day(self, d: date) -> np.ndarray:
        if d not in self.days:
            # write and forget the oldest dates (the backfill walks through years)
            while len(self.days) >= self.keep:
                old = min(self.days)
                self._write(old)
                del self.days[old]
            raw = self.store.get(hkey(d))
            self.days[d] = read_cog(raw) if raw else np.full((24, grid.HEIGHT, grid.WIDTH), np.nan, np.float32)
        return self.days[d]

    def get(self, t: datetime) -> np.ndarray:
        return self._day(t.date())[t.hour]

    def set(self, t: datetime, field: np.ndarray) -> None:
        self._day(t.date())[t.hour] = field
        self.dirty.add(t.date())

    def _write(self, d: date) -> None:
        if d in self.dirty:
            names = [f"{d.isoformat()}T{h:02d}:00Z" for h in range(24)]
            self.store.put(hkey(d), cog(self.days[d], names), "image/tiff")
            self.dirty.discard(d)

    def flush(self) -> None:
        for d in sorted(self.days):
            self._write(d)


def make_days(store, hours: Hours, index: dict, first: date, last: date) -> None:
    """Day sums and the running total for the rain days first..last (their hours must be there)."""
    inside = load_mask(store)
    cum_start = rain_day(parse_hour(index["cum_start"]))
    prev = None
    if first > cum_start:
        raw = store.get(ckey(first - timedelta(days=1)))
        if raw is None:
            raise RuntimeError(f"running total for {first - timedelta(days=1)} is missing")
        prev = read_cog(raw)[0]
    d = first
    while d <= last:
        t0 = datetime(d.year, d.month, d.day, DAY_START, tzinfo=UTC)
        stack = np.stack([hours.get(t0 + i * HOUR) for i in range(24)])
        gone = [i for i in range(24) if np.isnan(stack[i][inside]).all()]
        day = np.where(np.isnan(stack).all(axis=0), np.nan, np.nansum(stack, axis=0)).astype(np.float32)
        day[~inside] = np.nan
        total = np.nan_to_num(day) + (np.nan_to_num(prev) if prev is not None else 0)
        total[~inside] = np.nan
        store.put(dkey(d), cog(day, [d.isoformat()]), "image/tiff")
        store.put(ckey(d), cog(total, [f"{index['cum_start']} .. {d.isoformat()}"]), "image/tiff")
        if gone:
            index["missing_hours"][d.isoformat()] = len(gone)
        else:
            index["missing_hours"].pop(d.isoformat(), None)
        prev = total
        d += timedelta(days=1)
    span = index.get("day") or {"first": first.isoformat()}
    index["day"] = {"first": min(span["first"], first.isoformat()), "last": last.isoformat()}


def _extend_hours(index: dict, first: datetime, last: datetime) -> None:
    span = index.get("hour") or {"first": fmt_hour(first)}
    index["hour"] = {"first": min(span["first"], fmt_hour(first)), "last": fmt_hour(last)}


def _days_ready(index: dict, last_hour: datetime) -> tuple[date, date] | None:
    """Rain days that are complete up to `last_hour` and not made yet."""
    done = date.fromisoformat(index["day"]["last"]) if index.get("day") else None
    first = done + timedelta(days=1) if done else rain_day(parse_hour(index["cum_start"]))
    last = rain_day(last_hour + HOUR) - timedelta(days=1)  # the last day whose final hour is in
    return (first, last) if first <= last else None


# ---------- backfill from KNMI's yearly zips of final hourly sums ----------
def backfill(store, zips: list[Path], start: datetime, until: datetime, log=print) -> dict:
    index = load_index(store) or new_index(start)
    hours = Hours(store)
    first = last = None
    seen = np.zeros((grid.HEIGHT, grid.WIDTH), bool)  # cells with data in any hour
    for zp in sorted(zips):
        with zipfile.ZipFile(zp) as z:
            names = sorted(n for n in z.namelist() if n.endswith(".h5"))
            log(f"{zp.name}: {len(names)} hours")
            for n in names:
                t = end_time(n) - HOUR  # by start time
                if t < start or t >= until:
                    continue
                field = grid.to_rd(read_mm(z.read(n)))
                seen |= ~np.isnan(field)
                hours.set(t, field)
                first = first or t
                last = t
    hours.flush()
    if last is None:
        raise SystemExit("no hours found in the given period")
    if store.get(MASK) is None:
        store.put(MASK, cog(np.where(seen, 1, np.nan).astype(np.float32), ["cells with data"]), "image/tiff")
    _extend_hours(index, first, last)
    index["final_until"] = fmt_hour(last + HOUR)
    span = _days_ready(index, last)
    if span:
        log(f"days {span[0]} .. {span[1]}")
        make_days(store, hours, index, *span)
    save_index(store, index)
    return index


def download_final_zips(knmi: Knmi, years: range, folder: Path, log=print) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    names = knmi.list_files(FINAL_HOURLY, limit=100)
    out = []
    for y in years:
        name = next((n for n in names if f"_01H_{y}0101T" in n), None)
        if not name:
            log(f"no final hourly zip for {y}")
            continue
        p = folder / name
        if not p.exists():
            log(f"downloading {name}")
            p.write_bytes(knmi.download(FINAL_HOURLY, name))
        out.append(p)
    return out


# ---------- the hourly job: real-time 5-minute sums ----------
def rt_name(t: datetime) -> str:
    return f"RAD_NL25_RAC_RT_{t:%Y%m%d%H%M}.h5"


def live(store, knmi: Knmi, max_hours: int = 48, now: datetime | None = None, log=print) -> dict:
    index = load_index(store)
    if not index or not index.get("hour"):
        raise SystemExit("no index yet: run the backfill first")
    now = now or datetime.now(UTC)
    hours = Hours(store)
    inside = load_mask(store)
    last = parse_hour(index["hour"]["last"])
    added = 0
    try:
        files = knmi.list_files(REALTIME, begin=rt_name(last + HOUR), limit=12 * max_hours + 12)
        by_hour: dict[datetime, list[str]] = {}
        for f in files:
            by_hour.setdefault(end_time(f) - timedelta(minutes=5), []).append(f)  # a 5-min file by its start
        while added < max_hours:
            h = last + HOUR
            parts = [f for t, fs in by_hour.items() if h <= t < h + HOUR for f in fs]
            late = now - (h + HOUR) > timedelta(hours=3)
            if len(parts) < 12 and not (late and len(parts) >= 9):
                if late and h + HOUR < now - timedelta(hours=6):
                    log(f"{fmt_hour(h)}: only {len(parts)} of 12 files, kept as missing")
                    hours.set(h, np.full((grid.HEIGHT, grid.WIDTH), np.nan, np.float32))
                    last = h
                    added += 1
                    continue
                break  # not complete yet: next run
            fields = np.stack([grid.to_rd(read_mm(knmi.download(REALTIME, f))) for f in sorted(parts)])
            hour = np.where(np.isnan(fields).all(axis=0), np.nan, np.nansum(fields, axis=0) * 12 / len(parts))
            hour[~inside] = np.nan
            hours.set(h, hour.astype(np.float32))
            last = h
            added += 1
    except OutOfRequests as e:
        log(f"stopping early: {e}")
    hours.flush()
    if added:
        _extend_hours(index, parse_hour(index["hour"]["first"]), last)
        span = _days_ready(index, last)
        if span:
            make_days(store, hours, index, *span)
        save_index(store, index)
    log(f"added {added} hours, last {fmt_hour(last)}; days up to {index['day']['last'] if index.get('day') else '-'}")
    return index


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="winemap_ingest.rain")
    ap.add_argument("--store", help="'r2' or a folder (default: WX_STORE)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument(
        "--zips", type=Path, required=True, help="folder with KNMI's yearly hourly zips (downloaded if missing)"
    )
    b.add_argument("--start", default="2019-10-01T08")
    b.add_argument("--until", required=True, help="first hour NOT to take from the final data, e.g. 2026-09-30T08")
    lv = sub.add_parser("live")
    lv.add_argument("--max-hours", type=int, default=48)
    a = ap.parse_args(argv)
    store = open_store(a.store)
    if a.cmd == "backfill":
        start, until = parse_hour(a.start), parse_hour(a.until)
        zips = download_final_zips(Knmi(), range(start.year, until.year + 1), a.zips)
        index = backfill(store, zips, start, until)
    else:
        index = live(store, Knmi(), a.max_hours)
    print(json.dumps({k: index[k] for k in ("hour", "day", "final_until")}))


if __name__ == "__main__":
    main()
