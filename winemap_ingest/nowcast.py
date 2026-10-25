"""The KNMI precipitation nowcast (radar_forecast 2.0): 25 images, 5-minute sums from now to +2 hours, a new run every
5 minutes. Written as one COG per run (25 bands, mm/h like the radar map) plus `rain/nowcast/index.json`, which points
at the newest run. Files get the run time in their name, so a browser never mixes two runs; runs older than a day
are removed.

    python -m winemap_ingest.nowcast
"""

from __future__ import annotations

import io
import json
import re
from datetime import UTC, datetime, timedelta

import h5py
import numpy as np

from . import grid
from .knmi import Knmi, OutOfRequests
from .radar import _calibration
from .rain import cog, load_mask
from .store import open_store

NOWCAST = ("radar_forecast", "2.0")
PREFIX = "rain/nowcast/"
INDEX = PREFIX + "index.json"
KEEP = timedelta(days=1)


def _time(v) -> datetime:
    s = v.decode() if isinstance(v, bytes) else str(v)
    return datetime.strptime(s.split(".")[0], "%d-%b-%Y;%H:%M:%S").replace(tzinfo=UTC)


def read_nowcast(data: bytes) -> tuple[datetime, list[datetime], np.ndarray]:
    """Run time, the end time of each 5-minute step, and the steps as mm/h on the RD grid (steps, H, W)."""
    with h5py.File(io.BytesIO(data), "r") as f:
        names = sorted((k for k in f if re.fullmatch(r"image\d+", k)), key=lambda k: int(k[5:]))
        run = _time(f["overview"].attrs["product_datetime_start"])
        times, fields = [], []
        for k in names:
            raw = f[f"{k}/image_data"][...]
            cal = f[f"{k}/calibration"].attrs
            a, b = _calibration(cal)
            mm = raw.astype(np.float32) * a + b
            for key in ("calibration_missing_data", "calibration_out_of_image"):
                if key in cal:
                    for v in np.atleast_1d(cal[key]):
                        mm[raw == int(v)] = np.nan
            times.append(_time(f[k].attrs["image_datetime_valid"]))
            fields.append(grid.to_rd(mm) * 12)  # 5-minute sum -> mm/h
    order = np.argsort(times)
    return run, [times[i] for i in order], np.stack([fields[i] for i in order])


def update(store, knmi: Knmi, now: datetime | None = None, log=print) -> dict | None:
    now = now or datetime.now(UTC)
    old = json.loads(store.get(INDEX) or b"null")
    files = knmi.list_files_desc(NOWCAST, limit=1)
    if not files:
        log("no nowcast files")
        return old
    name = files[0]
    if old and old.get("source") == name:
        log(f"{name}: already there")
        return old
    run, times, steps = read_nowcast(knmi.download(NOWCAST, name))
    inside = load_mask(store)
    steps[:, ~inside] = np.nan  # the same area as the other sums
    key = f"{PREFIX}{run:%Y%m%d%H%M}.tif"
    store.put(key, cog(steps, [t.strftime("%Y-%m-%dT%H:%MZ") for t in times]), "image/tiff")
    index = {
        "version": 1,
        "units": "mm/h",
        "source": name,
        "run": run.isoformat().replace("+00:00", "Z"),
        "file": key[len("rain/") :],  # relative to rain/
        "steps": [t.isoformat().replace("+00:00", "Z") for t in times],  # end time of each 5-minute step
        "updated": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    store.put(INDEX, json.dumps(index).encode(), "application/json")
    # tidy up: runs older than a day
    for k in store.list(PREFIX):
        m = re.search(r"(\d{12})\.tif$", k)
        if m and datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=UTC) < now - KEEP:
            store.delete(k)
    log(f"nowcast {run:%Y-%m-%d %H:%M} UTC, {len(times)} steps -> {key}")
    return index


def main() -> None:
    try:
        update(open_store(), Knmi(keep=0))  # 2 requests a run, and it has priority: no reserve
    except OutOfRequests as e:
        print(f"skipped: {e}")  # the next run (in 5 minutes) tries again


if __name__ == "__main__":
    main()
