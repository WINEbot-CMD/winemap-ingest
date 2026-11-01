"""European radar rain (Winemap backlog 1.6, docs/38 §10; Jasper 1 Nov 2026: "European + national where sharper"): the
EUMETNET OPERA composites (CC BY 4.0, via MeteoGate's open bucket, no key needed), mirrored into winemap-wx so the
viewer can read them (the bucket sends no CORS headers). NL keeps KNMI's sharper radar.

    rain/eu/RATE/<YYYYMMDDTHHMM>.tif   instantaneous rain rate (mm/h), every 15 minutes
    rain/eu/ACRR/<YYYYMMDDTHHMM>.tif   1-hour sum (mm) ending at that time, every 15 minutes (a running hour)
    rain/eu/index.json                 the times there, the projection and the grid

Each file: the composite's rain band only (its quality band left out), float32, NaN = no radar coverage / no data,
0 = dry; DEFLATE COG. 2 km grid in OPERA's Lambert equal-area (lat_0 55, lon_0 10). Files older than KEEP go.

    python -m winemap_ingest.opera          # every 15 minutes in the workflow
"""

from __future__ import annotations

import json
import re
import tempfile
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy

from .store import open_store

BUCKET = "https://s3.waw3-1.cloudferro.com/openradar-24h"
PRODUCTS = ("RATE", "ACRR")
PREFIX = "rain/eu/"
INDEX = PREFIX + "index.json"
KEEP = timedelta(hours=48)
BUDGET = timedelta(minutes=6)  # the workflow stops a run after 10 minutes
NAME = re.compile(r"OPERA@(\d{8}T\d{4})@0@(RATE|ACRR)\.tiff$")
COG = dict(compress="DEFLATE", predictor=3, blocksize=512, overview_resampling="average", BIGTIFF="NO")
ATTRIBUTION = "EUMETNET OPERA radar composite (CC BY 4.0), via MeteoGate"


def stamp(t: datetime) -> str:
    return t.strftime("%Y%m%dT%H%M")


def parse_stamp(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%dT%H%M").replace(tzinfo=UTC)


def listing(day: datetime) -> list[tuple[str, str, datetime]]:
    """(product, source key, time) of one day's composites in the open bucket."""
    out, token = [], None
    while True:
        q = f"?list-type=2&prefix={day:%Y/%m/%d}/OPERA/COMP/" + (f"&continuation-token={token}" if token else "")
        with urllib.request.urlopen(BUCKET + q, timeout=60) as r:
            xml = r.read().decode()
        for key in re.findall(r"<Key>([^<]+)</Key>", xml):
            m = NAME.search(key)
            if m:
                out.append((m.group(2), key, parse_stamp(m.group(1))))
        nxt = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not nxt:
            return out
        token = urllib.parse.quote(nxt.group(1), safe="")


def clean(raw: bytes) -> tuple[bytes, dict]:
    """The rain band as a compact COG (NaN = no data) and the grid it is on."""
    with tempfile.TemporaryDirectory() as t:
        src, mid, dst = Path(t) / "in.tif", Path(t) / "mid.tif", Path(t) / "out.tif"
        src.write_bytes(raw)
        with rasterio.open(src) as r:
            a = r.read(1).astype("float32")
            fill = float(r.nodatavals[0]) if r.nodatavals[0] is not None else -9999000.0
            a[(a == fill) | (a < 0)] = np.nan
            prof = dict(driver="GTiff", width=r.width, height=r.height, count=1, dtype="float32", crs=r.crs,
                        transform=r.transform, nodata=np.nan)  # fmt: skip
            grid = {
                "proj4": r.crs.to_proj4(),
                "x0": r.transform.c,
                "y0": r.transform.f,
                "res": r.transform.a,
                "width": r.width,
                "height": r.height,
            }
        with rasterio.open(mid, "w", **prof) as o:
            o.write(np.round(a, 2), 1)
        rio_copy(mid, dst, driver="COG", **COG)
        return dst.read_bytes(), grid


def run(store, now: datetime | None = None, log=print, budget: timedelta = BUDGET) -> int:
    """Copy what is new (the newest first, for at most `budget`: a first run catches up over a few runs), drop what is
    too old, write the index. What is already in the bucket counts, also when an earlier run stopped. Returns the
    files copied."""
    now = now or datetime.now(UTC)
    start = datetime.now(UTC)
    index = json.loads(store.get(INDEX) or b"{}")
    have = {p: set(index.get(p, [])) for p in PRODUCTS}
    for p in PRODUCTS:  # files an interrupted run copied but did not index
        have[p] |= {k.rsplit("/", 1)[1][:-4] for k in store.list(f"{PREFIX}{p}/") if k.endswith(".tif")}
    todo = [(p, k, t) for day in (now - timedelta(days=1), now) for p, k, t in listing(day)]
    todo = sorted((x for x in todo if stamp(x[2]) not in have[x[0]] and now - x[2] <= KEEP), key=lambda x: x[2],
                  reverse=True)  # fmt: skip
    n = 0
    for product, key, t in todo:
        if datetime.now(UTC) - start > budget:
            log(f"time is up: {len(todo) - n} files left for the next run")
            break
        with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=120) as r:
            data, grid = clean(r.read())
        store.put(f"{PREFIX}{product}/{stamp(t)}.tif", data, "image/tiff")
        have[product].add(stamp(t))
        index["grid"] = grid
        n += 1
    # older than KEEP: out of the index and the bucket
    for product in PRODUCTS:
        old = {s for s in have[product] if now - parse_stamp(s) > KEEP}
        for s in old:
            store.delete(f"{PREFIX}{product}/{s}.tif")
        have[product] -= old
        index[product] = sorted(have[product])
    index.update(
        {
            "source": ATTRIBUTION,
            "units": {"RATE": "mm/h", "ACRR": "mm"},
            "updated": now.isoformat(timespec="seconds"),
        }
    )
    store.put(INDEX, json.dumps(index, indent=1).encode(), "application/json")
    log(f"{n} OPERA files copied; RATE {len(index['RATE'])}, ACRR {len(index['ACRR'])} in the last {KEEP}")
    return n


def main() -> None:
    run(open_store())


if __name__ == "__main__":
    main()
