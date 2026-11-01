import json
from datetime import UTC, datetime, timedelta

import numpy as np
import rasterio
from rasterio.transform import from_origin

from winemap_ingest import opera
from winemap_ingest.store import LocalStore


def composite(path, fill=-9999000.0):
    """Like an OPERA GeoTIFF: band 1 rain, band 2 quality; the fill value outside the radars."""
    a = np.array([[0.0, 1.5, fill], [3.25, 0.0, fill]], dtype="float32")
    prof = dict(driver="GTiff", width=3, height=2, count=2, dtype="float32", nodata=fill,
                crs="+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 +ellps=WGS84 +units=m",
                transform=from_origin(-1000, 1000, 2000, 2000))  # fmt: skip
    with rasterio.open(path, "w", **prof) as o:
        o.write(a, 1)
        o.write(np.ones_like(a), 2)
    return path.read_bytes()


def test_clean_keeps_the_rain_band_with_nan_outside(tmp_path):
    raw, grid = opera.clean(composite(tmp_path / "c.tif"))
    with rasterio.MemoryFile(raw) as m, m.open() as r:
        assert r.count == 1 and r.dtypes[0] == "float32"
        a = r.read(1)
    assert np.isnan(a[0, 2]) and a[0, 1] == 1.5 and a[1, 0] == 3.25
    assert grid["res"] == 2000 and grid["width"] == 3 and "lat_0=55" in grid["proj4"]


def test_run_copies_new_files_and_drops_old_ones(tmp_path, monkeypatch):
    now = datetime(2026, 11, 1, 9, 30, tzinfo=UTC)
    store = LocalStore(tmp_path / "store")
    old = now - timedelta(hours=60)
    store.put(f"rain/eu/RATE/{opera.stamp(old)}.tif", b"old")
    store.put(opera.INDEX, json.dumps({"RATE": [opera.stamp(old)], "ACRR": []}).encode())
    src = composite(tmp_path / "c.tif")
    keys = [("RATE", "k/r1", now - timedelta(minutes=30)), ("ACRR", "k/a1", now - timedelta(minutes=30)),
            ("RATE", "k/r0", now - timedelta(hours=50))]  # fmt: skip
    monkeypatch.setattr(opera, "listing", lambda day: keys if day.date() == now.date() else [])

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return src

    monkeypatch.setattr(opera.urllib.request, "urlopen", lambda url, timeout=0: Resp())
    assert opera.run(store, now, log=lambda *_: None) == 2  # the 50 h old one is past KEEP
    index = json.loads(store.get(opera.INDEX))
    assert index["RATE"] == ["20261101T0900"] and index["ACRR"] == ["20261101T0900"]
    assert store.get(f"rain/eu/RATE/{opera.stamp(old)}.tif") is None  # removed
    assert "CC BY 4.0" in index["source"] and index["grid"]["res"] == 2000
    assert opera.run(store, now, log=lambda *_: None) == 0  # nothing new the second time
