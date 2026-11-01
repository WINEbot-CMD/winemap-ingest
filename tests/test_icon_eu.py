import json
import struct
from datetime import UTC, datetime, timedelta

import imagecodecs
import numpy as np
import pytest
import rasterio

from winemap_ingest import grib2, icon_eu
from winemap_ingest.store import LocalStore


def sm(v: int, n: int) -> bytes:
    """GRIB's sign-and-magnitude integer."""
    return ((abs(v) | (1 << (8 * n - 1))) if v < 0 else v).to_bytes(n, "big")


def section(num: int, body: bytes) -> bytes:
    return (len(body) + 5).to_bytes(4, "big") + bytes([num]) + body


def message(values: np.ndarray, lon0=336.5, lat0=70.5, d=0.0625, south_first=False) -> bytes:
    """A GRIB2 field like DWD's: regular lat/lon (3.0), CCSDS packing (5.42), 0.01 precision."""
    nj, ni = values.shape
    rows = values[::-1] if south_first else values
    ref = float(rows.min())
    x = np.round((rows.ravel() - ref) * 100).astype(">u2")
    flags, block, rsi = 14, 32, 128  # MSB, preprocess, 3-byte off: like DWD's
    data = imagecodecs.aec_encode(x.tobytes(), bitspersample=16, flags=flags, blocksize=block, rsi=rsi)
    s1 = bytes(7) + (2026).to_bytes(2, "big") + bytes([11, 1, 0, 0, 0]) + bytes(2)
    la1 = lat0 - (nj - 1) * d if south_first else lat0
    s3 = (bytes([0]) + (ni * nj).to_bytes(4, "big") + bytes(2) + (0).to_bytes(2, "big") + bytes(16)
          + ni.to_bytes(4, "big") + nj.to_bytes(4, "big") + bytes(8) + sm(round(la1 * 1e6), 4)
          + sm(round(lon0 * 1e6), 4) + bytes(1) + bytes(8) + round(d * 1e6).to_bytes(4, "big")
          + round(d * 1e6).to_bytes(4, "big") + bytes([0x40 if south_first else 0]))  # fmt: skip
    s4 = bytes(4) + bytes(20)
    s5 = ((ni * nj).to_bytes(4, "big") + (42).to_bytes(2, "big") + struct.pack(">f", ref * 100) + sm(0, 2) + sm(2, 2)
          + bytes([16, 0, flags, block]) + rsi.to_bytes(2, "big"))  # fmt: skip
    body = section(1, s1) + section(3, s3) + section(4, s4) + section(5, s5)
    body += section(6, bytes([255])) + section(7, data)
    total = 16 + len(body) + 4
    return b"GRIB" + bytes(2) + bytes([0, 2]) + total.to_bytes(8, "big") + body + b"7777"


def test_read_decodes_ccsds_with_north_first():
    a = np.array([[280.0, 281.25, 279.5], [270.13, 300.0, 285.5]], dtype="float32")
    f = grib2.read(message(a))
    assert f.values.shape == (2, 3)
    np.testing.assert_allclose(f.values, a, atol=0.006)
    assert f.lon0 == -23.5 and f.lat0 == 70.5 and f.dlon == 0.0625
    assert f.ref_time == datetime(2026, 11, 1, tzinfo=UTC)


def test_read_flips_rows_that_go_south_to_north():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    f = grib2.read(message(a, south_first=True))
    np.testing.assert_allclose(f.values, a, atol=0.006)
    assert f.lat0 == 70.5


def test_nights_are_the_whole_ones_inside_the_run():
    r00 = datetime(2026, 11, 1, 0, tzinfo=UTC)
    assert [f"{m:%d}" for m in icon_eu.nights(r00)] == ["02", "03", "04", "05"]
    r12 = datetime(2026, 11, 1, 12, tzinfo=UTC)
    assert [f"{m:%d}" for m in icon_eu.nights(r12)] == ["02", "03", "04", "05", "06"]


def test_night_min_takes_only_the_blocks_inside_the_night():
    r = datetime(2026, 11, 1, 0, tzinfo=UTC)
    tmin = {s: np.full((1, 1), 280.0) for s in icon_eu.STEPS if s}
    tmin[18] = np.full((1, 1), 250.0)  # 12-18 UTC on the 1st: before the night of the 2nd starts
    tmin[30] = np.full((1, 1), 270.0)  # 06 UTC on the 2nd: inside
    tmin[37] = np.full((1, 1), 260.0)  # 13 UTC on the 2nd: after
    m = icon_eu.night_min(r, datetime(2026, 11, 2, tzinfo=UTC), tmin)
    assert m[0, 0] == 270.0


def test_rain_rates_are_mm_per_hour_per_step():
    acc = {s: np.full((1, 1), float(s)) for s in icon_eu.STEPS}  # 1 mm every hour
    acc[78] = acc[77].copy()  # a dry hour
    rates = icon_eu.rain_rates(acc)
    assert rates[1][0, 0] == 1.0 and rates[78][0, 0] == 0.0
    assert rates[81][0, 0] == pytest.approx((81 - 77) / 3)  # 3-hour step after +78 h
    assert set(rates) == set(icon_eu.STEPS[1:])


def test_run_writes_a_new_run_and_drops_the_old_one(tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    store.put(icon_eu.PREFIX + "2026103112/T2M/x.tif", b"old")
    store.put(icon_eu.INDEX, json.dumps({"run": "2026103112"}).encode())
    r = datetime(2026, 11, 1, 0, tzinfo=UTC)
    monkeypatch.setattr(icon_eu, "latest_run", lambda now: r)
    grid = grib2.read(message(np.zeros((2, 3), dtype="float32")))

    def get(field, run, step):
        v = {"T2M": 283.15, "PREC": float(step), "TMIN": 273.15 - (step == 30) * 3}[field]
        return grib2.Field(np.full((2, 3), v, np.float32), grid.lon0, grid.lat0, grid.dlon, grid.dlat, run)

    assert icon_eu.run(store, now=r + timedelta(hours=4), get=get, log=lambda *_: None) == "2026110100"
    ix = json.loads(store.get(icon_eu.INDEX))
    assert ix["run"] == "2026110100" and len(ix["T2M"]) == len(icon_eu.STEPS)
    assert len(ix["RAIN"]) == len(icon_eu.STEPS) - 1
    assert ix["TMIN"] == ["20261102", "20261103", "20261104", "20261105"]
    assert store.list(icon_eu.PREFIX + "2026103112/") == []
    with rasterio.MemoryFile(store.get(icon_eu.PREFIX + "2026110100/TMIN/20261102.tif")) as m, m.open() as t:
        assert t.crs.to_epsg() == 4326 and t.read(1)[0, 0] == pytest.approx(-3.0)
        assert t.transform.c == pytest.approx(-23.5 - 0.0625 / 2)
    with rasterio.MemoryFile(store.get(icon_eu.PREFIX + "2026110100/T2M/20261101T0100.tif")) as m, m.open() as t:
        assert t.read(1)[0, 0] == pytest.approx(10.0)
    # the same run again: nothing to do
    assert icon_eu.run(store, now=r + timedelta(hours=5), get=get, log=lambda *_: None) is None
