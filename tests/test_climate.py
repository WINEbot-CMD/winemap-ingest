from datetime import date

import netCDF4
import numpy as np

from winemap_ingest import climate
from winemap_ingest.store import LocalStore


def knmi_file(day: date, values: np.ndarray, code: str = "TN1") -> tuple[str, bytes]:
    """Bytes shaped like a KNMI daily grid (prediction (1, y, x), float32, -9999 = no data)."""
    s = day.strftime("%Y%m%d")
    name = f"INTER_OPER_R___{code}_____L3__{s}T000000_{s}T000000_0006.nc"
    d = netCDF4.Dataset("k.nc", "w", memory=1)
    d.createDimension("time", 1)
    d.createDimension("y", values.shape[0])
    d.createDimension("x", values.shape[1])
    d.createVariable("x", "f8", ("x",))[:] = np.arange(values.shape[1]) * 1000.0
    d.createVariable("y", "f8", ("y",))[:] = np.arange(values.shape[0]) * 1000.0
    d.createVariable("prediction", "f4", ("time", "y", "x"), fill_value=-9999.0)[0] = values
    return name, bytes(d.close())


def read(raw: bytes, var: str):
    with netCDF4.Dataset("r.nc", memory=raw) as d:
        t = [date.fromordinal(climate.EPOCH.toordinal() + int(i)) for i in d["time"][:]]
        v = d[var]
        assert v.dtype == np.int16 and v.filters()["zlib"]
        return t, v[:]  # masked and scaled on read


def test_knmi_day_and_int16():
    assert climate.knmi_day("INTER_OPER_R___TN1_____L3__20240101T000000_20240102T000000_0006.nc") == date(2024, 1, 1)
    q = climate.to_int16(np.array([1.234, np.nan, 1e6, -1e6]))
    assert list(q) == [123, climate.FILL, 32767, -32767]


def test_knmi_pack_keeps_values_and_gaps():
    a = np.array([[-1.234, 0.0], [20.004, -9999.0]], dtype="f4")
    grids, days = [], []
    for day, vals in ((date(2024, 1, 1), a), (date(2024, 1, 3), a + 1)):
        g, coords = climate.knmi_read(knmi_file(day, vals)[1])
        grids.append(g)
        days.append(day)
    t, v = read(climate.knmi_pack("tn", days, grids, coords), "tn")
    assert t == days
    assert np.ma.getmaskarray(v)[0, 1, 1]
    assert np.allclose(v[0].compressed(), [-1.23, 0.0, 20.0], atol=0.0051)


def test_pack_local_knmi_only_complete_months(tmp_path):
    raw = tmp_path / "raw"
    a = np.ones((2, 2), dtype="f4")
    for name, version in climate.KNMI_DATASETS.values():
        folder = raw / f"{name}_v{version}" / "2023"
        folder.mkdir(parents=True)
        for d in range(1, 32):  # all of January, one day of February
            fname, data = knmi_file(date(2023, 1, d), a)
            (folder / fname).write_bytes(data)
        fname, data = knmi_file(date(2023, 2, 1), a)
        (folder / fname).write_bytes(data)
    store = LocalStore(tmp_path / "store")
    assert climate.pack_local_knmi(store, raw, [2023], log=lambda _: None) == 4
    assert sorted(store.list("climate/knmi/nl/")) == [
        f"climate/knmi/nl/{v}/2023/2023-01.nc" for v in sorted(climate.VARS)
    ]


def era5_file(var: str, first: date, n: int, values: np.ndarray) -> bytes:
    """Bytes shaped like a CDS file: t2m in K, or tp in m stamped at 00 UTC of the next day."""
    d = netCDF4.Dataset("e.nc", "w", memory=1)
    d.createDimension("valid_time", n)
    d.createDimension("latitude", 2)
    d.createDimension("longitude", 2)
    d.createVariable("latitude", "f8", ("latitude",))[:] = [52.0, 51.9]
    d.createVariable("longitude", "f8", ("longitude",))[:] = [5.0, 5.1]
    t = d.createVariable("valid_time", "i8", ("valid_time",))
    if var == "rr":
        t.units = "seconds since 1970-01-01"
        t[:] = [((first - date(1970, 1, 1)).days + i) * 86400 for i in range(n)]
    else:
        t.units = f"days since {first.isoformat()} 00:00:00"
        t[:] = range(n)
    d.createVariable("tp" if var == "rr" else "t2m", "f4", ("valid_time", "latitude", "longitude"))[:] = values
    return bytes(d.close())


def test_era5_pack_units_and_rain_day():
    v = np.full((3, 2, 2), 283.15, dtype="f4")
    v[:, 1, 1] = np.nan  # sea
    t, a = read(climate.era5_pack("tn", era5_file("tn", date(2026, 9, 1), 3, v)), "tn")
    assert t[0] == date(2026, 9, 1) and np.allclose(a[0, 0, 0], 10.0) and np.ma.getmaskarray(a)[0, 1, 1]
    r = np.full((3, 2, 2), 0.0123, dtype="f4")  # 12.3 mm
    t, a = read(climate.era5_pack("rr", era5_file("rr", date(2026, 9, 1), 3, r)), "rr")
    assert t == [date(2026, 8, 31), date(2026, 9, 1), date(2026, 9, 2)]  # 00 UTC = the day before
    assert np.allclose(a, 12.3, atol=0.006)


class FakeKnmi:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.downloads = 0

    def list_files(self, dataset, begin=None, limit=500):
        return sorted(f for f in self.files if begin is None or f > begin)[:limit]

    def download(self, dataset, filename):
        self.downloads += 1
        return self.files[filename]


def test_run_knmi_writes_missing_months_and_skips_done(tmp_path, monkeypatch):
    files = dict(knmi_file(date(1961, 1, d), np.ones((2, 2), dtype="f4")) for d in range(1, 32))
    store = LocalStore(tmp_path)
    api = FakeKnmi(files)
    monkeypatch.setattr(
        climate, "knmi_month_files", lambda api, var, y, m: [f for f in api.files if f"__{y}{m:02d}" in f]
    )
    assert climate.run_knmi(store, api, minutes=5, years=(1961, 1961), log=lambda _: None) == 4
    assert api.downloads == 4 * 31
    assert climate.run_knmi(store, api, minutes=5, years=(1961, 1961), log=lambda _: None) == 0  # nothing left


def test_progress_counts_months(tmp_path):
    store = LocalStore(tmp_path)
    for v in climate.VARS:
        store.put(f"climate/knmi/nl/{v}/1961/1961-01.nc", b"x")
        store.put(f"climate/knmi/nl/{v}/2026/2026-09.nc", b"x")
    p = climate.progress(store, "knmi", "ok", today=date(2026, 10, 27))
    assert p["files"] == 8 and p["total"] == (65 * 12 + 9) * 4
    assert (p["oldest"], p["newest"], p["org"]) == ("1961-01", "2026-09", "KNMI")


def test_latest_version_per_day():
    a = "INTER_OPER_R___TX1_____L3__20180101T000000_20180102T000000_0002.nc"
    b = "INTER_OPER_R___TX1_____L3__20180101T000000_20180102T000000_0006.nc"
    c = "INTER_OPER_R___TX1_____L3__20180102T000000_20180103T000000_0002.nc"
    assert climate.latest_per_day([b, c, a]) == [b, c]
