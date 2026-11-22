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


class FakeRemote:
    __hash__ = None  # like ecmwf.datastores' Remote (22 Nov 2026: it broke a dict of remotes)

    def __init__(self, request, plan):
        self.request, self.plan, self.deleted = request, list(plan), False

    @property
    def status(self):
        return self.plan.pop(0) if len(self.plan) > 1 else self.plan[0]

    def download(self, path):
        open(path, "wb").write(b"nc")

    def delete(self):
        self.deleted = True


class FakeClient:
    def __init__(self, plan=("accepted", "running", "successful")):
        self.plan, self.remotes = plan, []

    def submit(self, dataset, request):
        self.remotes.append(FakeRemote(request, self.plan))
        return self.remotes[-1]


def test_world_download_a_month_per_request_for_the_box_around_the_missing_areas(tmp_path, monkeypatch):
    store = LocalStore(tmp_path)
    store.put(climate.world_key("oc", "tg", 2024, 12), b"done")  # already there: not asked again
    monkeypatch.setattr(
        climate, "era5_cut_areas", lambda var, raw, areas: {r: {(2024, 12): b"packed", (2025, 1): b"x"} for r in areas}
    )
    client = FakeClient()
    n = climate.run_era5_world(
        store, minutes=5, years=(2024, 2024), queue=3, log=lambda *_: None, after_europe=False, client=client, poll=0
    )
    seen = [(tuple(r.request["month"]), tuple(r.request["area"]), r.request["variable"][0]) for r in client.remotes]
    assert n == len(seen) == len(climate.VARS) * 12  # one request per variable and month, every area in it
    assert all(len(m) == 1 for m, _, _ in seen)
    full = tuple(climate.world_box(climate.WORLD_AREAS.values()))
    assert full == (62.0, -170.0, -56.0, 180.0)
    assert all(a == full for m, a, _ in seen if m != ("12",))
    # December's tg: oc was there, so the box of the other four (and only the asked month is stored)
    others = tuple(climate.world_box(x for r, x in climate.WORLD_AREAS.items() if r != "oc"))
    assert (("12",), others, "2m_temperature") in seen
    assert store.get(climate.world_key("na", "rr", 2024, 12)) == b"packed"
    assert store.get(climate.world_key("oc", "tg", 2024, 12)) == b"done"  # not overwritten
    assert store.get(climate.world_key("na", "rr", 2025, 1)) is None


def test_world_leaves_nothing_in_the_cds_queue_when_the_time_is_up(tmp_path):
    client = FakeClient(plan=("accepted",))  # CDS never starts them
    n = climate.run_era5_world(
        LocalStore(tmp_path),
        0.001,
        years=(2024, 2024),
        queue=4,
        log=lambda *_: None,
        after_europe=False,
        client=client,
        poll=0.01,
    )
    assert n == 0 and len(client.remotes) == 4 and all(r.deleted for r in client.remotes)


def test_world_stops_after_five_failures(tmp_path):
    client = FakeClient(plan=("failed",))
    n = climate.run_era5_world(
        LocalStore(tmp_path),
        5,
        years=(2024, 2024),
        queue=2,
        log=lambda *_: None,
        after_europe=False,
        client=client,
        poll=0,
    )
    assert n == 0 and len(client.remotes) <= 6


def test_a_box_cut_into_areas_is_packed_as_the_areas_asked_alone():
    v = np.arange(3 * 2 * 2, dtype="f4").reshape(3, 2, 2) + 280.0  # lat 52.0, 51.9 x lon 5.0, 5.1
    box = era5_file("tn", date(2026, 3, 1), 3, v)
    parts = climate.era5_cut_areas("tn", box, {"w": [52.0, 5.0, 51.9, 5.0], "ne": [52.0, 5.1, 52.0, 5.1]})
    t_w, a_w = read(parts["w"][(2026, 3)], "tn")
    t_ne, a_ne = read(parts["ne"][(2026, 3)], "tn")
    _, a_all = read(climate.era5_pack("tn", box), "tn")
    assert a_w.shape == (3, 2, 1) and a_ne.shape == (3, 1, 1)
    assert np.allclose(a_w, a_all[:, :, :1]) and np.allclose(a_ne, a_all[:, :1, 1:])
    assert t_w == t_ne


def test_a_year_split_into_months_is_packed_as_months_asked_alone():
    v = np.arange(4 * 2 * 2, dtype="f4").reshape(4, 2, 2) + 280.0
    two_months = era5_file("tx", date(2026, 1, 30), 4, v)  # 30, 31 January, 1, 2 February
    parts = climate.era5_pack_months("tx", two_months)
    assert set(parts) == {(2026, 1), (2026, 2)}
    alone = climate.era5_pack("tx", era5_file("tx", date(2026, 2, 1), 2, v[2:]))
    t1, a1 = read(parts[(2026, 2)], "tx")
    t2, a2 = read(alone, "tx")
    assert t1 == t2 and np.allclose(a1, a2)


def test_world_waits_until_europe_is_complete(tmp_path, monkeypatch):
    store = LocalStore(tmp_path)
    monkeypatch.setenv("CDS_API_KEY", "x")
    assert not climate.europe_complete(store)
    assert climate.run_era5_world(store, minutes=5, years=(2024, 2024), log=lambda *_: None) == 0  # no CDS call
    for v in climate.VARS:
        store.put(climate.key_of("era5land", v, climate.ERA5_FIRST_YEAR, 1), b"x")
    assert climate.europe_complete(store)


def test_era5_completes_a_short_recent_month_and_asks_the_next_months_rain(tmp_path, monkeypatch):
    # 5 Nov 2026: October was fetched on 1 Nov, ended on the 26th and was never completed, so the season could not be
    # computed; the rain of 31 Oct is in the November file (its 00 UTC step on the 1st)
    store = LocalStore(tmp_path)
    days = {"2026-10": 26, "2026-09": 30}
    for ym, n in days.items():
        for v in climate.VARS:
            store.put(climate.key_of("era5land", v, int(ym[:4]), int(ym[5:])), f"{n}".encode())
    monkeypatch.setattr(climate, "days_in", lambda s, key: int(s.get(key)))
    have = set(store.list("climate/era5land/eu/"))

    def due(var, y, m, today):
        return climate.era5_due(store, climate.key_of("era5land", var, y, m), var, y, m, have, today)

    # too early: the last days are not there yet
    assert not due("tx", 2026, 10, date(2026, 11, 5))
    assert not due("rr", 2026, 11, date(2026, 11, 5))
    # then the short month again, the complete one not, and of the running month only the rain, once
    assert due("tx", 2026, 10, date(2026, 11, 6))
    assert not due("tx", 2026, 9, date(2026, 11, 6))
    assert due("rr", 2026, 11, date(2026, 11, 7))
    assert not due("tx", 2026, 11, date(2026, 11, 7))
    have.add(climate.key_of("era5land", "rr", 2026, 11))
    assert not due("rr", 2026, 11, date(2026, 11, 20))
    # a month that is still missing is asked as before; an old short month is left alone
    assert due("tx", 2026, 8, date(2026, 11, 6))
    assert not due("tx", 2026, 9, date(2027, 3, 1))
    # the finished November comes in full in December
    store.put(climate.key_of("era5land", "rr", 2026, 11), b"7")
    assert due("rr", 2026, 11, date(2026, 12, 6))
