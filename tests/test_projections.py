import netCDF4
import numpy as np
import pytest

from winemap_ingest import projections
from winemap_ingest.store import LocalStore


class FakeGrid:
    """Like RemoteGrid: world 1-degree axes (lon -180..179), value = a per-variable constant + month / 100."""

    VALUES = {"tasmax": 25.0, "tasmin": 13.0, "pr": 60.0, "hd35": 1.0, "tr": 2.0, "fd": 4.0}
    opened: list[str] = []

    def __init__(self, url):
        self.var = next(v for v in self.VALUES if f"/{v}/" in url)
        FakeGrid.opened.append(self.var)
        self.lat = np.arange(-90.0, 91.0)
        self.lon = np.arange(-180.0, 180.0)
        self.n_time = 1032

    def rows(self, t, r0, r1):
        month = t % 12 + 1
        return np.full((r1 - r0, len(self.lon)), self.VALUES[self.var] + (month - 1) * 0 + 0.0, dtype="f4")


def test_window_wraps_greenwich_and_cuts_the_box():
    lat = np.arange(-90.0, 90.25, 0.25)
    for lon in (np.arange(-180, 180, 0.25), np.arange(0, 360, 0.25)):  # both conventions
        r0, r1, cols, la, lo = projections.window(lat, lon, projections.REGIONS["eu"][0])
        assert la[0] == 34 and la[-1] == 72 and r1 - r0 == len(la)
        assert lo[0] == -25 and lo[-1] == 45 and np.all(np.diff(lo) > 0)


def test_bcis_from_months():
    shape = (2, 7, 3, 1)
    v = {k: np.full(shape, x) for k, x in FakeGrid.VALUES.items()}
    lat = np.array([38.0, 45.0, 52.0])
    out = projections.bcis(v, lat)
    # Tgem 19, Huglin per day (19 + 25) / 2 - 10 = 12; 214 days Apr-Oct, 183 Apr-Sep
    assert np.allclose(out["EHI"][0, :, 0], 12 * 214 * np.array([1.02, 1.04, 1.06]))
    assert np.allclose(out["HI"][0, :, 0], 12 * 183 * np.array([1.02, 1.04, 1.06]))
    assert np.allclose(out["GST"], 19) and np.allclose(out["GSP"], 7 * 60)
    assert np.allclose(out["EFD"], 0.5 * 4 + 4) and np.allclose(out["EHD"], 7) and np.allclose(out["TN"], 14)


def test_run_writes_one_file_per_model_run_and_skips_done(tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    store.put(projections.key("eu", "m1", "ssp245"), b"done")
    reported = []
    monkeypatch.setattr(projections, "report", lambda f, log=print: reported.append(f()))
    monkeypatch.setattr(projections, "LAST_YEAR", 2016)
    monkeypatch.setattr(projections, "MODELS", ["m1"])
    FakeGrid.opened.clear()
    assert projections.run(store, ["m1"], "eu", log=lambda *_: None, open_grid=FakeGrid) == 1
    assert FakeGrid.opened == projections.VARS  # only ssp585 was read
    raw = store.get(projections.key("eu", "m1", "ssp585"))
    with netCDF4.Dataset("p.nc", memory=raw) as d:
        assert list(d["year"][:]) == [2015, 2016]
        assert d["GST"].dtype == np.int16 and d["GST"].filters()["zlib"]
        assert np.allclose(d["GST"][:], 19) and np.allclose(d["GSP"][:], 420)
        assert d["lat"][0] == 34 and d["lon"][0] == -25
        assert "approximated" in d.EFD_note
    key, last = reported[-1]
    assert key == "cckp-eu" and last["files"] == last["total"] == 2
    assert last["detail"] == "SSP245 1 / 1 · SSP585 1 / 1" and last["manual"]
    assert reported[0][1]["lastResult"].startswith("running: m1 ssp585 tasmax (1/6)")


def test_southern_season_is_october_to_april_and_the_year_is_the_vintage():
    steps = projections.season_steps("S")
    assert steps[:3] == [-1, -1, -1]  # 2015's season would start in October 2014: before the data
    assert steps[3:7] == [0, 1, 2, 3]  # January - April 2015
    assert steps[7:10] == [9, 10, 11]  # the 2016 vintage starts in October 2015
    assert projections.season_steps("N")[:7] == [3, 4, 5, 6, 7, 8, 9]  # April - October 2015


def test_southern_bcis_use_the_southern_months():
    shape = (1, 7, 1, 1)
    v = {k: np.zeros(shape) for k in ("hd35", "tr", "pr")}
    v["tasmax"] = np.full(shape, 30.0)
    v["tasmin"] = np.full(shape, 10.0)
    v["fd"] = np.zeros(shape)
    v["fd"][0, 0], v["fd"][0, 1] = 4, 2  # October, November
    out = projections.bcis(v, np.array([-34.0]), "S")
    days = projections.SEASONS["S"][1].sum()
    assert out["EHI"][0, 0, 0] == pytest.approx(days * 15 * 1.02)  # (20 + 30) / 2 - 10 = 15 a day, k 1.02 at 34 S
    assert out["EFD"][0, 0, 0] == pytest.approx(4)  # 0.5 x October + November
