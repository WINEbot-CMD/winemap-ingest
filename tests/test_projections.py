import netCDF4
import numpy as np

from winemap_ingest import projections
from winemap_ingest.store import LocalStore


def global_file(path, var="tas"):
    """Shaped like a CCKP climatology: (12, lat, lon), lon 0..360, value = month + lon/1000."""
    lat = np.arange(-89.875, 90, 0.25)
    lon = np.arange(0.125, 360, 0.25)
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", 12)
        d.createDimension("lat", len(lat))
        d.createDimension("lon", len(lon))
        d.createVariable("lat", "f8", ("lat",))[:] = lat
        d.createVariable("lon", "f8", ("lon",))[:] = lon
        v = d.createVariable(f"climatology-{var}-monthly-mean", "f4", ("time", "lat", "lon"))
        v[:] = np.arange(12)[:, None, None] + lon[None, None, :] / 1000
        v.units = "degC"


def test_europe_box_wraps_the_date_line_of_greenwich(tmp_path):
    global_file(tmp_path / "g.nc")
    with netCDF4.Dataset("e.nc", memory=projections.europe(tmp_path / "g.nc", "tas")) as d:
        lon, lat, v = d["lon"][:], d["lat"][:], d["tas"]
        assert lon[0] == -24.875 and lon[-1] == 44.875 and np.all(np.diff(lon) > 0)
        assert lat[0] >= 34 and lat[-1] <= 72 and v.shape == (12, len(lat), len(lon))
        assert v.units == "degC" and v.filters()["zlib"]
        west = list(lon).index(-0.125)  # 359.875 at the source
        assert abs(v[3, 0, west] - (3 + 0.359875)) < 1e-4 and abs(v[3, 0, west + 1] - (3 + 0.000125)) < 1e-4


def test_url_and_run_skip_what_is_there(tmp_path, monkeypatch):
    assert projections.url("tas", "miroc6-r1i1p1f1", "ssp585", "2080-2099").endswith(
        "tas/miroc6-r1i1p1f1-ssp585/climatology-tas-monthly-mean_cmip6-x0.25_miroc6-r1i1p1f1-ssp585"
        "_climatology_mean_2080-2099.nc"
    )
    store = LocalStore(tmp_path / "store")
    fetched = []

    def fake(url, dest):
        fetched.append(url)
        global_file(dest)

    monkeypatch.setattr(projections.urllib.request, "urlretrieve", fake)
    monkeypatch.setattr(projections, "RUNS", [("ssp245", "2040-2059")])
    monkeypatch.setattr(projections, "VARS", ["tas", "pr"])
    store.put("climate/cckp/eu/m1/ssp245_2040-2059/tas.nc", b"x")
    assert projections.run(store, ["m1"]) == 1
    assert len(fetched) == 1 and "/pr/" in fetched[0]
    assert store.get("climate/cckp/eu/m1/ssp245_2040-2059/pr.nc")[:3] == b"\x89HD"
