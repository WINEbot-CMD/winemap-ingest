import netCDF4
import numpy as np
import pytest

from winemap_ingest import remote_nc


def grid_file(path, fmt, **opts):
    """A small (time, lat, lon) file like CCKP's: value = t * 1000 + row * 10 + column / 100, one month per chunk."""
    lat, lon = np.arange(-2.0, 3.0), np.arange(-180.0, 180.0, 60.0)
    with netCDF4.Dataset(path, "w", format=fmt) as d:
        d.createDimension("time", None if fmt == "NETCDF3_CLASSIC" else 4)
        d.createDimension("lat", len(lat))
        d.createDimension("lon", len(lon))
        d.createVariable("time", "f8", ("time",))[:] = np.arange(4)
        d.createVariable("lat", "f4", ("lat",))[:] = lat
        d.createVariable("lon", "f4", ("lon",))[:] = lon
        dims = ("time", "lat", "lon")
        v = d.createVariable("timeseries-x-monthly-mean", "f4", dims, fill_value=np.float32(1e20), **opts)
        t, r, c = np.meshgrid(np.arange(4), np.arange(len(lat)), np.arange(len(lon)), indexing="ij")
        a = (t * 1000 + r * 10 + c / 100).astype("f4")
        a[2, 1, 3] = 1e20  # a gap
        v[:] = a
    return path.read_bytes()


@pytest.mark.parametrize(
    "fmt,opts",
    [
        ("NETCDF3_CLASSIC", {}),  # most CCKP variables
        ("NETCDF4", {"chunksizes": (1, 5, 6), "shuffle": True}),  # tas
        ("NETCDF4", {"chunksizes": (1, 5, 6), "zlib": True, "shuffle": True}),  # hd35
    ],
)
def test_rows_from_each_layout(tmp_path, monkeypatch, fmt, opts):
    raw = grid_file(tmp_path / "g.nc", fmt, **opts)
    monkeypatch.setattr(remote_nc, "fetch", lambda url, a, b, tries=5: raw[a : min(b, len(raw) - 1) + 1])

    class Local(remote_nc._RangeFile):
        def __init__(self, url):
            self.url, self.pos, self.size = url, 0, len(raw)

    monkeypatch.setattr(remote_nc, "_RangeFile", Local)
    g = remote_nc.RemoteGrid("https://example/g.nc")
    assert g.n_time == 4 and list(g.lat) == [-2, -1, 0, 1, 2] and g.lon[0] == -180
    rows = g.rows(2, 1, 3)  # month 2, rows 1-2
    assert rows.shape == (2, 6)
    assert np.allclose(rows[1], 2000 + 20 + np.arange(6) / 100)
    assert np.isnan(rows[0, 3]) and np.allclose(rows[0, :3], 2000 + 10 + np.arange(3) / 100)
    assert np.allclose(g.rows(3, 4, 5)[0], 3000 + 40 + np.arange(6) / 100)
