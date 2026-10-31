import numpy as np
from rasterio.transform import from_origin

from winemap_ingest import projection_maps as pm


def test_delta_adds_the_change_rain_as_a_ratio_counts_not_below_zero():
    base = np.array([[1500.0, 400.0]])
    assert np.allclose(pm.delta(base, np.array([[[100.0, -50.0]]]), "EHI"), [[[1600.0, 350.0]]])
    assert np.allclose(pm.delta(base, np.array([[[1.1, 0.5]]]), "GSP"), [[[1650.0, 200.0]]])
    assert np.allclose(pm.delta(np.array([[2.0, 0.5]]), np.array([[[-3.0, 1.0]]]), "EHD"), [[[0.0, 1.5]]])


def test_to_grid_puts_the_coarse_cells_on_the_fine_grid_north_up():
    lat = np.array([50.0, 50.25])  # south to north, as the projection files
    lon = np.array([5.0, 5.25])
    a = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, lat, lon): north row = 3, 4
    grid = pm.Grid("EPSG:4326", from_origin(4.9, 50.35, 0.1, 0.1), 5, 5)
    out = pm.to_grid(a, lat, lon, grid, pm.Resampling.nearest)[0]
    assert out[0, 0] == 3 and out[0, -1] == 4  # the top row is the north
    assert out[-1, 0] == 1 and out[-1, -1] == 2


def test_only_the_years_from_2026():
    assert pm.FIRST == 2026 and list(pm.BASE) == list(range(2019, 2026))
