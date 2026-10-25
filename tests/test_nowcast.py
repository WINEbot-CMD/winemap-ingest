import io
import json
from datetime import UTC, datetime, timedelta

import h5py
import numpy as np

from winemap_ingest import grid, nowcast, rain
from winemap_ingest.store import LocalStore


def nowcast_h5(run: datetime, mm5: float) -> bytes:
    b = io.BytesIO()
    with h5py.File(b, "w") as f:
        f.create_group("overview").attrs["product_datetime_start"] = np.bytes_(
            run.strftime("%d-%b-%Y;%H:%M:%S.000").upper()
        )
        for i in range(25):
            g = f.create_group(f"image{i + 1}")
            g.attrs["image_datetime_valid"] = np.bytes_(
                (run + timedelta(minutes=5 * i)).strftime("%d-%b-%Y;%H:%M:%S.000").upper()
            )
            g.create_dataset("image_data", data=np.full((grid.RADAR_H, grid.RADAR_W), round(mm5 * 100), np.uint16))
            c = g.create_group("calibration")
            c.attrs["calibration_formulas"] = np.bytes_(b"GEO=0.010000*PV+0.000000")
            c.attrs["calibration_out_of_image"] = np.array([65535])
    return b.getvalue()


class Fake:
    def __init__(self, name, data):
        self.name, self.data, self.downloads = name, data, 0

    def list_files_desc(self, dataset, limit=1):
        return [self.name]

    def download(self, dataset, name):
        self.downloads += 1
        return self.data


def test_nowcast_run_becomes_a_cog_with_an_index_and_old_runs_go(tmp_path):
    store = LocalStore(tmp_path / "wx")
    store.put("rain/nowcast/202610231000.tif", b"old")
    run = datetime(2026, 10, 25, 14, 45, tzinfo=UTC)
    fake = Fake("RAD_NL25_RAC_FM_202610251445.h5", nowcast_h5(run, 0.1))
    index = nowcast.update(store, fake, now=run + timedelta(minutes=3), log=lambda *_: None)
    assert index["file"] == "nowcast/202610251445.tif"
    assert len(index["steps"]) == 25 and index["steps"][0] == "2026-10-25T14:45:00Z"
    data = rain.read_cog(store.get("rain/" + index["file"]))
    assert data.shape == (25, grid.HEIGHT, grid.WIDTH)
    assert np.allclose(data[:, 100, 100], 1.2)  # 0.1 mm per 5 minutes = 1.2 mm/h
    assert store.get("rain/nowcast/202610231000.tif") is None  # older than a day
    assert json.loads(store.get("rain/nowcast/index.json"))["source"] == fake.name
    # the same run again: no download
    nowcast.update(store, fake, now=run + timedelta(minutes=6), log=lambda *_: None)
    assert fake.downloads == 1
