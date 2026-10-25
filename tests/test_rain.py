import io
import zipfile
from datetime import UTC, datetime, timedelta

import h5py
import numpy as np
import pytest

from winemap_ingest import grid, rain
from winemap_ingest.radar import end_time, read_mm
from winemap_ingest.store import LocalStore

H = timedelta(hours=1)


def h5(mm: float, missing: bool = False, nl_only: bool = False) -> bytes:
    """A KNMI-like radar file with the same accumulation everywhere (or only in the eastern part, like the final
    data that covers the Netherlands only)."""
    raw = np.full((grid.RADAR_H, grid.RADAR_W), 65535 if missing else round(mm * 100), np.uint16)
    if nl_only:
        raw[:, :400] = 65535
    b = io.BytesIO()
    with h5py.File(b, "w") as f:
        d = f.create_dataset("image1/image_data", data=raw)
        del d
        cal = f["image1"].create_group("calibration")
        cal.attrs["calibration_formulas"] = np.bytes_(b"GEO=0.01*PV+0.0")
        cal.attrs["calibration_missing_data"] = np.array([65534])
        cal.attrs["calibration_out_of_image"] = np.array([65535])
    return b.getvalue()


def test_reads_calibrated_mm_and_end_time():
    mm = read_mm(h5(1.25))
    assert mm.shape == (765, 700)
    assert np.allclose(mm, 1.25)
    assert np.isnan(read_mm(h5(0, missing=True))).all()
    assert end_time("RAD_NL25_RAC_RT_202610250950.h5") == datetime(2026, 10, 25, 9, 50, tzinfo=UTC)
    assert end_time("RAD_NL25_RAC_MFBS_01H_202601232400_NL.h5") == datetime(2026, 1, 24, 0, 0, tzinfo=UTC)


def test_rain_day_starts_at_08_utc():
    assert str(rain.rain_day(datetime(2026, 10, 25, 7, tzinfo=UTC))) == "2026-10-24"
    assert str(rain.rain_day(datetime(2026, 10, 25, 8, tzinfo=UTC))) == "2026-10-25"


@pytest.fixture
def final_zip(tmp_path):
    """Three days of final hourly files, 0.5 mm every hour (12 mm per day)."""
    p = tmp_path / "RADNL_CLIM____MFBSNL25_01H_20260101T000000_20270101T000000_0002.zip"
    t = datetime(2026, 9, 1, 0, tzinfo=UTC)
    with zipfile.ZipFile(p, "w") as z:
        while t < datetime(2026, 9, 4, 12, tzinfo=UTC):
            end = t + H
            z.writestr(
                f"RAD_NL25_RAC_MFBS_01H/2026/09/RAD_NL25_RAC_MFBS_01H_{end:%Y%m%d%H%M}_NL.h5", h5(0.5, nl_only=True)
            )
            t = end
    return p


def test_backfill_makes_hours_days_and_running_total(tmp_path, final_zip):
    store = LocalStore(tmp_path / "wx")
    start = datetime(2026, 9, 1, 8, tzinfo=UTC)
    until = datetime(2026, 9, 4, 8, tzinfo=UTC)
    index = rain.backfill(store, [final_zip], start, until, log=lambda *_: None)
    assert index["hour"] == {"first": "2026-09-01T08", "last": "2026-09-04T07"}
    assert index["final_until"] == "2026-09-04T08"
    assert index["day"] == {"first": "2026-09-01", "last": "2026-09-03"}
    hours = rain.read_cog(store.get("rain/hour/2026/2026-09-01.tif"))
    assert hours.shape == (24, grid.HEIGHT, grid.WIDTH)
    nl = ~np.isnan(rain.read_cog(store.get("rain/mask.tif"))[0])
    assert 0 < nl.mean() < 1
    assert np.isnan(hours[:8]).all() and np.allclose(hours[8:][:, nl], 0.5)  # before the start: no data
    assert np.allclose(rain.read_cog(store.get("rain/day/2026/2026-09-02.tif"))[0][nl], 12)
    cum = rain.read_cog(store.get("rain/cum/2026/2026-09-03.tif"))[0]
    assert np.allclose(cum[nl], 36) and np.isnan(cum[~nl]).all()  # no data outside: not 0
    assert index["missing_hours"] == {}


def backfill(store, final_zip):
    start, until = datetime(2026, 9, 1, 8, tzinfo=UTC), datetime(2026, 9, 4, 8, tzinfo=UTC)
    return rain.backfill(store, [final_zip], start, until, log=lambda *_: None)


class FakeKnmi:
    """Real-time 5-minute files: 0.1 mm each (1.2 mm per hour), available up to `upto`."""

    def __init__(self, upto: datetime, skip: set[datetime] = frozenset()):
        self.files = []
        t = datetime(2026, 9, 4, 8, 5, tzinfo=UTC)
        while t <= upto:
            if t not in skip:
                self.files.append(rain.rt_name(t))
            t += timedelta(minutes=5)
        self.downloads = 0

    def list_files(self, dataset, begin=None, limit=500):
        return [f for f in self.files if begin is None or f > begin][:limit]

    def download(self, dataset, name):
        self.downloads += 1
        return h5(0.1)


def test_live_adds_complete_hours_then_days(tmp_path, final_zip):
    store = LocalStore(tmp_path / "wx")
    rain.backfill(
        store,
        [final_zip],
        datetime(2026, 9, 1, 8, tzinfo=UTC),
        datetime(2026, 9, 4, 8, tzinfo=UTC),
        log=lambda *_: None,
    )
    # files up to 5 Sep 08:20: hours 4 Sep 08 .. 5 Sep 07 are complete, 5 Sep 08 is not yet
    knmi = FakeKnmi(datetime(2026, 9, 5, 8, 20, tzinfo=UTC))
    now = datetime(2026, 9, 5, 8, 25, tzinfo=UTC)
    index = rain.live(store, knmi, max_hours=48, now=now, log=lambda *_: None)
    assert index["hour"]["last"] == "2026-09-05T07"
    assert index["final_until"] == "2026-09-04T08"  # the new hours are real-time
    assert index["day"]["last"] == "2026-09-04"
    nl = ~np.isnan(rain.read_cog(store.get("rain/mask.tif"))[0])
    day = rain.read_cog(store.get("rain/day/2026/2026-09-04.tif"))[0]
    assert np.allclose(day[nl], 24 * 1.2) and np.isnan(day[~nl]).all()  # real-time data cut to the same area
    assert np.allclose(rain.read_cog(store.get("rain/cum/2026/2026-09-04.tif"))[0][nl], 36 + 24 * 1.2)
    # nothing new: no downloads, nothing changes
    again = FakeKnmi(datetime(2026, 9, 5, 8, 20, tzinfo=UTC))
    rain.live(store, again, now=now, log=lambda *_: None)
    assert again.downloads == 0


def test_live_scales_an_hour_with_a_few_files_missing_once_it_is_late(tmp_path, final_zip):
    store = LocalStore(tmp_path / "wx")
    rain.backfill(
        store,
        [final_zip],
        datetime(2026, 9, 1, 8, tzinfo=UTC),
        datetime(2026, 9, 4, 8, tzinfo=UTC),
        log=lambda *_: None,
    )
    skip = {datetime(2026, 9, 4, 8, 10, tzinfo=UTC), datetime(2026, 9, 4, 8, 15, tzinfo=UTC)}
    knmi = FakeKnmi(datetime(2026, 9, 4, 12, 0, tzinfo=UTC), skip)
    index = rain.live(store, knmi, now=datetime(2026, 9, 4, 13, 0, tzinfo=UTC), log=lambda *_: None)
    assert index["hour"]["last"] == "2026-09-04T11"
    hours = rain.read_cog(store.get("rain/hour/2026/2026-09-04.tif"))
    nl = ~np.isnan(rain.read_cog(store.get("rain/mask.tif"))[0])
    assert np.allclose(hours[8][nl], 1.2)  # 10 of 12 files, scaled up


# ---------- final data replaces real-time hours ----------
def test_final_data_replaces_real_time_hours_and_remakes_the_totals(tmp_path, final_zip):
    store = LocalStore(tmp_path / "wx")
    backfill(store, final_zip)
    rain.live(
        store,
        FakeKnmi(datetime(2026, 9, 5, 8, 20, tzinfo=UTC)),
        now=datetime(2026, 9, 5, 9, tzinfo=UTC),
        log=lambda *_: None,
    )
    # KNMI's new archive: 2 mm in every hour of 4 Sep 08 .. 5 Sep 07 (the real-time data said 1.2)
    new = tmp_path / "new.zip"
    with zipfile.ZipFile(new, "w") as z:
        t = datetime(2026, 9, 4, 8, tzinfo=UTC)
        while t < datetime(2026, 9, 5, 8, tzinfo=UTC):
            end = t + H
            z.writestr(
                f"RAD_NL25_RAC_MFBS_01H/2026/09/RAD_NL25_RAC_MFBS_01H_{end:%Y%m%d%H%M}_NL.h5", h5(2.0, nl_only=True)
            )
            t = end
    name = "RADNL_CLIM____MFBSNL25_01H_20260101T000000_20270101T000000_0002.zip"

    class Archive:
        def file_info(self, dataset, limit=100):
            return {name: "2026-10-15T00:16:02+00:00"}

        def download(self, dataset, n):
            return new.read_bytes()

    index = rain.refresh_final(
        store, Archive(), tmp_path / "zips", now=datetime(2026, 10, 16, tzinfo=UTC), log=lambda *_: None
    )
    assert index["final_until"] == "2026-09-05T08"
    nl = ~np.isnan(rain.read_cog(store.get("rain/mask.tif"))[0])
    assert np.allclose(rain.read_cog(store.get("rain/day/2026/2026-09-04.tif"))[0][nl], 48)
    assert np.allclose(rain.read_cog(store.get("rain/cum/2026/2026-09-04.tif"))[0][nl], 36 + 48)
    # the same archive again: nothing to do
    again = rain.refresh_final(
        store, Archive(), tmp_path / "zips", now=datetime(2026, 10, 17, tzinfo=UTC), log=lambda *_: None
    )
    assert again["final_until"] == "2026-09-05T08"
