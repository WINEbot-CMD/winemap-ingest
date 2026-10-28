from datetime import UTC, date, datetime, timedelta

import numpy as np

from winemap_ingest import hourly
from winemap_ingest.store import LocalStore

SAMPLE = """# SOURCE: ROYAL NETHERLANDS METEOROLOGICAL INSTITUTE (KNMI)
# STN         LON(east)   LAT(north)  ALT(m)      NAME
# 260         5.180       52.100      1.90        De Bilt
# 310         3.596       51.442      8.00        Vlissingen
# STN,YYYYMMDD,HH,    T,   TD,    U,   RH,   DR
  260,20240421,    1,   48,    6,   74,    0,    0
  260,20240421,    4,  -12,   19,   90,   -1,    2
  310,20240421,    1,     ,     ,     ,     ,
"""


def test_station_text_to_rows():
    stations, rows = hourly.parse_hourly(SAMPLE)
    assert stations == [[260, 5.18, 52.1, 1.9, "De Bilt"], [310, 3.596, 51.442, 8.0, "Vlissingen"]]
    assert rows == [
        [260, "2024-04-21T01:00Z", 4.8, 0.6, 74, 0.0, 0.0],
        [260, "2024-04-21T04:00Z", -1.2, 1.9, 90, 0.0, 0.2],  # RH -1 = less than 0.05 mm
    ]  # the empty station is left out


def test_stations_are_stored_per_year(tmp_path):
    store = LocalStore(tmp_path)
    n = hourly.run_stations(store, years=(2001, 2000), fetch=lambda a, b: SAMPLE, log=lambda *_: None)
    assert n == 2
    assert store.get("climate/knmi-stations/stations.csv").decode().startswith("stn,lon,lat,alt,name")
    assert hourly.run_stations(store, years=(2001, 2000), fetch=lambda a, b: SAMPLE, log=lambda *_: None) == 0


def test_era5_rain_per_hour():
    start = datetime(2024, 5, 1, 0, tzinfo=UTC)
    hours = [start + timedelta(hours=i) for i in range(26)]
    # a running sum from 00 UTC: 1 mm an hour; the 00 UTC value holds the whole day before (24 mm)
    acc = np.array([24.0] + [float(h) for h in range(1, 24)] + [24.0, 1.0])[:, None, None]
    per = hourly.deaccumulate(acc, hours)[:, 0, 0]
    assert np.isnan(per[0])  # no hour before it in this month file
    assert np.allclose(per[1:], 1.0)


def test_era5_months_newest_first_season_only():
    got = list(hourly.months_newest_first(2023, date(2024, 6, 30), None))
    assert got[:3] == [(2024, 6), (2024, 5), (2024, 4)] and got[3] == (2023, 10) and (2023, 3) not in got
