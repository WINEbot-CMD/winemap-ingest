# winemap-ingest

The hourly data job behind **Winemap Live / Season**: precipitation sums for the Netherlands from KNMI radar data.
The viewer itself lives in a separate (private) repository; this repository only holds the job that prepares the data.

## What it makes

Cloud-optimised GeoTIFFs on the RD New 1 km grid (EPSG:28992, 300 × 340 cells, float32 mm, NaN = no data), stored
in a Cloudflare R2 bucket:

| Key | Content |
|---|---|
| `rain/hour/YYYY/YYYY-MM-DD.tif` | 24 bands, band *i* = the hour starting at *i*:00 UTC |
| `rain/day/YYYY/YYYY-MM-DD.tif` | the precipitation day from 08:00 UTC on that date to 08:00 UTC the next day (as KNMI) |
| `rain/cum/YYYY/YYYY-MM-DD.tif` | running total from `cum_start` to the end of that day |
| `rain/mask.tif` | 1 = cells with data (the Netherlands, as in KNMI's final data) |
| `rain/index.json` | the available hours and days, and up to when the data is final |

A sum over any period (this season, or chosen days) is `cum[last day] − cum[day before the first]`: two files.

## Sources (KNMI, CC BY 4.0)

- **Final** (gauge-adjusted, climatological): `rad_nl25_rac_mfbs_01h`, one zip per year, for the backfill.
- **Real-time**: `nl_rdr_data_rtcor_5m`, 5-minute sums, added up per hour by the hourly job.

`final_until` in the index tells which hours are final; later hours are real-time (replaced by final data later).
Real-time data is cut to the same area as the final data, so sums over both stay comparable.

## Running

```sh
pip install -r requirements.txt -r requirements-dev.txt
pytest -q

export KNMI_API_KEY=...            # free key from the KNMI Data Platform
export WX_STORE=r2                 # or a local folder
export R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=winemap-wx

# once: final hourly sums from KNMI's yearly zips (downloaded into --zips)
python -m winemap_ingest.rain backfill --zips data/zips --start 2019-10-01T08 --until 2026-09-30T08
# every hour (GitHub Actions, .github/workflows/rain.yml): the newest real-time hours
python -m winemap_ingest.rain live --max-hours 48
```

The job keeps to KNMI's request limit (1 000 per hour per key) and leaves a reserve for other jobs; when the limit is
close it stops and the next run continues where it stopped.

On Windows with PostGIS installed, unset `PROJ_LIB` / `GDAL_DATA` first: the wheels bring their own PROJ data.

Data: © KNMI, CC BY 4.0.
