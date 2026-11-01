"""Municipality borders for Europe (Winemap, Jasper 1 Nov 2026: white borders on the aerial photo, and municipalities to
pick in Analytics) from OpenStreetMap (ODbL: credit "© OpenStreetMap contributors"). Overture's divisions mix
municipalities with villages and Eurostat's LAU is not for commercial use, so the borders come from OSM itself.

Per country the Geofabrik extract; from it the administrative boundaries at levels 6-9. Which level is the
municipality differs per country (8 in most, 7 for Portuguese concelhos, Polish gminy and Nordic kommuner, ...): the
level whose count is closest to the country's number of municipalities is taken, and the report says which and how
many. Kept: the areas whose inside point lies in the country (the extracts reach over the borders).

    regions/eu/municipalities.fgb    EPSG:3035, ~20 m simplified, spatial index (the viewer reads what is in view)
    regions/eu/municipalities.json   name search: [name, iso, lon, lat] per municipality
    regions/eu/municipalities_report.json   per country: level, count, expected, source date

    python -m winemap_ingest.municipalities [ISO ...]     # all countries, or these; monthly in the workflow
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from .store import open_store

GEOFABRIK = "https://download.geofabrik.de/europe/{}-latest.osm.pbf"
PREFIX = "regions/eu/"
LEVELS = (6, 7, 8, 9)
SIMPLIFY_M = 20
ATTRIBUTION = "© OpenStreetMap contributors (ODbL)"

# ISO -> (Geofabrik name, about how many municipalities the country has)
COUNTRIES: dict[str, tuple[str, int]] = {
    "AL": ("albania", 61), "AT": ("austria", 2093), "BA": ("bosnia-herzegovina", 143), "BE": ("belgium", 565),
    "BG": ("bulgaria", 265), "CH": ("switzerland", 2121), "CY": ("cyprus", 380), "CZ": ("czech-republic", 6258),
    "DE": ("germany", 10751), "DK": ("denmark", 98), "EE": ("estonia", 79), "ES": ("spain", 8132),
    "FI": ("finland", 308), "FR": ("france", 34935), "GB": ("united-kingdom", 360), "GE": ("georgia", 69),
    "GR": ("greece", 332), "HR": ("croatia", 556), "HU": ("hungary", 3155), "IE": ("ireland-and-northern-ireland", 31),
    "IT": ("italy", 7896), "LT": ("lithuania", 60), "LU": ("luxembourg", 100), "LV": ("latvia", 43),
    "MD": ("moldova", 898), "ME": ("montenegro", 25), "MK": ("macedonia", 80), "MT": ("malta", 68),
    "NL": ("netherlands", 342), "NO": ("norway", 357), "PL": ("poland", 2477), "PT": ("portugal", 308),
    "RO": ("romania", 3181), "RS": ("serbia", 174), "SE": ("sweden", 290), "SI": ("slovenia", 212),
    "SK": ("slovakia", 2890), "TR": ("turkey", 973), "UA": ("ukraine", 1469), "XK": ("kosovo", 38),
}  # fmt: skip


def boundaries(pbf: Path) -> list[dict]:
    """The administrative areas of an extract: OSM id, level, name, ISO country code (level 2), shapely geometry."""
    import osmium
    from shapely import wkb

    fab = osmium.geom.WKBFactory()
    out = []
    fp = (
        osmium.FileProcessor(str(pbf))
        .with_areas(osmium.filter.TagFilter(("boundary", "administrative")))
        .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA))
        .with_filter(osmium.filter.TagFilter(("boundary", "administrative")))
    )
    for a in fp:
        tags = a.tags
        try:
            level = int(tags.get("admin_level", ""))
        except ValueError:
            continue
        if level not in (2, 3, 4, *LEVELS):  # 2-4 for the country's outline, see country_of
            continue
        try:
            geom = wkb.loads(fab.create_multipolygon(a), hex=True)
        except Exception:  # noqa: BLE001 - a broken ring (cut off at the extract's edge): skipped
            continue
        out.append({
            "id": f"{'w' if a.from_way() else 'r'}{a.orig_id()}",
            "level": level,
            "name": tags.get("name", ""),
            "iso": tags.get("ISO3166-1", tags.get("ISO3166-1:alpha2", "")),
            "iso2": tags.get("ISO3166-2", ""),
            "geometry": geom,
        })  # fmt: skip
    return out


def pick_level(counts: dict[int, int], expected: int) -> int | None:
    """The level whose count is closest to the expected number of municipalities (by ratio, so 1000 vs 1100 is as near
    as 100 vs 110); None when no level has any."""
    have = {lv: n for lv, n in counts.items() if n}
    if not have:
        return None
    return min(have, key=lambda lv: abs(np.log(have[lv] / expected)))


def country_of(iso: str, areas: list[dict]):
    """The country's outline: its level-2 area; when that is not whole in the extract (a kingdom or republic with parts
    overseas: NL, FR, ES, PT, DK, NO) its level-3 area, else its level-4 regions together. None when none is there."""
    from shapely import union_all

    ours = [a for a in areas if a["iso"] == iso or a["iso2"].startswith(f"{iso}-")]
    for level in (2, 3):
        a = next((a for a in ours if a["level"] == level), None)
        if a:
            return a["geometry"]
    parts = [a["geometry"] for a in ours if a["level"] == 4]
    return union_all(parts) if parts else None


def municipalities_of(iso: str, areas: list[dict]) -> tuple[list[dict], dict]:
    """The country's municipalities among an extract's areas, and the line for the report."""
    from shapely import prepared

    country = country_of(iso, areas)
    if country is None:
        return [], {"error": "no country outline in the extract"}
    inside = prepared.prep(country)
    candidates = [a for a in areas if a["level"] in LEVELS and inside.contains(a["geometry"].representative_point())]
    counts = {lv: sum(a["level"] == lv for a in candidates) for lv in LEVELS}
    level = pick_level(counts, COUNTRIES[iso][1])
    chosen = [dict(a, iso=iso) for a in candidates if a["level"] == level]
    return chosen, {"level": level, "count": len(chosen), "expected": COUNTRIES[iso][1], "per_level": counts}


def download(name: str, to: Path, log=print) -> Path:
    import shutil
    import ssl

    import certifi

    path = to / f"{name}.osm.pbf"
    t = time.time()
    # certifi's roots: Windows' own store lacked Let's Encrypt's newer chain (Geofabrik, Nov 2026)
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(GEOFABRIK.format(name), timeout=600, context=ctx) as r, path.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    log(f"  {name}: {path.stat().st_size / 1e6:.0f} MB in {time.time() - t:.0f} s")
    return path


def compact(features: list[dict]):
    """A country's municipalities as they are kept: EPSG:3035, ~20 m simplified, with their name point (lon / lat), so
    only the small result stays in memory while the next country is read."""
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"id": [f["id"] for f in features], "name": [f["name"] for f in features], "iso": [f["iso"] for f in features]},
        geometry=[f["geometry"] for f in features],
        crs="EPSG:4326",
    )
    pts = gdf.geometry.representative_point()
    gdf["lon"], gdf["lat"] = pts.x.round(4), pts.y.round(4)
    gdf = gdf.to_crs("EPSG:3035")
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_M, preserve_topology=True).buffer(0)
    return gdf


def write(parts: list, work: Path) -> tuple[bytes, bytes, int]:
    """The FlatGeobuf (spatial index) and the name list; an area in two extracts is kept once."""
    import pandas as pd

    gdf = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    names = [[n, i, x, y] for n, i, x, y in zip(gdf["name"], gdf["iso"], gdf["lon"], gdf["lat"], strict=True)]
    path = work / "municipalities.fgb"
    gdf.drop(columns=["lon", "lat"]).to_file(path, driver="FlatGeobuf", SPATIAL_INDEX="YES")
    return path.read_bytes(), json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode(), len(gdf)


def run(store, only: list[str] | None = None, log=print, fetch=download) -> dict:
    report, parts = {}, []
    with tempfile.TemporaryDirectory() as t:
        work = Path(t)
        for iso in only or sorted(COUNTRIES):
            pbf = fetch(COUNTRIES[iso][0], work, log=log)
            t0 = time.time()
            chosen, line = municipalities_of(iso, boundaries(pbf))
            pbf.unlink(missing_ok=True)
            report[iso] = line
            if chosen:
                parts.append(compact(chosen))
            log(f"  {iso}: {line} ({time.time() - t0:.0f} s)")
        fgb, names, n = write(parts, work)
    store.put(PREFIX + "municipalities.fgb", fgb, "application/octet-stream")
    store.put(PREFIX + "municipalities.json", names, "application/json")
    summary = {"source": ATTRIBUTION, "updated": time.strftime("%Y-%m-%d"), "countries": report}
    store.put(PREFIX + "municipalities_report.json", json.dumps(summary, indent=1).encode(), "application/json")
    log(f"{n} municipalities, {len(fgb) / 1e6:.0f} MB")
    return summary


def main() -> None:
    run(open_store(), only=[a.upper() for a in sys.argv[1:]] or None)


if __name__ == "__main__":
    main()
