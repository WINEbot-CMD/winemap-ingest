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
    regions/eu/borders_osm.fgb       the countries (level 2) and their provinces / regions (level 4) from the same
                                     data, EPSG:3035, ~30 m: every level lines up (Jasper, 1 Nov 2026: geoBoundaries,
                                     Natural Earth and OSM side by side did not)
    regions/eu/regions_osm.geojson   the regions only, ~1 km simplified, with names: zoomed out, Analytics' regions
                                     and the search

In the workflow one job per country (a whole run in one job ran out of memory at Germany, 1 Nov 2026) writes its part,
then one job joins them:

    python -m winemap_ingest.municipalities part DE      # regions/eu/municipalities/parts/DE.fgb + DE.json
    python -m winemap_ingest.municipalities merge        # the parts -> the files above
    python -m winemap_ingest.municipalities [ISO ...]    # both, here and now (all countries, or these)
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
PARTS = PREFIX + "municipalities/parts/"
LEVELS = (5, 6, 7, 8, 9)  # 5: Lithuania's and Latvia's municipalities
# where the count alone picks wrong: Germany's level 9 (10 618) came nearer the number of Gemeinden than its level 8
# (11 160, with the unincorporated areas), but the Gemeinden are level 8 (first run, 1 Nov 2026)
LEVEL_OVERRIDE = {"DE": 8}
REGION_COVER = 0.8  # a region level must cover this much of the country (Portugal's level 4 is the two archipelagos)
READ_SIMPLIFY_DEG = 0.00005  # ~5 m: kept finer than the final 20 m, but no full-detail rings held in memory
SIMPLIFY_M = 20
BORDERS_SIMPLIFY_M = 30  # countries and regions: drawn from country zoom on
REGIONS_OVERVIEW_M = 1000  # the regions' overview file
ATTRIBUTION = "© OpenStreetMap contributors (ODbL)"

# ISO -> (Geofabrik name, about how many municipalities the country has)
COUNTRIES: dict[str, tuple[str, int]] = {
    "AD": ("andorra", 7), "AL": ("albania", 61), "BY": ("belarus", 1200), "FO": ("faroe-islands", 29),
    "IS": ("iceland", 64), "LI": ("liechtenstein", 11), "MC": ("monaco", 1), "AT": ("austria", 2093),
    "BA": ("bosnia-herzegovina", 143), "BE": ("belgium", 565),
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


def prefilter(pbf: Path, log=print) -> Path:
    """Only the administrative boundaries (with their ways and nodes) of an extract, by osmium-tool when it is there
    (the workflow installs it): France's 5 GB took Python over 90 minutes, the boundaries alone take minutes."""
    import shutil
    import subprocess

    tool = shutil.which("osmium")
    if not tool:
        return pbf
    out = pbf.with_name(pbf.stem + ".adm.pbf")
    t = time.time()
    cmd = [tool, "tags-filter", str(pbf), "r/boundary=administrative", "-o", str(out), "--overwrite"]
    subprocess.run(cmd, check=True)
    pbf.unlink(missing_ok=True)
    log(f"  boundaries only: {out.stat().st_size / 1e6:.0f} MB in {time.time() - t:.0f} s")
    return out


def boundaries(pbf: Path) -> list[dict]:
    """The administrative areas of an extract: OSM id, level, name, ISO country code (level 2), shapely geometry.
    The node locations go into a file beside the extract, not into memory (Germany's ~450 M nodes)."""
    import osmium
    from shapely import wkb

    fab = osmium.geom.WKBFactory()
    out = []
    nodes = pbf.with_suffix(".nodes")
    fp = (
        osmium.FileProcessor(str(pbf))
        .with_locations(f"sparse_file_array,{nodes}")
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
        if level not in (2, 3, 4, 5, *LEVELS):  # 2-4 for the country's outline (country_of), 4-6 its regions
            continue
        try:
            geom = wkb.loads(fab.create_multipolygon(a), hex=True).simplify(READ_SIMPLIFY_DEG, preserve_topology=True)
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
    del fp  # closes the node file (Windows keeps an open file)
    try:
        nodes.unlink(missing_ok=True)
    except PermissionError:
        pass  # the temporary folder goes anyway
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
    level = LEVEL_OVERRIDE.get(iso) or pick_level(counts, COUNTRIES[iso][1])
    chosen = [dict(a, iso=iso) for a in candidates if a["level"] == level]
    return chosen, {"level": level, "count": len(chosen), "expected": COUNTRIES[iso][1], "per_level": counts}


def borders_of(iso: str, areas: list[dict], below: int | None = None) -> list[dict]:
    """The country's outline (level 2) and its provinces / regions, from the same extract as its municipalities, so
    all share the same line pieces. Regions: the coarsest of levels 4-6 that covers most of the country (REGION_COVER;
    Luxembourg's cantons are level 6, Portugal's level 4 only the archipelagos) and is coarser than its municipalities
    (`below`); none when no level does."""
    from shapely import prepared

    country = country_of(iso, areas)
    if country is None:
        return []
    inside = prepared.prep(country)
    ours = [a for a in areas if a["level"] in (4, 5, 6) and inside.contains(a["geometry"].representative_point())]
    level = None
    for lv in (4, 5, 6):
        if below is not None and lv >= below:
            break
        here = [a for a in ours if a["level"] == lv]
        if len(here) >= 2 and sum(a["geometry"].area for a in here) >= REGION_COVER * country.area:
            level = lv
            break
    regions = [dict(a, iso=iso, level=4) for a in ours if a["level"] == level]  # shown as the region level
    return [{"id": f"c{iso}", "level": 2, "name": iso, "iso": iso, "geometry": country}, *regions]


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


def compact(features: list[dict], simplify_m: float = SIMPLIFY_M):
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
    gdf["geometry"] = gdf.geometry.simplify(simplify_m, preserve_topology=True).buffer(0)
    return gdf


def write(parts: list, work: Path) -> tuple[bytes, bytes, int]:
    """The FlatGeobuf (spatial index) and the name list; an area in two extracts is kept once."""
    import pandas as pd

    gdf = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    names = [[n, i, x, y] for n, i, x, y in zip(gdf["name"], gdf["iso"], gdf["lon"], gdf["lat"], strict=True)]
    path = work / "municipalities.fgb"
    gdf.drop(columns=["lon", "lat"]).to_file(path, driver="FlatGeobuf", SPATIAL_INDEX="YES")
    return path.read_bytes(), json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode(), len(gdf)


def country_part(store, iso: str, log=print, fetch=download) -> dict:
    """One country: its municipalities as a part (FlatGeobuf with the name points) and its report line."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        work = Path(t)
        pbf = prefilter(fetch(COUNTRIES[iso][0], work, log=log), log=log)
        t0 = time.time()
        areas = boundaries(pbf)
        pbf.unlink(missing_ok=True)
        chosen, line = municipalities_of(iso, areas)
        if chosen:
            path = work / f"{iso}.fgb"
            compact(chosen).to_file(path, driver="FlatGeobuf")
            store.put(f"{PARTS}{iso}.fgb", path.read_bytes(), "application/octet-stream")
        borders = borders_of(iso, areas, below=line.get("level"))
        if borders:
            path = work / f"{iso}_borders.fgb"
            b = compact(borders, BORDERS_SIMPLIFY_M)
            b["level"] = [f["level"] for f in borders]
            b.drop(columns=["lon", "lat"]).to_file(path, driver="FlatGeobuf")
            store.put(f"{PARTS}{iso}_borders.fgb", path.read_bytes(), "application/octet-stream")
        line["regions"] = max(0, len(borders) - 1)
        store.put(f"{PARTS}{iso}.json", json.dumps(line).encode(), "application/json")
        log(f"  {iso}: {line} ({time.time() - t0:.0f} s)")
        return line


def merge(store, log=print) -> dict:
    """The countries' parts joined into the files the viewer reads, and the report."""
    import geopandas as gpd
    import pandas as pd

    report, parts, borders = {}, [], []
    with tempfile.TemporaryDirectory() as t:
        work = Path(t)
        for key in sorted(store.list(PARTS)):
            name = key.rsplit("/", 1)[1]
            if name.endswith(".json"):
                report[name.split(".")[0]] = json.loads(store.get(key))
            elif name.endswith(".fgb"):
                path = work / name
                path.write_bytes(store.get(key))
                (borders if name.endswith("_borders.fgb") else parts).append(gpd.read_file(path))
        if not parts:
            raise SystemExit("no country parts to merge")
        fgb, names, n = write(parts, work)
        if borders:
            all_borders = pd.concat(borders, ignore_index=True).drop_duplicates("id")
            path = work / "borders_osm.fgb"
            all_borders.to_file(path, driver="FlatGeobuf", SPATIAL_INDEX="YES")
            store.put(PREFIX + "borders_osm.fgb", path.read_bytes(), "application/octet-stream")
            overview = all_borders[all_borders["level"] == 4].copy()
            overview["geometry"] = overview.geometry.simplify(REGIONS_OVERVIEW_M, preserve_topology=True)
            path = work / "regions_osm.geojson"
            overview[["id", "name", "iso", "geometry"]].to_file(path, driver="GeoJSON", COORDINATE_PRECISION=0)
            store.put(PREFIX + "regions_osm.geojson", path.read_bytes(), "application/geo+json")
    store.put(PREFIX + "municipalities.fgb", fgb, "application/octet-stream")
    store.put(PREFIX + "municipalities.json", names, "application/json")
    summary = {"source": ATTRIBUTION, "updated": time.strftime("%Y-%m-%d"), "countries": report}
    store.put(PREFIX + "municipalities_report.json", json.dumps(summary, indent=1).encode(), "application/json")
    log(f"{n} municipalities in {len(parts)} countries, {len(fgb) / 1e6:.0f} MB")
    return summary


def run(store, only: list[str] | None = None, log=print, fetch=download) -> dict:
    """All countries (or these) and the merge, in one go."""
    for iso in only or sorted(COUNTRIES):
        country_part(store, iso, log=log, fetch=fetch)
    return merge(store, log=log)


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["part"]:
        country_part(open_store(), args[1].upper())
    elif args[:1] == ["merge"]:
        merge(open_store())
    else:
        run(open_store(), only=[a.upper() for a in args] or None)


if __name__ == "__main__":
    main()
