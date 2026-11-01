import json

from shapely.geometry import box

from winemap_ingest import municipalities as mu
from winemap_ingest.store import LocalStore


def area(id_, level, x0, y0, x1, y1, name="", iso="", iso2=""):
    return {"id": id_, "level": level, "name": name, "iso": iso, "iso2": iso2, "geometry": box(x0, y0, x1, y1)}


def test_pick_level_takes_the_count_nearest_the_expected_number():
    assert mu.pick_level({6: 12, 7: 0, 8: 340, 9: 2500}, 342) == 8
    assert mu.pick_level({6: 20, 7: 300, 8: 3000, 9: 0}, 308) == 7  # Portugal's concelhos are level 7
    assert mu.pick_level({6: 0, 7: 0, 8: 0, 9: 0}, 100) is None


def test_country_outline_falls_back_to_level_3_then_the_regions():
    whole = [area("r1", 2, 0, 0, 10, 10, iso="LU")]
    assert mu.country_of("LU", whole).bounds == (0, 0, 10, 10)
    europe_part = [area("r2", 3, 0, 0, 5, 5, iso2="NL-X"), area("r3", 4, 0, 0, 1, 1, iso2="NL-ZE")]
    assert mu.country_of("NL", europe_part).bounds == (0, 0, 5, 5)
    regions = [area("r4", 4, 0, 0, 1, 1, iso2="FR-A"), area("r5", 4, 1, 0, 2, 1, iso2="FR-B")]
    assert mu.country_of("FR", regions).bounds == (0, 0, 2, 1)
    assert mu.country_of("DE", regions) is None


def test_municipalities_are_the_chosen_level_inside_the_country(monkeypatch):
    areas = [
        area("c", 2, 0, 0, 10, 10, iso="LU"),
        *[area(f"m{i}", 8, i, 0, i + 1, 1, name=f"Gemeng {i}") for i in range(9)],
        area("out", 8, 20, 0, 21, 1, name="Over the border"),
        area("d", 6, 0, 0, 10, 5, name="District"),
    ]
    monkeypatch.setitem(mu.COUNTRIES, "LU", ("luxembourg", 10))
    chosen, line = mu.municipalities_of("LU", areas)
    assert line["level"] == 8 and line["count"] == 9
    assert {f["iso"] for f in chosen} == {"LU"} and "Over the border" not in {f["name"] for f in chosen}


def test_run_writes_the_file_the_names_and_the_report(tmp_path, monkeypatch):
    store = LocalStore(tmp_path / "store")
    lon0, lat0 = 6.0, 49.6  # Luxembourg: real coordinates, written in EPSG:3035
    d = 0.01
    areas = [area("c", 2, lon0, lat0, lon0 + 0.2, lat0 + 0.2, iso="LU")]
    areas += [area(f"m{i}", 8, lon0 + i * d, lat0, lon0 + (i + 1) * d, lat0 + d, name=f"Gemeng {i}") for i in range(10)]
    monkeypatch.setattr(mu, "boundaries", lambda pbf: areas)
    monkeypatch.setitem(mu.COUNTRIES, "LU", ("luxembourg", 10))
    fake = lambda name, to, log=print: (to / "x.pbf").write_bytes(b"") or to / "x.pbf"  # noqa: E731
    summary = mu.run(store, only=["LU"], log=lambda *_: None, fetch=fake)
    assert summary["countries"]["LU"]["count"] == 10 and "OpenStreetMap" in summary["source"]
    names = json.loads(store.get(mu.PREFIX + "municipalities.json"))
    assert names[0][0].startswith("Gemeng") and names[0][1] == "LU" and 6.0 < names[0][2] < 6.2
    import pyogrio

    borders = pyogrio.read_dataframe(tmp_path / "store" / mu.PREFIX / "borders_osm.fgb")
    assert list(borders["level"]) == [2] and borders["iso"][0] == "LU"  # the country (no level-4 regions here)
    info = pyogrio.read_info(tmp_path / "store" / mu.PREFIX / "municipalities.fgb")
    assert info["features"] == 10 and "3035" in info["crs"]
