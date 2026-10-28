import json
from datetime import UTC, date, datetime, timedelta

import numpy as np

from winemap_ingest import alerts, grid, rain
from winemap_ingest.store import LocalStore

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
PLACE = {"id": "p1", "name": "Blok B7", "type": "parcel", "x": 154500, "y": 436000}


def field(rate_from: int | None, rate=2.0):
    """A nowcast: 25 steps of 5 minutes; from step `rate_from` on `rate` mm/h at every cell."""
    f = np.zeros((25, grid.HEIGHT, grid.WIDTH), np.float32)
    if rate_from is not None:
        f[rate_from:] = rate
    return f, [NOW + timedelta(minutes=5 * i) for i in range(25)]


def setup(tmp_path):
    store = LocalStore(tmp_path)
    store.put(alerts.PLACES, json.dumps({"places": [PLACE]}).encode())
    sent = []
    return store, sent, lambda topic, title, msg, prio="default", tags="": sent.append((title, msg))


def test_rain_soon_warns_once_and_again_after_a_dry_spell(tmp_path):
    store, sent, send = setup(tmp_path)
    dry = lambda x, y, now: (None, None, None)  # noqa: E731
    run = lambda t, f: alerts.run(store, now=t, rain_field=f, forecast=dry, send=send, log=lambda *_: None)  # noqa: E731
    run(NOW, field(None))
    assert sent == []
    run(NOW + timedelta(minutes=5), field(4))  # starts in 20 minutes (relative to the field's own steps)
    assert sent and sent[0][0] == "Regen bij Blok B7" and "mm in de komende 2 uur" in sent[0][1]
    run(NOW + timedelta(minutes=10), field(2))
    assert len(sent) == 1  # still raining: no new message
    for i in range(3):
        run(NOW + timedelta(minutes=15 + 5 * i), field(None))
    run(NOW + timedelta(minutes=40), field(0))
    assert len(sent) == 2  # dry for 15 minutes, then rain again: a new message


def test_frost_and_heavy_rain_once_an_hour(tmp_path):
    store, sent, send = setup(tmp_path)
    cold = lambda x, y, now: ((-1.4, NOW + timedelta(hours=18)), (8.0, NOW), 25.0)  # noqa: E731
    alerts.run(store, now=NOW, rain_field=field(None), forecast=cold, send=send, log=lambda *_: None)
    titles = sorted(t for t, _ in sent)
    assert titles == ["Veel regen bij Blok B7", "Vorst bij Blok B7"]
    assert any("-1.4 °C" in m for _, m in sent)
    alerts.run(
        store, now=NOW + timedelta(minutes=30), rain_field=field(None), forecast=cold, send=send, log=lambda *_: None
    )
    assert len(sent) == 2  # the hourly check waits; and the warning is already active


def quiet(store, send, t, forecast, places=None, sums=None):
    if places is not None:
        store.put(alerts.PLACES, json.dumps({"places": places}).encode())
    return alerts.run(
        store, now=t, rain_field=field(None), forecast=forecast, sums=sums, send=send, log=lambda *_: None
    )


def test_frost_after_budburst_is_urgent_and_limits_come_from_the_place(tmp_path):
    store, sent, send = setup(tmp_path)
    prios = []
    send2 = lambda topic, title, msg, prio="default", tags="": (sent.append((title, msg)), prios.append(prio))  # noqa: E731
    chilly = lambda x, y, now: ((1.0, NOW + timedelta(hours=6)), (12.0, NOW), 0.0)  # noqa: E731
    quiet(store, send2, NOW, chilly)
    assert sent == []  # 1 °C: above the default limit of 0 °C
    quiet(store, send2, NOW + timedelta(hours=1), chilly, places=[{**PLACE, "frostC": 2, "budburst": "2026-04-12"}])
    assert sent[0][0] == "Vorst bij Blok B7" and "jonge scheuten" in sent[0][1] and prios == ["urgent"]


def test_heat_warns_and_warns_again_when_it_gets_worse(tmp_path):
    store, sent, send = setup(tmp_path)
    hot = lambda c: lambda x, y, now: ((15.0, NOW), (c, NOW + timedelta(hours=30)), 0.0)  # noqa: E731
    quiet(store, send, NOW, hot(31.0))
    assert [t for t, _ in sent] == ["Hitte bij Blok B7"]
    quiet(store, send, NOW + timedelta(hours=1), hot(32.0))
    assert len(sent) == 1  # still hot, not worse
    quiet(store, send, NOW + timedelta(hours=2), hot(36.0))
    assert sent[-1][0] == "Extreme hitte bij Blok B7"
    quiet(store, send, NOW + timedelta(hours=3), hot(25.0))
    quiet(store, send, NOW + timedelta(hours=4), hot(30.5), places=[{**PLACE, "heatC": 33}])
    assert len(sent) == 2  # over again, and 30.5 °C is below this place's own limit


class FakeSums:
    def __init__(self, mm):
        self.mm = mm

    def since(self, day, c):
        return self.mm.get(day.isoformat())


def test_spray_again_once_per_spray_past_its_washoff_limit(tmp_path):
    store, sent, send = setup(tmp_path)
    mild = lambda x, y, now: (None, None, None)  # noqa: E731
    sprays = [{"date": "2026-10-20", "washoff": 25}, {"date": "2026-10-24"}]
    place = {**PLACE, "sprays": sprays}
    sums = FakeSums({"2026-10-20": 18.0, "2026-10-24": 9.0})
    quiet(store, send, NOW, mild, places=[place], sums=sums)
    assert sent == []  # 18 < 25 mm, 9 < 15 mm
    sums.mm["2026-10-24"] = 16.2
    quiet(store, send, NOW + timedelta(hours=1), mild, sums=sums)
    assert sent == [("Opnieuw spuiten? Blok B7", "16 mm regen sinds je bespuiting op 24 okt (afspoelgrens 15 mm).")]
    quiet(store, send, NOW + timedelta(hours=2), mild, sums=sums)
    assert len(sent) == 1  # once per spray
    state = quiet(store, send, NOW + timedelta(hours=3), mild, places=[PLACE], sums=sums)
    assert not any(":spray:" in k for k in state["active"])  # sprays gone from the list: forgotten


def test_rain_since_a_day_from_the_running_totals_and_hours(tmp_path):
    store = LocalStore(tmp_path)
    shape = (grid.HEIGHT, grid.WIDTH)
    index = rain.new_index(datetime(2026, 10, 1, 8, tzinfo=UTC))
    index["hour"] = {"first": "2026-10-01T00", "last": "2026-10-26T10"}
    index["day"] = {"first": "2026-10-01", "last": "2026-10-24"}  # rain day 24 Oct = 24 Oct 08 .. 25 Oct 08 UTC
    store.put(rain.INDEX, json.dumps(index).encode())
    store.put(rain.ckey(date(2026, 10, 19)), rain.cog(np.full(shape, 100.0)))
    store.put(rain.ckey(date(2026, 10, 24)), rain.cog(np.full(shape, 112.0)))
    hours = np.zeros((24, *shape), np.float32)
    hours[3] = 1.5  # 20 Oct 03 UTC: before that rain day starts, counted as the head
    store.put(rain.hkey(date(2026, 10, 20)), rain.cog(hours))
    tail = np.zeros((24, *shape), np.float32)
    tail[9] = 2.0  # 25 Oct 09 UTC: after the last whole rain day
    tail[7] = 50.0  # 25 Oct 07 UTC: inside rain day 24 Oct, already in the total
    store.put(rain.hkey(date(2026, 10, 25)), rain.cog(tail))
    mm = alerts.RainSums(store).since(date(2026, 10, 20), (10, 10))
    assert mm == 12.0 + 1.5 + 2.0


def test_messages_are_kept_in_the_history(tmp_path):
    store, sent, send = setup(tmp_path)
    cold = lambda x, y, now: ((-1.4, NOW + timedelta(hours=18)), (8.0, NOW), 25.0)  # noqa: E731
    alerts.run(store, now=NOW, rain_field=field(None), forecast=cold, send=send, log=lambda *_: None)
    h = json.loads(store.get(alerts.HISTORY))["alerts"]
    assert sorted(a["kind"] for a in h) == ["frost", "heavy"]
    assert all(a["id"] == "p1" and a["at"].startswith("2026-10-26") for a in h)
    # nothing new: the history stays as it is; very old entries drop out when new ones come
    alerts.run(
        store, now=NOW + timedelta(hours=2), rain_field=field(None), forecast=cold, send=send, log=lambda *_: None
    )
    assert len(json.loads(store.get(alerts.HISTORY))["alerts"]) == 2
    old = {"id": "p1", "kind": "rain", "at": "2020-01-01T00:00:00+00:00", "title": "x", "text": "y"}
    store.put(alerts.HISTORY, json.dumps({"alerts": [old, *h]}).encode())
    alerts.run(
        store, now=NOW + timedelta(minutes=5), rain_field=field(0), forecast=cold, send=send, log=lambda *_: None
    )
    kinds = [a["kind"] for a in json.loads(store.get(alerts.HISTORY))["alerts"]]
    assert "rain" in kinds and len(kinds) == 3  # the 2020 entry is gone, the new rain warning is in
