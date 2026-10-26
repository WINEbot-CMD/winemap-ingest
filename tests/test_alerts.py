import json
from datetime import UTC, datetime, timedelta

import numpy as np

from winemap_ingest import alerts, grid
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
    dry = lambda x, y, now: (None, None)  # noqa: E731
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
    cold = lambda x, y, now: ((-1.4, NOW + timedelta(hours=18)), 25.0)  # noqa: E731
    alerts.run(store, now=NOW, rain_field=field(None), forecast=cold, send=send, log=lambda *_: None)
    titles = sorted(t for t, _ in sent)
    assert titles == ["Veel regen bij Blok B7", "Vorst bij Blok B7"]
    assert any("-1.4 °C" in m for _, m in sent)
    alerts.run(
        store, now=NOW + timedelta(minutes=30), rain_field=field(None), forecast=cold, send=send, log=lambda *_: None
    )
    assert len(sent) == 2  # the hourly check waits; and the warning is already active
