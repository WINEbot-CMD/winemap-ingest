"""ntfy alerts per vineyard / parcel (Jasper, 26 Oct 2026). The places (name + one RD point) come from the site
(`alerts/places-live.json`, written by /api/places). Checked every 5 minutes, after the nowcast:

  rain   the nowcast expects >= 0.5 mm in the next 2 hours, starting within the hour
  heavy  HARMONIE expects >= 20 mm in the next 24 hours        (checked once an hour)
  frost  HARMONIE 2 m temperature below 0 °C in the next 48 h  (checked once an hour)

A message goes out only when a warning starts (and, for the nowcast, after 15 dry minutes it can start again), so a
long frosty spell or a rainy afternoon gives one message, not one every 5 minutes. State: `alerts/state.json`.

    python -m winemap_ingest.alerts
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from . import grid
from .health import notify
from .rain import read_cog
from .store import open_store

PLACES = "alerts/places-live.json"
STATE = "alerts/state.json"
RAIN_MM = 0.5  # in the next 2 hours
RAIN_SOON = timedelta(hours=1)
HEAVY_MM = 20.0  # in the next 24 hours
FROST_C = 0.0
HOURLY = timedelta(minutes=55)
WMS = "https://geoservices.knmi.nl/adagucserver?dataset=uwcw_ha43_nl_2km"
NL = ZoneInfo("Europe/Amsterdam")


def cell(x: float, y: float) -> tuple[int, int] | None:
    a, _, c, _, e, f = grid.TRANSFORM
    col, row = int((x - c) // a), int((y - f) // e)
    return (col, row) if 0 <= col < grid.WIDTH and 0 <= row < grid.HEIGHT else None


def rain_soon(rates: np.ndarray, steps: list[datetime], now: datetime) -> tuple[float, datetime | None]:
    """Total mm in the series (5-minute rates in mm/h) and when it starts (>= 0.1 mm/h), if within the hour."""
    rates = np.nan_to_num(rates)
    total = float((rates * 5 / 60).sum())
    start = next((steps[i] - timedelta(minutes=5) for i, r in enumerate(rates) if r >= 0.1), None)
    if start is not None and start - now > RAIN_SOON:
        start = None
    return total, start


def _series(layer: str, x: float, y: float, hours: int, now: datetime) -> dict[datetime, float]:
    t0 = now.replace(minute=0, second=0, microsecond=0)
    t1 = t0 + timedelta(hours=hours)
    q = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": layer,
        "QUERY_LAYERS": layer,
        "CRS": "EPSG:28992",
        "BBOX": f"{x - 500},{y - 500},{x + 500},{y + 500}",
        "WIDTH": "3",
        "HEIGHT": "3",
        "I": "1",
        "J": "1",
        "INFO_FORMAT": "application/json",
        "TIME": f"{t0:%Y-%m-%dT%H:%M:%SZ}/{t1:%Y-%m-%dT%H:%M:%SZ}",
    }
    req = urllib.request.Request(f"{WMS}&{urllib.parse.urlencode(q)}", headers={"User-Agent": "winemap-alerts"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    d = (data[0] if isinstance(data, list) and data else data).get("data", {})
    while isinstance(d, dict) and d and not any(str(k)[:2] == "20" and "T" in str(k) for k in d):
        d = next(iter(d.values()))
    out = {}
    for k, v in (d or {}).items():
        try:
            out[datetime.fromisoformat(k.replace("Z", "+00:00"))] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def harmonie(x: float, y: float, now: datetime) -> tuple[tuple[float, datetime] | None, float | None]:
    """(lowest 2 m temperature and when, in 48 h), total precipitation in 24 h — from the public KNMI map service."""
    temp = _series("air_temperature_hagl", x, y, 48, now)
    rain = _series("total_precipitation_accumulation_01h_hagl", x, y, 24, now)
    low = min(((v, t) for t, v in temp.items()), default=None)
    return low, (sum(rain.values()) if rain else None)


def _when(t: datetime) -> str:
    days = ["ma", "di", "wo", "do", "vr", "za", "zo"]
    lt = t.astimezone(NL)
    return f"{days[lt.weekday()]} {lt:%H:%M}"


def run(store, now: datetime | None = None, rain_field=None, forecast=harmonie, send=notify, log=print) -> dict:
    now = now or datetime.now(UTC)
    places = (json.loads(store.get(PLACES) or b"{}")).get("places", [])
    state = json.loads(store.get(STATE) or b"{}")
    active: dict = state.setdefault("active", {})
    topic = os.environ.get("NTFY_TOPIC", "")
    if not places:
        log("no places yet")
        return state

    # nowcast: rain in the next 2 hours
    if rain_field is None:
        idx = json.loads(store.get("rain/nowcast/index.json") or b"{}")
        raw = store.get("rain/" + idx["file"]) if idx.get("file") else None
        rain_field = (
            (read_cog(raw), [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in idx["steps"]]) if raw else None
        )
    for p in places:
        c = cell(p["x"], p["y"])
        if rain_field is None or c is None:
            continue
        field, steps = rain_field
        total, start = rain_soon(field[:, c[1], c[0]], steps, now)
        key = f"{p['id']}:rain"
        if total >= RAIN_MM and start is not None:
            if key not in active:
                mins = max(0, round((start - now).total_seconds() / 60))
                when = "nu" if mins <= 5 else f"over {mins} min"
                send(
                    topic,
                    f"Regen bij {p['name']}",
                    f"Regen {when}: {total:.1f} mm in de komende 2 uur.",
                    "default",
                    "umbrella",
                )
            active[key] = {"since": active.get(key, {}).get("since", now.isoformat()), "dry": 0}
        elif key in active:
            active[key]["dry"] = active[key].get("dry", 0) + 1
            if active[key]["dry"] >= 3:  # 15 minutes dry: it may warn again
                del active[key]

    # HARMONIE once an hour: frost and heavy rain
    last = state.get("hourly")
    if not last or now - datetime.fromisoformat(last) >= HOURLY:
        for p in places:
            try:
                low, rain24 = forecast(p["x"], p["y"], now)
            except Exception as e:  # noqa: BLE001
                log(f"{p['name']}: forecast failed ({e})")
                continue
            frost = low is not None and low[0] < FROST_C
            heavy = rain24 is not None and rain24 >= HEAVY_MM
            checks = {
                "frost": (frost, f"{low[0]:.1f} °C verwacht ({_when(low[1])})." if frost else ""),
                "heavy": (heavy, f"{rain24:.0f} mm verwacht in de komende 24 uur." if heavy else ""),
            }
            for kind, (on, text) in checks.items():
                key = f"{p['id']}:{kind}"
                if on and key not in active:
                    title = f"Vorst bij {p['name']}" if kind == "frost" else f"Veel regen bij {p['name']}"
                    send(
                        topic,
                        title,
                        text,
                        "high" if kind == "frost" else "default",
                        "snowflake" if kind == "frost" else "cloud_with_rain",
                    )
                    active[key] = {"since": now.isoformat()}
                elif not on:
                    active.pop(key, None)
        state["hourly"] = now.isoformat()

    state["checked"] = now.isoformat()
    store.put(STATE, json.dumps(state, indent=1).encode(), "application/json")
    log(f"{len(places)} places; active: {sorted(active)}")
    return state


def main() -> None:
    run(open_store())


if __name__ == "__main__":
    main()
