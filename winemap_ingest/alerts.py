"""ntfy alerts per vineyard / parcel (Jasper, 26 Oct 2026). The places (name + one RD point, the alert limits, the
budburst date and the recent sprays) come from the site (`alerts/places-live.json`, written by /api/places). Checked
every 5 minutes, after the nowcast:

  rain   the nowcast expects >= 0.5 mm in the next 2 hours, starting within the hour
  heavy  HARMONIE expects >= 20 mm in the next 24 hours                    (checked once an hour)
  frost  HARMONIE 2 m temperature below the place's limit (0 °C) in 48 h  (hourly; urgent after budburst)
  heat   HARMONIE 2 m temperature >= 30 °C, high >= 35 °C in 48 h (limits per place)    (hourly)
  spray  rain since a spray (radar sums) passes its wash-off limit (15 mm): "spray again?"  (hourly)

A message goes out only when a warning starts (and, for the nowcast, after 15 dry minutes it can start again), so a
long frosty spell or a rainy afternoon gives one message, not one every 5 minutes. State: `alerts/state.json`.

    python -m winemap_ingest.alerts
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from . import grid
from .health import notify
from .rain import DAY_START, INDEX, ckey, hkey, parse_hour, read_cog
from .store import open_store

PLACES = "alerts/places-live.json"
STATE = "alerts/state.json"
RAIN_MM = 0.5  # in the next 2 hours
RAIN_SOON = timedelta(hours=1)
HEAVY_MM = 20.0  # in the next 24 hours
FROST_C = 0.0  # defaults; a place can bring its own (passport of the vineyard / parcel)
HEAT_C = 30.0
HEAT_SEVERE_C = 35.0
WASHOFF_MM = 15.0
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


Extreme = tuple[float, datetime] | None


def harmonie(x: float, y: float, now: datetime) -> tuple[Extreme, Extreme, float | None]:
    """Lowest and highest 2 m temperature (and when) in 48 h, total precipitation in 24 h — from the public KNMI map
    service."""
    temp = _series("air_temperature_hagl", x, y, 48, now)
    rain = _series("total_precipitation_accumulation_01h_hagl", x, y, 24, now)
    low = min(((v, t) for t, v in temp.items()), default=None)
    high = max(((v, t) for t, v in temp.items()), default=None)
    return low, high, (sum(rain.values()) if rain else None)


class RainSums:
    """Rain (mm) at a cell since the start of a day (00 UTC), from our radar sums (rain.py layout), like the passport:
    whole rain days (08-08 UTC) from the running totals, the hours around them from the hour files."""

    def __init__(self, store):
        self.store = store
        self.index = json.loads(store.get(INDEX) or b"{}")
        self.files: dict[str, np.ndarray | None] = {}

    def _file(self, key: str) -> np.ndarray | None:
        if key not in self.files:
            raw = self.store.get(key)
            self.files[key] = read_cog(raw) if raw else None
        return self.files[key]

    def _cum(self, d: date, c: tuple[int, int]) -> float | None:
        f = self._file(ckey(d))
        return None if f is None else float(f[0, c[1], c[0]])

    def _hours(self, first: datetime, last: datetime, c: tuple[int, int]) -> float:
        total, t = 0.0, first
        while t <= last:
            f = self._file(hkey(t.date()))
            if f is not None:
                v = f[t.hour, c[1], c[0]]
                total += float(v) if np.isfinite(v) else 0.0
            t += timedelta(hours=1)
        return total

    def since(self, day: date, c: tuple[int, int]) -> float | None:
        if not self.index.get("hour") or not self.index.get("day"):
            return None
        last_hour = parse_hour(self.index["hour"]["last"])
        last_day = date.fromisoformat(self.index["day"]["last"])
        start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        if day > last_day:
            return self._hours(start, last_hour, c)
        end, before = self._cum(last_day, c), self._cum(day - timedelta(days=1), c)
        if end is None or before is None or not np.isfinite(end) or not np.isfinite(before):
            return None
        head = self._hours(start, start + timedelta(hours=DAY_START - 1), c)  # 00..07 UTC of that day
        tail_from = datetime(last_day.year, last_day.month, last_day.day, DAY_START, tzinfo=UTC) + timedelta(days=1)
        return max(0.0, end - before) + head + self._hours(tail_from, last_hour, c)


def _when(t: datetime) -> str:
    days = ["ma", "di", "wo", "do", "vr", "za", "zo"]
    lt = t.astimezone(NL)
    return f"{days[lt.weekday()]} {lt:%H:%M}"


def _day(d: date) -> str:
    months = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]
    return f"{d.day} {months[d.month - 1]}"


def _num(v, default: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def _budburst_this_season(p: dict, now: datetime) -> bool:
    """Budburst logged this year (on or before today)."""
    b = str(p.get("budburst") or "")
    return b[:4] == f"{now.year}" and b <= now.date().isoformat()


def run(
    store, now: datetime | None = None, rain_field=None, forecast=harmonie, sums=None, send=notify, log=print
) -> dict:
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

    # once an hour: HARMONIE (frost, heat, heavy rain) and the rain since each recent spray
    last = state.get("hourly")
    if not last or now - datetime.fromisoformat(last) >= HOURLY:
        for p in places:
            try:
                low, high, rain24 = forecast(p["x"], p["y"], now)
            except Exception as e:  # noqa: BLE001
                log(f"{p['name']}: forecast failed ({e})")
            else:
                _forecast_alerts(p, low, high, rain24, now, active, topic, send)
        _spray_alerts(places, sums if sums is not None else RainSums(store), now, active, topic, send)
        state["hourly"] = now.isoformat()

    state["checked"] = now.isoformat()
    store.put(STATE, json.dumps(state, indent=1).encode(), "application/json")
    log(f"{len(places)} places; active: {sorted(active)}")
    return state


def _forecast_alerts(p: dict, low: Extreme, high: Extreme, rain24, now: datetime, active: dict, topic, send) -> None:
    frost_c = _num(p.get("frostC"), FROST_C)
    heat_c = _num(p.get("heatC"), HEAT_C)
    severe_c = _num(p.get("heatSevereC"), HEAT_SEVERE_C)
    budburst = _budburst_this_season(p, now)
    # a level per kind (0 = off); a message when it starts or gets worse (heat 1 -> 2)
    levels = {
        "frost": 1 if low is not None and low[0] < frost_c else 0,
        "heat": (2 if high[0] >= severe_c else 1 if high[0] >= heat_c else 0) if high is not None else 0,
        "heavy": 1 if rain24 is not None and rain24 >= HEAVY_MM else 0,
    }
    for kind, level in levels.items():
        key = f"{p['id']}:{kind}"
        before = active.get(key, {}).get("level", 1) if key in active else 0
        if level == 0:
            active.pop(key, None)
            continue
        if level > before:
            if kind == "frost":
                title = f"Vorst bij {p['name']}"
                text = f"{low[0]:.1f} °C verwacht ({_when(low[1])})."
                if budburst:
                    text += " Na de knopuitloop: jonge scheuten kunnen bevriezen."
                prio, tags = ("urgent" if budburst else "high"), "snowflake"
            elif kind == "heat":
                title = f"{'Extreme hitte' if level == 2 else 'Hitte'} bij {p['name']}"
                text = f"{high[0]:.1f} °C verwacht ({_when(high[1])})."
                prio, tags = ("high" if level == 2 else "default"), "hot_face"
            else:
                title = f"Veel regen bij {p['name']}"
                text = f"{rain24:.0f} mm verwacht in de komende 24 uur."
                prio, tags = "default", "cloud_with_rain"
            send(topic, title, text, prio, tags)
        active[key] = {"since": active.get(key, {}).get("since", now.isoformat()), "level": max(level, before)}


def _spray_alerts(places: list, sums, now: datetime, active: dict, topic, send) -> None:
    """Jasper: "spray again?" once per spray, when the rain since it passes its wash-off limit."""
    current = set()
    for p in places:
        c = cell(p["x"], p["y"])
        for sp in (p.get("sprays") or [])[:20]:
            try:
                day = date.fromisoformat(str(sp.get("date")))
            except ValueError:
                continue
            key = f"{p['id']}:spray:{day.isoformat()}"
            current.add(key)
            if key in active or c is None or day > now.date():
                continue
            limit = _num(sp.get("washoff"), WASHOFF_MM)
            mm = sums.since(day, c)
            if mm is None or mm < limit:
                continue
            send(
                topic,
                f"Opnieuw spuiten? {p['name']}",
                f"{mm:.0f} mm regen sinds je bespuiting op {_day(day)} (afspoelgrens {limit:g} mm).",
                "default",
                "droplet",
            )
            active[key] = {"since": now.isoformat(), "mm": round(mm, 1)}
    for key in [k for k in active if ":spray:" in k and k not in current]:  # the spray left the list (old, deleted)
        del active[key]


def main() -> None:
    run(open_store())


if __name__ == "__main__":
    main()
