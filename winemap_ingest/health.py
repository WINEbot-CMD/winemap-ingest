"""Health check for Winemap (every 5 minutes, after the nowcast): is the site up (live + staging, through Cloudflare
Access with a service token), is the data fresh, do the jobs run? The result goes to `monitor/state.json` in R2 (the
/status page shows it) and a message goes to ntfy only when something changes: a problem starts or is solved.

    python -m winemap_ingest.health
env: CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET (Access service token), NTFY_TOPIC, KNMI_API_KEY, R2_*
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import UTC, datetime, timedelta

from .knmi import API
from .store import open_store

# the home page: every version of the site has it (the /api routes only exist after the monitoring release)
SITES = {
    "live": "https://winemap-nl.wopferu.workers.dev/",
    "staging": "https://staging-winemap-nl.wopferu.workers.dev/",
}
STATE = "monitor/state.json"
STATUS = "https://staging-winemap-nl.wopferu.workers.dev/api/status"  # has the free-plan meter (also live, later)
QUOTA_ALERT = 0.8
INGEST_RUNS = "https://api.github.com/repos/WINEbot-CMD/winemap-ingest/actions/runs?per_page=30"
NOWCAST_LATE = timedelta(minutes=45)
RAIN_LATE = timedelta(hours=12)  # the hourly job catches up slowly while the KNMI limit is shared


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse(s: str | None) -> datetime | None:
    if not s:
        return None
    if len(s) == 13:  # YYYY-MM-DDTHH
        s += ":00:00Z"
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def check_site(url: str, headers: dict[str, str]) -> dict:
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "winemap-health", **headers})
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read()
            ok = r.status == 200 and b"Winemap" in body and b"cloudflareaccess" not in body
            detail = "ok" if ok else f"HTTP {r.status}, not the Winemap page (Access login page?)"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, str(e)[:160]
    return {"ok": ok, "ms": round((time.monotonic() - t0) * 1000), "detail": detail}


def check_nowcast_path(site: str, headers: dict[str, str]) -> dict:
    """What Live (2 h) does in the browser: the nowcast index through the site's /wx route, then a range read of
    the nowcast COG it points at. ok = both work; age = minutes since the run."""
    base = site.rstrip("/")
    h = {"User-Agent": "winemap-health", **headers}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{base}/wx/rain/nowcast/index.json", headers=h), timeout=20
        ) as r:
            idx = json.loads(r.read())
        req = urllib.request.Request(f"{base}/wx/rain/{idx['file']}", headers={**h, "Range": "bytes=0-1023"})
        with urllib.request.urlopen(req, timeout=20) as r:
            part = r.read()
            ok = r.status == 206 and part[:2] in (b"II", b"MM")  # a TIFF, cut by the range
            detail = "ok" if ok else f"COG answered {r.status}, {len(part)} bytes"
        run = _parse(idx.get("run"))
        age = round((datetime.now(UTC) - run).total_seconds() / 60) if run else None
        return {"ok": ok, "detail": detail, "age_min": age}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": str(e)[:160], "age_min": None}


def knmi_quota(key: str) -> dict | None:
    if not key:
        return None
    try:
        req = urllib.request.Request(
            f"{API}/radar_forecast/versions/2.0/files?maxKeys=1", headers={"Authorization": key}
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return {
                "remaining": int(r.headers.get("X-Ratelimit-Remaining", -1)),
                "limit": int(r.headers.get("X-Ratelimit-Limit", -1)),
            }
    except urllib.error.HTTPError as e:
        return {"remaining": 0, "limit": int(e.headers.get("X-Ratelimit-Limit", 1000) or 1000), "error": e.code}
    except Exception:  # noqa: BLE001
        return None


def failing_jobs() -> dict[str, int]:
    """Workflows whose last runs failed, and how many in a row."""
    try:
        req = urllib.request.Request(
            INGEST_RUNS, headers={"User-Agent": "winemap-health", "Accept": "application/vnd.github+json"}
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            runs = json.loads(r.read())["workflow_runs"]
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, int] = {}
    done: set[str] = set()
    for w in runs:  # newest first
        name = w["name"]
        if name in done or w["status"] != "completed":
            continue
        if w["conclusion"] == "failure":
            out[name] = out.get(name, 0) + 1
        else:
            done.add(name)
    return out


def plan_use(headers: dict[str, str]) -> dict:
    """The free-plan meter from /api/status (Worker requests, R2, Analytics Engine): name -> share used (0..1)."""
    if not headers:
        return {}
    try:
        req = urllib.request.Request(STATUS, headers={"User-Agent": "winemap-health", **headers})
        with urllib.request.urlopen(req, timeout=30) as r:
            q = json.loads(r.read()).get("quota") or {}
    except Exception:  # noqa: BLE001
        return {}
    return {k: v["used"] / v["limit"] for k, v in q.items() if isinstance(v, dict) and v.get("limit")}


def problems(state: dict, now: datetime) -> dict[str, str]:
    """The problems right now: key -> message."""
    out = {}
    for name, s in state["sites"].items():
        if not s["ok"]:
            out[f"site:{name}"] = f"Winemap {name} is down: {s['detail']}"
        elif s.get("nowcast") and not s["nowcast"]["ok"]:
            out[f"wx:{name}"] = f"Live 2 h on {name} cannot read the nowcast: {s['nowcast']['detail']}"
    nc = _parse(state["data"].get("nowcast"))
    if nc and now - nc > NOWCAST_LATE:
        out["nowcast"] = f"Nowcast is {round((now - nc).total_seconds() / 60)} min old"
    rain = _parse(state["data"].get("rain_hour"))
    if rain and now - rain > RAIN_LATE:
        out["rain"] = f"Rain sums are {round((now - rain).total_seconds() / 3600)} h behind"
    for name, share in state.get("plan", {}).items():
        if share >= QUOTA_ALERT:
            out[f"plan:{name}"] = f"Free plan: {name} at {round(share * 100)} %"
    for name, n in state["jobs_failing"].items():
        if n >= 3:
            out[f"job:{name}"] = f"Job '{name}' failed {n} times in a row"
    return out


def notify(topic: str, title: str, message: str, priority: str = "default", tags: str = "warning") -> None:
    if not topic:
        return
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}",
        data=message.encode(),
        headers={"Title": title, "Priority": priority, "Tags": tags},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=20).close()
    except Exception as e:  # noqa: BLE001
        print(f"ntfy failed: {e}")


def run(
    store, now: datetime | None = None, sites=None, quota=None, jobs=None, plan=None, send=notify, log=print
) -> dict:
    now = now or datetime.now(UTC)
    old = json.loads(store.get(STATE) or b"{}")
    headers = {}
    if os.environ.get("CF_ACCESS_CLIENT_ID"):
        headers = {
            "CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        }
    # without the Access service token every check would only see the login page: then no site check (no false alarms)
    checked = sites if sites is not None else ({n: check_site(u, headers) for n, u in SITES.items()} if headers else {})
    if headers and sites is None:  # the 2-hour Live path, per site
        for name, url in SITES.items():
            if name in checked:
                checked[name]["nowcast"] = check_nowcast_path(url, headers)
    for name, s in checked.items():  # since when up / down
        prev = old.get("sites", {}).get(name)
        s["since"] = prev["since"] if prev and prev.get("ok") == s["ok"] else _iso(now)
    rain = json.loads(store.get("rain/index.json") or b"{}")
    nc = json.loads(store.get("rain/nowcast/index.json") or b"{}")
    state = {
        "checked": _iso(now),
        "sites": checked,
        "knmi": quota if quota is not None else knmi_quota(os.environ.get("KNMI_API_KEY", "")),
        "data": {"nowcast": nc.get("run"), "rain_hour": (rain.get("hour") or {}).get("last")},
        "jobs_failing": jobs if jobs is not None else failing_jobs(),
        "plan": plan if plan is not None else plan_use(headers),
    }
    now_p = problems(state, now)
    was = old.get("problems", {})
    topic = os.environ.get("NTFY_TOPIC", "")
    for key, msg in now_p.items():
        if key not in was:
            send(topic, "Winemap: problem", msg, "high" if key.startswith("site:") else "default", "warning")
    for key in was:
        if key not in now_p:
            send(topic, "Winemap: solved", f"Solved: {was[key]['message']}", "low", "white_check_mark")
    state["problems"] = {k: {"message": m, "since": was.get(k, {}).get("since", _iso(now))} for k, m in now_p.items()}
    store.put(STATE, json.dumps(state, indent=1).encode(), "application/json")
    up_down = ", ".join(f"{k}={'up' if v['ok'] else 'DOWN'}" for k, v in checked.items())
    log(f"sites {up_down}; problems: {list(now_p) or 'none'}")
    return state


def main() -> None:
    run(open_store())


if __name__ == "__main__":
    main()
