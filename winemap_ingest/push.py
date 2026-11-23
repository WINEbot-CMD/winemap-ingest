"""The frost alarm as a push message on the grower's phone (winemap-nl docs/65 §7 step 4; Jasper 22 Nov 2026).

The grower turns the alarm on per vineyard in the app (Vineyard manager > Alarms). The database gives this job, with its
own key, every alarm that has a phone: `frost_push_targets(push_key)` -> the vineyard's places (point, frost limit,
budburst) and the phones (token, platform, language, quiet hours). Once an hour the job checks the same HARMONIE 48 h
forecast as the ntfy alerts (alerts.py): frost when the lowest 2 m temperature goes below a place's limit (0 °C), urgent
after budburst. One message per vineyard when the frost starts (the coldest place), in the phone's language, through
Firebase Cloud Messaging (HTTP v1; Firebase also reaches iPhones once Apple is set up).

Quiet hours (per phone, Europe/Amsterdam): frost that starts in them waits until they end, and goes out then only if the
frost is still expected. Phones Firebase no longer knows (app removed) are forgotten (`frost_push_gone`).

The state (`alerts/push-state.json`) holds no tokens and no names: the R2 bucket is served under /wx/.

Databases: live (SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, FROST_PUSH_KEY) and staging (SUPABASE_STAGING_URL,
SUPABASE_STAGING_PUBLISHABLE_KEY, FROST_PUSH_KEY_STAGING); a database without its key is skipped.
Firebase: FIREBASE_SERVICE_ACCOUNT (the service account JSON; one Firebase project for both apps).

    python -m winemap_ingest.push
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

STATE = "alerts/push-state.json"
HOURLY = timedelta(minutes=55)
FROST_C = 0.0
NL = ZoneInfo("Europe/Amsterdam")
DATABASES = (
    ("live", "SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "FROST_PUSH_KEY"),
    ("staging", "SUPABASE_STAGING_URL", "SUPABASE_STAGING_PUBLISHABLE_KEY", "FROST_PUSH_KEY_STAGING"),
)

TEXTS = {
    "en": ("Frost at {name}", "{v} °C expected ({when}).", " After budburst: young shoots can freeze."),
    "nl": ("Vorst bij {name}", "{v} °C verwacht ({when}).", " Na de knopuitloop: jonge scheuten kunnen bevriezen."),
    "de": ("Frost bei {name}", "{v} °C erwartet ({when}).", " Nach dem Austrieb: junge Triebe können erfrieren."),
    "fr": ("Gel à {name}", "{v} °C prévus ({when}).", " Après le débourrement : les jeunes pousses peuvent geler."),
    "es": ("Helada en {name}", "{v} °C previstos ({when}).", " Tras la brotación: los brotes jóvenes pueden helarse."),
}
DAYS = {
    "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "nl": ["ma", "di", "wo", "do", "vr", "za", "zo"],
    "de": ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"],
    "fr": ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."],
    "es": ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"],
}


def message(lang: str, name: str, low: float, at: datetime, budburst: bool) -> tuple[str, str]:
    """Title and text of a frost message in the phone's language (English when unknown)."""
    lang = lang if lang in TEXTS else "en"
    title, body, after = TEXTS[lang]
    lt = at.astimezone(NL)
    when = f"{DAYS[lang][lt.weekday()]} {lt:%H:%M}"
    v = f"{low:.1f}" if lang == "en" else f"{low:.1f}".replace(".", ",")
    return title.format(name=name), body.format(v=v, when=when) + (after if budburst else "")


def in_quiet(device: dict, now: datetime) -> bool:
    """Whether it is in the phone's quiet hours now (from 22 until 7 runs past midnight; from == until: none)."""
    f, t = device.get("quiet_from"), device.get("quiet_to")
    if f is None or t is None or f == t:
        return False
    h = now.astimezone(NL).hour
    return f <= h < t if f < t else (h >= f or h < t)


def _num(v, default: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if x == x and abs(x) != float("inf") else default


def _budburst(p: dict, now: datetime) -> bool:
    b = str(p.get("budburst") or "")
    return b[:4] == f"{now.year}" and b <= now.date().isoformat()


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:24]


# --- Firebase Cloud Messaging (HTTP v1) ---------------------------------------------------------------------------


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


class Fcm:
    """Sends through Firebase with a service account: a signed JWT for an access token (an hour), then messages:send."""

    def __init__(self, account: dict):
        self.account = account
        self.project = account["project_id"]
        self._token: tuple[str, float] | None = None

    def _access(self) -> str:
        if self._token and self._token[1] > time.time() + 60:
            return self._token[0]
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        now = int(time.time())
        uri = self.account.get("token_uri", "https://oauth2.googleapis.com/token")
        claims = {
            "iss": self.account["client_email"],
            "scope": "https://www.googleapis.com/auth/firebase.messaging",
            "aud": uri,
            "iat": now,
            "exp": now + 3600,
        }
        head = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode()) + "." + _b64(json.dumps(claims).encode())
        key = serialization.load_pem_private_key(self.account["private_key"].encode(), None)
        jwt = head + "." + _b64(key.sign(head.encode(), padding.PKCS1v15(), hashes.SHA256()))
        body = f"grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&assertion={jwt}".encode()
        req = urllib.request.Request(uri, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=20) as r:
            got = json.load(r)
        self._token = (got["access_token"], time.time() + int(got.get("expires_in", 3600)))
        return self._token[0]

    def send(self, token: str, title: str, text: str, data: dict, urgent: bool) -> str:
        """'ok', 'gone' (Firebase no longer knows this phone) or 'error'."""
        msg = {
            "message": {
                "token": token,
                "notification": {"title": title, "body": text},
                "data": {k: str(v) for k, v in data.items()},
                "android": {"priority": "HIGH", "notification": {"tag": data.get("alarm", "frost")}},
                "apns": {"headers": {"apns-priority": "10"}, "payload": {"aps": {"sound": "default"}}},
            }
        }
        if urgent:
            msg["message"]["android"]["notification"]["notification_priority"] = "PRIORITY_MAX"
        req = urllib.request.Request(
            f"https://fcm.googleapis.com/v1/projects/{self.project}/messages:send",
            data=json.dumps(msg).encode(),
            headers={"Authorization": f"Bearer {self._access()}", "Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=20).close()
            return "ok"
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")
            if e.code == 404 or "UNREGISTERED" in err or ("INVALID_ARGUMENT" in err and "token" in err.lower()):
                return "gone"
            print(f"fcm {e.code}: {err[:300]}")
            return "error"


# --- the databases --------------------------------------------------------------------------------------------------


class Database:
    """One accounts database (live or staging), reached with the publishable key; the push key is the lock."""

    def __init__(self, name: str, url: str, key: str, push_key: str):
        self.name, self.url, self.key, self.push_key = name, url.rstrip("/"), key, push_key

    def _rpc(self, fn: str, body: dict):
        req = urllib.request.Request(
            f"{self.url}/rest/v1/rpc/{fn}",
            data=json.dumps({"push_key": self.push_key, **body}).encode(),
            headers={"apikey": self.key, "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def targets(self) -> list[dict]:
        return self._rpc("frost_push_targets", {}) or []

    def gone(self, tokens: list[str]) -> int:
        return self._rpc("frost_push_gone", {"tokens": tokens}) if tokens else 0


def databases_from_env() -> list[Database]:
    out = []
    for name, url, key, push in DATABASES:
        if os.environ.get(url) and os.environ.get(key) and os.environ.get(push):
            out.append(Database(name, os.environ[url], os.environ[key], os.environ[push]))
    return out


# --- the check ------------------------------------------------------------------------------------------------------


def _coldest(target: dict, forecast, now: datetime, log) -> tuple[dict | None, bool]:
    """The place with the deepest frost below its limit (None: no frost), and whether every forecast came back."""
    worst, complete = None, True
    for p in target.get("places") or []:
        try:
            low, _, _ = forecast(p["x"], p["y"], now)
        except Exception as e:  # noqa: BLE001
            log(f"push: forecast failed ({e})")
            complete = False
            continue
        if low is None or low[0] >= _num(p.get("frostC"), FROST_C):
            continue
        if worst is None or low[0] < worst["low"]:
            worst = {
                "place": p.get("id"),
                "low": round(float(low[0]), 1),
                "at": low[1].isoformat(),
                "budburst": _budburst(p, now),
            }
    return worst, complete


def run(store, now: datetime | None = None, forecast=None, dbs=None, sender=None, log=print) -> dict:
    now = now or datetime.now(UTC)
    if forecast is None:
        from .alerts import harmonie as forecast
    dbs = databases_from_env() if dbs is None else dbs
    if not dbs:
        log("push: no database with a push key")
        return {}
    if sender is None:
        account = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if not account:
            log("push: no FIREBASE_SERVICE_ACCOUNT")
            return {}
        sender = Fcm(json.loads(account))
    state = json.loads(store.get(STATE) or b"{}")
    last = state.get("hourly")
    hourly = not last or now - datetime.fromisoformat(last) >= HOURLY

    for db in dbs:
        st = state.setdefault(db.name, {})
        active: dict = st.setdefault("active", {})
        pending: dict = st.setdefault("pending", {})
        try:
            targets = {t["alarm"]: t for t in db.targets()}
        except Exception as e:  # noqa: BLE001
            log(f"push {db.name}: no targets ({e})")
            continue
        gone: list[str] = []

        def tell(t: dict, a: dict, device: dict, gone: list[str] = gone) -> None:
            names = {p.get("id"): p.get("name") for p in t.get("places") or []}
            title, text = message(
                device.get("lang") or "en",
                names.get(a["place"]) or "",
                a["low"],
                datetime.fromisoformat(a["at"]),
                a["budburst"],
            )
            data = {"kind": "frost", "vineyard_id": t["vineyard_id"], "alarm": t["alarm"]}
            if sender.send(device["token"], title, text, data, a["budburst"]) == "gone":
                gone.append(device["token"])

        if hourly:
            for alarm, t in targets.items():
                worst, complete = _coldest(t, forecast, now, log)
                if worst is None:
                    if complete:  # a failed forecast keeps what we had
                        active.pop(alarm, None)
                    continue
                if alarm not in active:
                    for d in t.get("devices") or []:
                        if in_quiet(d, now):
                            pending[f"{_hash(d['token'])}:{alarm}"] = now.isoformat()
                        else:
                            tell(t, worst, d)
                    active[alarm] = {"since": now.isoformat(), **worst}
                else:
                    active[alarm] = {**worst, "since": active[alarm].get("since", now.isoformat())}
            for alarm in [a for a in active if a not in targets]:  # alarm turned off, phone gone, member left
                del active[alarm]

        # phones whose quiet hours ended: the frost that waited, if it is still expected
        for k in list(pending):
            h, alarm = k.split(":", 1)
            t, a = targets.get(alarm), active.get(alarm)
            device = next((d for d in (t or {}).get("devices") or [] if _hash(d["token"]) == h), None)
            if t is None or a is None or device is None:
                del pending[k]
            elif not in_quiet(device, now):
                tell(t, a, device)
                del pending[k]

        if gone:
            try:
                log(f"push {db.name}: {db.gone(sorted(set(gone)))} phone(s) forgotten")
            except Exception as e:  # noqa: BLE001
                log(f"push {db.name}: frost_push_gone failed ({e})")
        log(f"push {db.name}: {len(targets)} alarm(s); frost: {len(active)}; waiting: {len(pending)}")

    if hourly:
        state["hourly"] = now.isoformat()
    store.put(STATE, json.dumps(state, indent=1).encode(), "application/json")
    return state


def main() -> None:
    from .store import open_store

    run(open_store())


if __name__ == "__main__":
    main()
