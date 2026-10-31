import json
from datetime import UTC, datetime, timedelta

from winemap_ingest import health
from winemap_ingest.store import LocalStore

NOW = datetime(2026, 10, 25, 16, 0, tzinfo=UTC)


def up(ok=True):
    return {
        "live": {"ok": ok, "ms": 120, "detail": "ok" if ok else "timeout"},
        "staging": {"ok": True, "ms": 90, "detail": "ok"},
    }


def test_alerts_only_when_a_problem_starts_or_ends(tmp_path):
    store = LocalStore(tmp_path)
    store.put("rain/nowcast/index.json", json.dumps({"run": "2026-10-25T15:55:00Z"}).encode())
    store.put("rain/index.json", json.dumps({"hour": {"last": "2026-10-25T14"}}).encode())
    sent = []

    def send(topic, title, msg, prio="default", tags=""):
        sent.append((title, msg))

    def run(t, sites):
        return health.run(
            store,
            now=t,
            sites=sites,
            quota={"remaining": 500, "limit": 1000},
            jobs={},
            plan={},
            send=send,
            log=lambda *_: None,
        )

    run(NOW, up())
    assert sent == []  # all fine
    s = run(NOW + timedelta(minutes=5), up(False))
    assert sent == [("Winemap: problem", "Winemap live is down: timeout")]
    assert s["sites"]["live"]["since"] == "2026-10-25T16:05:00Z"
    run(NOW + timedelta(minutes=10), up(False))
    assert len(sent) == 1  # still down: no new message
    run(NOW + timedelta(minutes=15), up())
    assert sent[-1] == ("Winemap: solved", "Solved: Winemap live is down: timeout")
    # the nowcast getting old is a problem too (jobs failing 3x as well)
    state = run(NOW + timedelta(hours=1), up())
    assert "nowcast" in state["problems"]
    assert health.problems({"sites": {}, "data": {}, "jobs_failing": {"Rain sums (hourly)": 3}}, NOW) == {
        "job:Rain sums (hourly)": "Job 'Rain sums (hourly)' failed 3 times in a row"
    }


def test_free_plan_alert_from_80_percent():
    st = {"sites": {}, "data": {}, "jobs_failing": {}, "plan": {"r2ClassA": 0.81, "workers": 0.2}}
    assert health.problems(st, NOW) == {"plan:r2ClassA": "Free plan: r2ClassA at 81 %"}


def test_supabase_down_is_a_problem_until_it_answers_again(tmp_path):
    store = LocalStore(tmp_path)
    store.put("rain/nowcast/index.json", json.dumps({"run": "2026-10-25T15:55:00Z"}).encode())
    store.put("rain/index.json", json.dumps({"hour": {"last": "2026-10-25T14"}}).encode())
    sent = []

    def run(t, ok):
        sb = {"ok": ok, "ms": 80, "detail": "ok, database 12 MB" if ok else "HTTP Error 503"}
        return health.run(
            store,
            now=t,
            sites=up(),
            quota={"remaining": 500, "limit": 1000},
            jobs={},
            plan={},
            supabase=sb,
            send=lambda topic, title, msg, *a: sent.append((title, msg)),
            log=lambda *_: None,
        )

    run(NOW, True)
    s = run(NOW + timedelta(minutes=5), False)
    assert "supabase" in s["problems"] and s["supabase"]["since"] == "2026-10-25T16:05:00Z"
    assert sent[-1][0] == "Winemap: problem" and "Supabase" in sent[-1][1]
    run(NOW + timedelta(minutes=10), True)
    assert sent[-1][0] == "Winemap: solved"


def test_each_backup_is_watched_on_its_own():
    st = {"sites": {}, "data": {}, "jobs_failing": {}, "plan": {}}
    fresh = {"ok": True, "laptop": {"last": "2026-10-24T12:00:00Z"}, "cloud": {"last": "2026-10-25T02:40:00Z"}}
    assert health.problems({**st, "backup": fresh}, NOW) == {}
    laptop_off = {**fresh, "laptop": {"last": "2026-10-20T12:00:00Z"}}
    got = health.problems({**st, "backup": laptop_off}, NOW)
    assert list(got) == ["backup:laptop"] and "5 days old" in got["backup:laptop"]
    no_cloud = {**fresh, "cloud": {"last": None}}
    assert "missing" in health.problems({**st, "backup": no_cloud}, NOW)["backup:cloud"]
    assert "failed" in health.problems({**st, "backup": {"ok": False, "detail": "x"}}, NOW)["backup"]
    assert health.problems(st, NOW) == {}  # not configured: no alarm


def test_check_backup_finds_the_newest_of_each():
    from datetime import UTC, datetime

    def obj(key, day, gb):
        return {"Key": key, "LastModified": datetime(2026, 10, day, 13, tzinfo=UTC), "Size": gb * 1e9}

    class Pages:
        def paginate(self, **kw):
            assert kw == {"Bucket": "winemap-backup", "Prefix": "daily/"}
            yield {
                "Contents": [
                    obj("daily/winemap-2026-10-26.7z", 26, 2.3),
                    obj("daily/winemap-2026-10-27.7z", 27, 0.1),
                    obj("daily/winemap-cloud-2026-10-27.7z", 27, 1.5),
                ]
            }

    class Client:
        def get_paginator(self, name):
            return Pages()

    got = health.check_backup("acc", "id", "secret", client=Client())
    assert got["ok"] and got["laptop"]["last"].startswith("2026-10-27T13") and got["laptop"]["gb"] == 0.1
    assert got["cloud"]["gb"] == 1.5


def test_keepalive_staging_database_and_powersync_once_a_day(tmp_path):
    """winemap-nl docs/39: the free plans stop a project after 1-2 weeks without use."""
    store = LocalStore(tmp_path)
    store.put("rain/nowcast/index.json", json.dumps({"run": "2026-10-25T15:55:00Z"}).encode())
    store.put("rain/index.json", json.dumps({"hour": {"last": "2026-10-25T14"}}).encode())
    sent = []

    def run(t, **checks):
        return health.run(
            store,
            now=t,
            sites=up(),
            quota={"remaining": 500, "limit": 1000},
            jobs={},
            plan={},
            send=lambda topic, title, msg, *a: sent.append((title, msg)),
            log=lambda *_: None,
            **checks,
        )

    s = run(
        NOW, supabase_staging={"ok": True, "ms": 70, "detail": "ok"}, powersync={"ok": True, "ms": 900, "detail": "ok"}
    )
    assert s["powersync"]["last_ok"] == "2026-10-25T16:00:00Z" and s["problems"] == {}
    # not due again within the day: the last result stays
    s = run(NOW + timedelta(hours=1), supabase_staging={"ok": True, "ms": 70, "detail": "ok"})
    assert s["powersync"]["last_ok"] == "2026-10-25T16:00:00Z"
    # a paused staging database and a failing keep-alive are problems
    s = run(
        NOW + timedelta(days=1),
        supabase_staging={"ok": False, "ms": 70, "detail": "HTTP Error 540"},
        powersync={"ok": False, "ms": 900, "detail": "HTTP Error 401"},
    )
    assert {"supabase_staging", "powersync"} <= set(s["problems"])  # (the test data is a day old by now too)
    assert s["powersync"]["last_ok"] == "2026-10-25T16:00:00Z"  # the last time it worked stays known
    # even when the check itself has nothing new to say, 3 days without success is a problem
    st = {
        "sites": {},
        "data": {},
        "jobs_failing": {},
        "plan": {},
        "powersync": {"ok": True, "last_ok": "2026-10-21T16:00:00Z"},
    }
    assert list(health.problems(st, NOW)) == ["powersync"]
