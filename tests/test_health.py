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


def test_a_backup_older_than_three_days_or_missing_is_a_problem():
    st = {"sites": {}, "data": {}, "jobs_failing": {}, "plan": {}}
    fresh = {**st, "backup": {"ok": True, "last": "2026-10-24T12:00:00Z"}}
    assert health.problems(fresh, NOW) == {}
    old = {**st, "backup": {"ok": True, "last": "2026-10-20T12:00:00Z"}}
    assert "5 days old" in health.problems(old, NOW)["backup"]
    assert "missing" in health.problems({**st, "backup": {"ok": True, "last": None}}, NOW)["backup"]
    assert "failed" in health.problems({**st, "backup": {"ok": False, "detail": "x"}}, NOW)["backup"]
    assert health.problems(st, NOW) == {}  # not configured: no alarm


def test_check_backup_finds_the_newest_daily():
    from datetime import UTC, datetime

    class Pages:
        def paginate(self, **kw):
            assert kw == {"Bucket": "winemap-backup", "Prefix": "daily/"}
            yield {
                "Contents": [
                    {
                        "Key": "daily/winemap-2026-10-26.7z",
                        "LastModified": datetime(2026, 10, 26, 13, tzinfo=UTC),
                        "Size": 2.3e9,
                    },
                    {
                        "Key": "daily/winemap-2026-10-27.7z",
                        "LastModified": datetime(2026, 10, 27, 13, tzinfo=UTC),
                        "Size": 2.4e9,
                    },
                ]
            }

    class Client:
        def get_paginator(self, name):
            return Pages()

    got = health.check_backup("acc", "id", "secret", client=Client())
    assert got["ok"] and got["last"].startswith("2026-10-27T13") and got["gb"] == 2.4
