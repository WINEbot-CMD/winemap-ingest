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
