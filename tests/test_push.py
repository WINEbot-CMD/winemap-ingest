import json
from datetime import UTC, datetime, timedelta

from winemap_ingest import push
from winemap_ingest.store import LocalStore

NOW = datetime(2026, 4, 20, 12, 0, tzinfo=UTC)  # 14:00 in Amsterdam
P1 = {"id": "p1", "name": "Blok B7", "type": "parcel", "x": 154500, "y": 436000, "frostC": -1, "budburst": None}
P2 = {"id": "p2", "name": "Dal", "type": "parcel", "x": 154600, "y": 436100, "frostC": 0, "budburst": "2026-04-10"}
PHONE = {"token": "tok-a" * 5, "platform": "android", "lang": "nl", "quiet_from": None, "quiet_to": None}


class Db:
    def __init__(self, targets, name="staging"):
        self.name, self._targets, self.forgotten = name, targets, []

    def targets(self):
        return self._targets

    def gone(self, tokens):
        self.forgotten += tokens
        return len(tokens)


class Sender:
    def __init__(self, answer="ok"):
        self.sent, self.answer = [], answer

    def send(self, token, title, text, data, urgent):
        self.sent.append({"token": token, "title": title, "text": text, "data": data, "urgent": urgent})
        return self.answer


def target(devices, places=(P1, P2)):
    return {"alarm": "a" * 32, "vineyard_id": "v1", "places": list(places), "devices": devices}


def temps(by_x):
    """A forecast: the lowest temperature per place (by its x), 18 hours ahead."""
    return lambda x, y, now: ((by_x[x], now + timedelta(hours=18)), None, None)


def go(store, t, db, sender, forecast):
    return push.run(store, now=t, forecast=forecast, dbs=[db], sender=sender, log=lambda *_: None)


def test_one_message_per_vineyard_for_the_coldest_place_in_the_phones_language(tmp_path):
    store, sender = LocalStore(tmp_path), Sender()
    db = Db([target([PHONE, {**PHONE, "token": "tok-b" * 5, "lang": "en"}])])
    go(store, NOW, db, sender, temps({154500: -1.5, 154600: -0.5}))  # both below their limit; P1 is colder
    assert [s["title"] for s in sender.sent] == ["Vorst bij Blok B7", "Frost at Blok B7"]
    assert sender.sent[0]["text"] == "-1,5 °C verwacht (di 08:00)."
    assert sender.sent[1]["text"] == "-1.5 °C expected (Tue 08:00)."
    assert not sender.sent[0]["urgent"]  # P1 has no budburst
    assert sender.sent[0]["data"] == {"kind": "frost", "vineyard_id": "v1", "alarm": "a" * 32}
    go(store, NOW + timedelta(hours=1), db, sender, temps({154500: -2.0, 154600: -0.5}))
    assert len(sender.sent) == 2  # still frost: no new message
    state = json.loads(store.get(push.STATE))
    assert "tok-a" not in json.dumps(state) and "Blok" not in json.dumps(state)  # no tokens, no names in R2


def test_limits_come_from_the_place_and_budburst_makes_it_urgent(tmp_path):
    store, sender = LocalStore(tmp_path), Sender()
    db = Db([target([PHONE])])
    go(store, NOW, db, sender, temps({154500: -0.8, 154600: 5.0}))  # P1 limit -1: no frost
    assert sender.sent == []
    go(store, NOW + timedelta(hours=1), db, sender, temps({154500: -0.8, 154600: -0.3}))  # P2 limit 0, after budburst
    [s] = sender.sent
    assert s["title"] == "Vorst bij Dal" and s["urgent"] and "jonge scheuten" in s["text"]


def test_frost_again_after_it_ends_and_only_once_an_hour(tmp_path):
    store, sender = LocalStore(tmp_path), Sender()
    db = Db([target([PHONE])])
    cold, warm = temps({154500: -3.0, 154600: 2.0}), temps({154500: 3.0, 154600: 3.0})
    go(store, NOW, db, sender, cold)
    go(store, NOW + timedelta(minutes=30), db, sender, warm)  # the hourly check waits
    go(store, NOW + timedelta(hours=1), db, sender, warm)  # frost no longer expected
    go(store, NOW + timedelta(hours=2), db, sender, cold)
    assert len(sender.sent) == 2


def test_a_failed_forecast_keeps_the_frost_on(tmp_path):
    store, sender = LocalStore(tmp_path), Sender()
    db = Db([target([PHONE], places=[P1])])
    go(store, NOW, db, sender, temps({154500: -3.0}))

    def broken(x, y, now):
        raise OSError("KNMI down")

    go(store, NOW + timedelta(hours=1), db, sender, broken)
    go(store, NOW + timedelta(hours=2), db, sender, temps({154500: -3.0}))
    assert len(sender.sent) == 1


def test_quiet_hours_hold_the_message_until_they_end(tmp_path):
    store, sender = LocalStore(tmp_path), Sender()
    night = datetime(2026, 4, 20, 21, 0, tzinfo=UTC)  # 23:00 in Amsterdam
    db = Db([target([{**PHONE, "quiet_from": 22, "quiet_to": 7}])])
    cold = temps({154500: -2.0, 154600: 1.0})
    go(store, night, db, sender, cold)
    assert sender.sent == []
    go(store, night + timedelta(hours=7), db, sender, cold)  # 06:00: still quiet
    assert sender.sent == []
    go(store, night + timedelta(hours=8, minutes=5), db, sender, cold)  # 07:05
    assert len(sender.sent) == 1
    go(store, night + timedelta(hours=9, minutes=5), db, sender, cold)
    assert len(sender.sent) == 1  # once


def test_quiet_hours_drop_the_message_when_the_frost_is_off_by_then(tmp_path):
    store, sender = LocalStore(tmp_path), Sender()
    night = datetime(2026, 4, 20, 21, 0, tzinfo=UTC)
    db = Db([target([{**PHONE, "quiet_from": 22, "quiet_to": 7}])])
    go(store, night, db, sender, temps({154500: -2.0, 154600: 1.0}))
    go(store, night + timedelta(hours=3), db, sender, temps({154500: 4.0, 154600: 4.0}))
    go(store, night + timedelta(hours=8, minutes=5), db, sender, temps({154500: 4.0, 154600: 4.0}))
    assert sender.sent == []


def test_phones_firebase_no_longer_knows_are_forgotten(tmp_path):
    store, sender = LocalStore(tmp_path), Sender(answer="gone")
    db = Db([target([PHONE])])
    go(store, NOW, db, sender, temps({154500: -2.0, 154600: 1.0}))
    assert db.forgotten == [PHONE["token"]]


def test_an_alarm_turned_off_is_dropped(tmp_path):
    store, sender = LocalStore(tmp_path), Sender()
    db = Db([target([PHONE])])
    cold = temps({154500: -2.0, 154600: 1.0})
    go(store, NOW, db, sender, cold)
    db._targets = []
    go(store, NOW + timedelta(hours=1), db, sender, cold)
    db._targets = [target([PHONE])]
    go(store, NOW + timedelta(hours=2), db, sender, cold)
    assert len(sender.sent) == 2  # turned on again during frost: a new message


def test_quiet_hours():
    at = lambda h: datetime(2026, 1, 15, h - 1, 0, tzinfo=UTC)  # noqa: E731  (winter: UTC+1)
    night = {"quiet_from": 22, "quiet_to": 7}
    assert push.in_quiet(night, at(23)) and push.in_quiet(night, at(6)) and not push.in_quiet(night, at(7))
    day = {"quiet_from": 12, "quiet_to": 14}
    assert push.in_quiet(day, at(13)) and not push.in_quiet(day, at(14))
    assert not push.in_quiet({"quiet_from": 5, "quiet_to": 5}, at(5))
    assert not push.in_quiet({}, at(5))


def test_no_key_no_work(tmp_path, monkeypatch):
    for _, *names in push.DATABASES:
        for n in names:
            monkeypatch.delenv(n, raising=False)
    assert push.run(LocalStore(tmp_path), now=NOW, sender=Sender(), log=lambda *_: None) == {}
