import json

from winemap_ingest import ctgb
from winemap_ingest.store import LocalStore

GRAPE_USE = {
    "targetCrops": [{"groupNL": "Fruitteelt", "items": [{"groupNL": "Druif", "items": [{"cropNL": "Wijndruif"}]}]}],
    "targetOrganisms": [{"items": [{"organismScientific": "Plasmopara viticola", "diseasesNL": ["Valse meeldauw"]}]}],
    "maximumProductDose": {"ratio": "0.5", "measure": {"unitNL": "kg/ha"}},
    "growthStage": {"from": 13, "to": 81},
    "applicationTiming": {"fromMonth": 5, "toMonth": 8},
    "minimumIntervalBetweenApplications": 10,
}
OTHER_USE = {**GRAPE_USE, "targetCrops": [{"cropNL": "Tomaat"}]}
RECORD = {
    "id": 123,
    "name": "Testmiddel",
    "lowRisk": False,
    "outcomeTypes": [{"descriptionNL": "Schimmelbestrijdingsmiddel"}],
    "compositions": {
        "substances": [
            {
                "substance": {"nameNL": "zwavel", "lowRisk": False},
                "function": {"isActiveSubstance": True},
                "concentration": "800",
                "concentrationUnit": {"unitNL": "G/KG"},
            },
            {"substance": {"nameNL": "Bacillus"}, "function": {"isActiveSubstance": True}, "concentration": "5x10^10"},
            {"substance": {"nameNL": "hulpstof"}, "function": {"isActiveSubstance": False}},
        ]
    },
    "authorisation": {"actual": [{"wCodings": [{"expirationDate": "2027-07-30T22:00:00Z"}]}]},
    "uses": [GRAPE_USE, OTHER_USE],
}


def test_a_product_keeps_its_grape_uses_and_active_substances():
    p = ctgb.product(RECORD)
    assert p["name"] == "Testmiddel" and p["expires"] == "2027-07-30"
    assert [s["name"] for s in p["substances"]] == ["zwavel", "Bacillus"]
    assert p["substances"][0]["concentration"] == 800.0 and p["substances"][1]["concentration"] == "5x10^10"
    assert len(p["uses"]) == 1
    u = p["uses"][0]
    assert u["targets"] == ["Valse meeldauw"] and u["maxDose"] == 0.5 and u["unit"] == "kg/ha"
    assert u["stages"] == [13, 81] and u["months"] == [5, 8] and u["interval"] == 10
    assert ctgb.product({**RECORD, "uses": [OTHER_USE]}) is None


def test_run_pages_and_writes_the_list(tmp_path):
    def fetch(path, params=None):
        if path == "/authorisations":
            return (
                {"meta": {"total": 1}, "data": [{"id": 123}]}
                if params["page[offset]"] == 0
                else {"meta": {}, "data": []}
            )
        return {"data": RECORD}

    store = LocalStore(tmp_path)
    assert ctgb.run(store, log=lambda *_: None, fetch=fetch) == 1
    doc = json.loads(store.get(ctgb.KEY))
    assert doc["products"][0]["id"] == "123" and "CC0" in doc["source"]
