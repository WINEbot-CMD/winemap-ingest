"""The plant protection products authorised for wine grapes in the Netherlands (winemap-nl docs/27, Jasper 28 Oct 2026:
"sprays from the Ctgb list"), from the Ctgb's official public API (MST public API; Ctgb content is CC0), once a day:

    ctgb/wijndruif.json   {"updated", "source", "products": [{id, name, substances: [{name, concentration, unit}],
                           types, lowRisk, expires, uses: [{targets, maxDose, unit, stages, months, interval}]}]}

in the public bucket (served as /wx/ctgb/wijndruif.json), for the product choice in the log. Only the uses on grapes
are kept; maxDose is the reference dose for the treatment frequency index (docs/28).

    python -m winemap_ingest.ctgb
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime

from .store import open_store

API = "https://public.mst.ctgb.nl/public-api/1.0"
CROP = "Wijndruif::gewas"
GRAPE = {"Wijndruif", "Druif"}
KEY = "ctgb/wijndruif.json"


def get(path: str, params: dict | None = None, opener=urllib.request.urlopen) -> dict:
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(
        url, headers={"User-Agent": "winemap-ingest (winemap-nl)", "Accept": "application/json"}
    )
    with opener(req, timeout=120) as r:
        return json.loads(r.read())


def _crops(node) -> set[str]:
    """Every crop name in a use's crop tree."""
    out = set()
    if isinstance(node, dict):
        if node.get("cropNL"):
            out.add(node["cropNL"])
        if node.get("groupNL"):
            out.add(node["groupNL"])
        for v in node.values():
            out |= _crops(v)
    elif isinstance(node, list):
        for v in node:
            out |= _crops(v)
    return out


def _targets(node) -> list[str]:
    """The diseases / pests a use is for (Dutch names, as in the database)."""
    out: list[str] = []
    if isinstance(node, dict):
        for d in node.get("diseasesNL") or []:
            if d not in out:
                out.append(d)
        if not node.get("diseasesNL") and node.get("organismNL") and node["organismNL"] not in out:
            out.append(node["organismNL"])
        for v in node.values():
            for t in _targets(v):
                if t not in out:
                    out.append(t)
    elif isinstance(node, list):
        for v in node:
            for t in _targets(v):
                if t not in out:
                    out.append(t)
    return out


def _number(v):
    """A number when it is one; otherwise the text as given (micro-organisms: '5x10^10' per gram), or None."""
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def product(record: dict) -> dict | None:
    """One authorisation -> what the log needs; None when it has no use on grapes."""
    uses = []
    for u in record.get("uses") or []:
        if not (_crops(u.get("targetCrops")) & GRAPE):
            continue
        dose = u.get("maximumProductDose") or {}
        timing = u.get("applicationTiming") or {}
        stage = u.get("growthStage") or {}
        uses.append(
            {
                "targets": _targets(u.get("targetOrganisms"))[:12],
                "maxDose": _number(dose.get("ratio")),
                "unit": (dose.get("measure") or {}).get("unitNL"),
                "stages": [stage.get("from"), stage.get("to")],
                "months": [timing.get("fromMonth"), timing.get("toMonth")],
                "interval": u.get("minimumIntervalBetweenApplications"),
                "maxApplications": u.get("maximumNumberOfApplications"),
            }
        )
    if not uses:
        return None
    subs = []
    for s in (record.get("compositions") or {}).get("substances") or []:
        if not (s.get("function") or {}).get("isActiveSubstance"):
            continue
        subs.append(
            {
                "name": (s.get("substance") or {}).get("nameNL"),
                "concentration": _number(s.get("concentration")),
                "unit": (s.get("concentrationUnit") or {}).get("unitNL"),
                "lowRisk": (s.get("substance") or {}).get("lowRisk"),
            }
        )
    expires = None
    for a in (record.get("authorisation") or {}).get("actual") or []:
        for w in a.get("wCodings") or []:
            if w.get("expirationDate") and (expires is None or w["expirationDate"] > expires):
                expires = w["expirationDate"]
    return {
        "id": str(record.get("id")),
        "name": record.get("name"),
        "substances": subs,
        "types": [t.get("descriptionNL") for t in record.get("outcomeTypes") or []],
        "lowRisk": record.get("lowRisk"),
        "expires": (expires or "")[:10] or None,
        "uses": uses,
    }


def run(store, log=print, fetch=get) -> int:
    ids: list[str] = []
    offset = 0
    while True:
        page = fetch(
            "/authorisations",
            {
                "filter[pppTargetCrops]": CROP,
                "filter[productStatus]": "Valid",
                "filter[categoryType]": "PPP",
                "page[limit]": 100,
                "page[offset]": offset,
                "summary": "true",
            },
        )
        ids += [str(x["id"]) for x in page["data"]]
        offset += len(page["data"])
        if not page["data"] or offset >= page["meta"]["total"]:
            break
    products = []
    for i in ids:
        p = product(fetch(f"/authorisations/{i}")["data"])
        if p:
            products.append(p)
        time.sleep(0.2)  # gently: one product after the other
    products.sort(key=lambda p: (p["name"] or "").lower())
    doc = {
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "Ctgb toelatingendatabank (MST public API), CC0 - www.ctgb.nl; the uses on wine grapes only",
        "products": products,
    }
    store.put(KEY, json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode(), "application/json")
    log(f"{len(products)} products for wine grapes written to {KEY}")
    return len(products)


def main() -> None:
    run(open_store())


if __name__ == "__main__":
    main()
