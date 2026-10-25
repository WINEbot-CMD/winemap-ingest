"""KNMI Open Data API client that keeps to the request limit (1 000 per hour per key, shared with other jobs)."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

API = "https://api.dataplatform.knmi.nl/open-data/v1/datasets"
REALTIME = ("nl_rdr_data_rtcor_5m", "1.0")  # radar 5-minute accumulations, real-time (gauge-adjusted)
FINAL_HOURLY = ("rad_nl25_rac_mfbs_01h", "2.0")  # climatological hourly accumulations, one zip per year


class OutOfRequests(Exception):
    """The hourly request limit is (almost) used up: stop and continue in the next run."""


class Knmi:
    def __init__(self, key: str | None = None, keep: int = 150, log=print):
        self.key = key or os.environ.get("KNMI_API_KEY", "")
        if not self.key:
            raise SystemExit("KNMI_API_KEY is not set")
        self.keep = keep  # requests left for other jobs on the same key
        self.remaining: int | None = None
        self.log = log

    def _get(self, url: str, raw: bool = False):
        if self.remaining is not None and self.remaining <= self.keep:
            raise OutOfRequests(f"{self.remaining} requests left this hour")
        req = urllib.request.Request(url, headers={"Authorization": self.key})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    left = r.headers.get("X-Ratelimit-Remaining")
                    if left is not None:
                        self.remaining = int(left)
                    body = r.read()
                return body if raw else json.loads(body)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    raise OutOfRequests("KNMI answered 429 (too many requests)") from e
                if e.code >= 500 and attempt < 3:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise

    def list_files(self, dataset: tuple[str, str], begin: str | None = None, limit: int = 500) -> list[str]:
        """File names in name order, starting after `begin` (a file name)."""
        name, version = dataset
        out: list[str] = []
        token = None
        while True:
            q = {"maxKeys": str(min(limit - len(out), 500)), "orderBy": "filename", "sorting": "asc"}
            if begin:
                q["begin"] = begin
            if token:
                q["nextPageToken"] = token
            j = self._get(f"{API}/{name}/versions/{version}/files?{urllib.parse.urlencode(q)}")
            out += [f["filename"] for f in j.get("files", []) if f["filename"] != begin]
            token = j.get("nextPageToken")
            if not token or not j.get("isTruncated") or len(out) >= limit:
                return out

    def list_files_desc(self, dataset: tuple[str, str], limit: int = 1) -> list[str]:
        """The newest files (by creation time), newest first."""
        name, version = dataset
        q = {"maxKeys": str(limit), "orderBy": "created", "sorting": "desc"}
        j = self._get(f"{API}/{name}/versions/{version}/files?{urllib.parse.urlencode(q)}")
        return [f["filename"] for f in j.get("files", [])]

    def file_info(self, dataset: tuple[str, str], limit: int = 100) -> dict[str, str]:
        """File name -> lastModified, for datasets with a few files (the yearly archives)."""
        name, version = dataset
        q = {"maxKeys": str(limit), "orderBy": "filename", "sorting": "asc"}
        j = self._get(f"{API}/{name}/versions/{version}/files?{urllib.parse.urlencode(q)}")
        return {f["filename"]: f["lastModified"] for f in j.get("files", [])}

    def download(self, dataset: tuple[str, str], filename: str) -> bytes:
        name, version = dataset
        j = self._get(f"{API}/{name}/versions/{version}/files/{urllib.parse.quote(filename)}/url")
        with urllib.request.urlopen(j["temporaryDownloadUrl"], timeout=300) as r:  # S3 link: no API request
            return r.read()
