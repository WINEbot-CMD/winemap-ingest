"""Upload a local store folder (e.g. the laptop backfill) to R2: only files that are missing or differ in size, and
`rain/index.json` last, so the site never sees an index that points at files that are not there yet.

    python -m winemap_ingest.sync FOLDER [--prefix rain/]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .store import R2Store

TYPES = {".tif": "image/tiff", ".json": "application/json"}


def remote_sizes(store: R2Store, prefix: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for page in store.s3.get_paginator("list_objects_v2").paginate(Bucket=store.bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"]] = o["Size"]
    return out


def sync(folder: Path, store: R2Store, prefix: str = "rain/", workers: int = 16, log=print) -> int:
    files = {p.relative_to(folder).as_posix(): p for p in (folder / prefix).rglob("*") if p.is_file()}
    files = {k: p for k, p in files.items() if not k.endswith(".part")}
    have = remote_sizes(store, prefix)
    todo = [k for k, p in sorted(files.items()) if have.get(k) != p.stat().st_size]
    last = [k for k in todo if k.endswith("index.json")]
    todo = [k for k in todo if k not in last]
    log(f"{len(files)} files, {len(todo) + len(last)} to upload")

    def up(k: str) -> None:
        store.put(k, files[k].read_bytes(), TYPES.get(files[k].suffix, "application/octet-stream"))

    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for _ in pool.map(up, todo):
            done += 1
            if done % 500 == 0:
                log(f"{done} / {len(todo)}")
    for k in last:
        up(k)
    return len(todo) + len(last)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="winemap_ingest.sync")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--prefix", default="rain/")
    a = ap.parse_args(argv)
    n = sync(a.folder, R2Store(), a.prefix)
    print(f"uploaded {n} files")


if __name__ == "__main__":
    main()
