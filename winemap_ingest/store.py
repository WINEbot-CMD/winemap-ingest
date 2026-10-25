"""Where the results go: Cloudflare R2 (S3 API) online, or a local folder (tests, trial runs, the laptop backfill)."""

from __future__ import annotations

import os
from pathlib import Path


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get(self, key: str) -> bytes | None:
        p = self.root / key
        return p.read_bytes() if p.exists() else None

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)

    def __repr__(self) -> str:
        return f"LocalStore({self.root})"


class R2Store:
    """R2 bucket through its S3 API (account id, access key and secret from the environment)."""

    def __init__(self, bucket: str | None = None):
        import boto3  # only needed online

        account = os.environ["R2_ACCOUNT_ID"]
        self.bucket = bucket or os.environ.get("R2_BUCKET", "winemap-wx")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    def get(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def __repr__(self) -> str:
        return f"R2Store({self.bucket})"


def open_store(where: str | None = None):
    """`r2` or a folder path; default from WX_STORE."""
    where = where or os.environ.get("WX_STORE", "")
    if not where:
        raise SystemExit("WX_STORE is not set (use 'r2' or a folder)")
    return R2Store() if where == "r2" else LocalStore(where)
