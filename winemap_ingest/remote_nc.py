"""Read a slab of rows from a large remote NetCDF file by HTTP range, without downloading the file (Winemap docs/38).

The World Bank's CMIP6 time series are 4.3 GB per variable for the world, stored without compression, one month after
the other: the classic NetCDF format (CDF-1/2, big-endian, contiguous) or HDF5 with one chunk per month, some with
HDF5's byte shuffle (no compression, but the 1st bytes of all values first, then the 2nd, ...). Either way the rows of
a latitude band (Europe: ~0.9 of 4.3 MB a month) are one range request, or one per byte with the shuffle.

    nc = RemoteGrid(url)                   # reads the header (a few requests)
    nc.lat, nc.lon, nc.n_time              # the axes
    nc.rows(month, r0, r1)                 # float32 (r1 - r0, n_lon), NaN for the fill value
"""

from __future__ import annotations

import io
import struct
import time
import urllib.request
import zlib

import numpy as np

TYPES = {1: ("b", 1), 2: ("c", 1), 3: (">i2", 2), 4: (">i4", 4), 5: (">f4", 4), 6: (">f8", 8)}


def fetch(url: str, start: int, end: int, tries: int = 5) -> bytes:
    """Bytes start..end (inclusive), with a few retries (S3 sometimes drops a connection)."""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except OSError:
            if i == tries - 1:
                raise
        time.sleep(2**i)
    raise OSError(f"short read {url} {start}-{end}")


class _Header:
    """A parser of the classic NetCDF header (the format spec: 'NetCDF Classic and 64-bit Offset Format')."""

    def __init__(self, b: bytes):
        self.b, self.p = b, 4
        self.version = b[3]

    def i4(self) -> int:
        v = struct.unpack(">i", self.b[self.p : self.p + 4])[0]
        self.p += 4
        return v

    def i8(self) -> int:
        v = struct.unpack(">q", self.b[self.p : self.p + 8])[0]
        self.p += 8
        return v

    def name(self) -> str:
        n = self.i4()
        s = self.b[self.p : self.p + n].decode()
        self.p += (n + 3) // 4 * 4
        return s

    def attrs(self) -> dict:
        tag, n = self.i4(), self.i4()
        out = {}
        if tag == 0:
            return out
        for _ in range(n):
            key, t, m = self.name(), self.i4(), self.i4()
            code, size = TYPES[t]
            raw = self.b[self.p : self.p + m * size]
            self.p += (m * size + 3) // 4 * 4
            out[key] = raw.decode(errors="replace") if code == "c" else np.frombuffer(raw, dtype=code)
        return out

    def parse(self):
        numrecs = self.i4()
        tag, n = self.i4(), self.i4()
        dims = [(self.name(), self.i4()) for _ in range(n)] if tag else []
        self.attrs()
        tag, n = self.i4(), self.i4()
        vars_ = {}
        for _ in range(n if tag else 0):
            name = self.name()
            ids = [self.i4() for _ in range(self.i4())]
            attrs = self.attrs()
            t, vsize = self.i4(), self.i4()
            begin = self.i8() if self.version == 2 else self.i4()
            vars_[name] = dict(dims=ids, attrs=attrs, type=TYPES[t][0], size=TYPES[t][1], vsize=vsize, begin=begin)
        return numrecs, dims, vars_


class RemoteGrid:
    """One (time, lat, lon) variable of a remote NetCDF file; the variable is the one with three dimensions."""

    def __init__(self, url: str):
        self.url = url
        head = fetch(url, 0, 1 << 16)
        if head[:3] == b"CDF":
            self._classic(head)
        elif head[:8] == b"\x89HDF\r\n\x1a\n":
            self._hdf5()
        else:
            raise ValueError(f"not NetCDF: {url}")

    def _classic(self, head: bytes):
        numrecs, dims, vs = _Header(head).parse()
        name = next(k for k, v in vs.items() if len(v["dims"]) == 3)
        v = vs[name]
        rec = [i for i, (_, n) in enumerate(dims) if n == 0]  # the record (unlimited) dimension, if any
        n_lat, n_lon = dims[v["dims"][1]][1], dims[v["dims"][2]][1]
        is_rec = rec and v["dims"][0] == rec[0]
        recsize = sum(x["vsize"] for x in vs.values() if rec and x["dims"] and x["dims"][0] == rec[0])
        if is_rec and sum(1 for x in vs.values() if x["dims"] and x["dims"][0] == rec[0]) == 1:
            recsize = n_lat * n_lon * v["size"]  # one record variable: no padding between records
        self.n_time = numrecs if is_rec else dims[v["dims"][0]][1]
        self._step = recsize if is_rec else n_lat * n_lon * v["size"]
        self._begin, self._dtype, self._size = v["begin"], v["type"], v["size"]
        self.fill = float(v["attrs"].get("_FillValue", [np.nan])[0])
        self.n_lon = n_lon
        self.lat = self._axis(vs, dims, v["dims"][1])
        self.lon = self._axis(vs, dims, v["dims"][2])
        self.name = name

    def _axis(self, vs, dims, dim_id) -> np.ndarray:
        dim = dims[dim_id][0]
        a = vs[dim]
        raw = fetch(self.url, a["begin"], a["begin"] + dims[dim_id][1] * a["size"] - 1)
        return np.frombuffer(raw, dtype=a["type"]).astype("f8")

    def _hdf5(self):
        import h5py

        f = h5py.File(io.BufferedReader(_RangeFile(self.url), buffer_size=1 << 20), "r")
        name = next(k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3)
        d = f[name]
        pl = d.id.get_create_plist()
        filters = {pl.get_filter(i)[0] for i in range(pl.get_nfilters())}
        if filters - {h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE} or d.chunks not in (None, (1,) + d.shape[1:]):
            raise ValueError(f"HDF5 layout not one month per chunk: {d.chunks} filters {filters}")
        self._shuffle = h5py.h5z.FILTER_SHUFFLE in filters
        # compressed (deflate): a month's whole chunk is fetched and unpacked, the rows can not be read on their own
        self._deflate = h5py.h5z.FILTER_DEFLATE in filters
        self.n_time, n_lat, self.n_lon = d.shape
        self._n_elem = n_lat * self.n_lon
        chunk = d.id.get_chunk_info_by_coord
        info = [chunk((t, 0, 0)) for t in range(self.n_time)] if d.chunks else None
        self._offsets = [i.byte_offset for i in info] if info else None
        self._sizes = [i.size for i in info] if info else None
        if self._offsets is None:
            self._begin, self._step = d.id.get_offset(), n_lat * self.n_lon * d.dtype.itemsize
        self._dtype, self._size = d.dtype.str, d.dtype.itemsize
        fv = d.attrs.get("_FillValue")
        self.fill = float(fv[0] if fv is not None and np.ndim(fv) else fv) if fv is not None else np.nan
        self.lat, self.lon, self.name = f["lat"][:].astype("f8"), f["lon"][:].astype("f8"), name

    def rows(self, t: int, r0: int, r1: int) -> np.ndarray:
        """Rows r0..r1-1 of month t as float32; the fill value becomes NaN."""
        base = self._offsets[t] if getattr(self, "_offsets", None) else self._begin + t * self._step
        if getattr(self, "_deflate", False):
            raw = zlib.decompress(fetch(self.url, base, base + self._sizes[t] - 1))
            if self._shuffle:  # the bytes were grouped per position: k-th bytes of all values, then the next
                raw = np.frombuffer(raw, "u1").reshape(self._size, self._n_elem).T.tobytes()
            row = self.n_lon * self._size
            raw = raw[r0 * row : r1 * row]
        elif getattr(self, "_shuffle", False):
            # byte k of value e sits at k * (values in the chunk) + e
            e0, e1 = r0 * self.n_lon, r1 * self.n_lon
            starts = [base + k * self._n_elem for k in range(self._size)]
            planes = [np.frombuffer(fetch(self.url, b + e0, b + e1 - 1), "u1") for b in starts]
            raw = np.stack(planes, axis=-1).tobytes()
        else:
            row = self.n_lon * self._size
            raw = fetch(self.url, base + r0 * row, base + r1 * row - 1)
        a = np.frombuffer(raw, dtype=self._dtype).reshape(r1 - r0, self.n_lon).astype("f4")
        a[(a == self.fill) | (np.abs(a) > 1e19)] = np.nan
        return a


class _RangeFile(io.RawIOBase):
    """A read-only file over HTTP range requests, for h5py to read the HDF5 header and chunk index."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        data = fetch(self.url, self.pos, min(self.size, self.pos + len(b)) - 1)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)
