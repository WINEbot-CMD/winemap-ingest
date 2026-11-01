"""A small GRIB2 reader for DWD's ICON open data (docs/38 §10): one field per file, a regular lat/lon grid (template
3.0), CCSDS/AEC packing (template 5.42) or simple packing (5.0). Rasterio's GRIB driver and the eccodes wheels need a
libaec build that is missing on Windows; imagecodecs ships libaec everywhere, so the decoding is done here.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np


@dataclass
class Field:
    values: np.ndarray  # (nj, ni) float32, north row first, NaN = missing
    lon0: float  # west edge cell centre
    lat0: float  # north edge cell centre
    dlon: float
    dlat: float
    ref_time: datetime


def _signed(b: bytes) -> int:
    """GRIB's sign-and-magnitude integers."""
    v = int.from_bytes(b, "big")
    top = 1 << (8 * len(b) - 1)
    return -(v & (top - 1)) if v & top else v


def _sections(raw: bytes):
    if raw[:4] != b"GRIB" or raw[7] != 2:
        raise ValueError("not a GRIB2 message")
    pos = 16
    while pos < len(raw) - 4:
        if raw[pos : pos + 4] == b"7777":
            return
        n = int.from_bytes(raw[pos : pos + 4], "big")
        yield raw[pos + 4], raw[pos : pos + n]
        pos += n


def read(raw: bytes) -> Field:
    sec: dict[int, bytes] = {}
    for num, body in _sections(raw):
        sec.setdefault(num, body)  # the first message only
    s1, s3, s5, s6, s7 = sec[1], sec[3], sec[5], sec[6], sec[7]
    year = int.from_bytes(s1[12:14], "big")
    ref = datetime(year, s1[14], s1[15], s1[16], s1[17], s1[18], tzinfo=UTC)

    if int.from_bytes(s3[12:14], "big") != 0:
        raise ValueError("only regular lat/lon grids (template 3.0)")
    ni, nj = int.from_bytes(s3[30:34], "big"), int.from_bytes(s3[34:38], "big")
    la1, lo1 = _signed(s3[46:50]) / 1e6, _signed(s3[50:54]) / 1e6
    if lo1 >= 180:  # 0..360 -> -180..180
        lo1 -= 360
    di, dj = int.from_bytes(s3[63:67], "big") / 1e6, int.from_bytes(s3[67:71], "big") / 1e6
    scan = s3[71]

    npts = int.from_bytes(s5[5:9], "big")
    tpl = int.from_bytes(s5[9:11], "big")
    r = struct.unpack(">f", s5[11:15])[0]
    e, d = _signed(s5[15:17]), _signed(s5[17:19])
    nbits = s5[19]
    data = s7[5:]
    if nbits == 0:
        x = np.zeros(npts)
    elif tpl == 42:
        import imagecodecs

        flags, block, rsi = s5[21], s5[22], int.from_bytes(s5[23:25], "big")
        size = 1 if nbits <= 8 else 2 if nbits <= 16 else 3 if nbits <= 24 and flags & 2 else 4
        whole = -(-npts // (block * rsi)) * block * rsi  # libaec fills up to a whole reference-sample interval
        out = imagecodecs.aec_decode(data, bitspersample=nbits, flags=flags, blocksize=block, rsi=rsi,
                                     out=whole * size)  # fmt: skip
        b = np.frombuffer(out, np.uint8)[: npts * size].reshape(npts, size).astype(np.uint32)
        if not flags & 4:  # LSB first
            b = b[:, ::-1]
        x = np.zeros(npts, np.uint32)
        for k in range(size):
            x = (x << 8) | b[:, k]
    elif tpl == 0:
        bits = np.unpackbits(np.frombuffer(data, np.uint8))[: npts * nbits].reshape(npts, nbits)
        x = bits @ (1 << np.arange(nbits - 1, -1, -1, dtype=np.uint64))
    else:
        raise ValueError(f"data template 5.{tpl} not supported")
    vals = ((r + x.astype(np.float64) * 2.0**e) / 10.0**d).astype(np.float32)

    full = np.full(ni * nj, np.nan, np.float32)
    if s6[5] == 0:  # bitmap
        mask = np.unpackbits(np.frombuffer(s6[6:], np.uint8))[: ni * nj].astype(bool)
        full[mask] = vals
    else:
        full[:] = vals
    grid = full.reshape(nj, ni)
    if scan & 0x40:  # rows go south to north: flip so row 0 is the north
        grid = grid[::-1]
        la1 = la1 + (nj - 1) * dj
    return Field(np.ascontiguousarray(grid), lo1, la1, di, dj, ref)
