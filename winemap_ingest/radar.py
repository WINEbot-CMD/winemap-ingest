"""Reading KNMI radar HDF5 files (real-time 5-minute and climatological hourly accumulations)."""

from __future__ import annotations

import io
import re
from datetime import UTC, datetime, timedelta

import h5py
import numpy as np

# RAD_NL25_RAC_RT_202610250950.h5 (5 min, real-time) / RAD_NL25_RAC_MFBS_01H_202607252000_NL.h5 (hour, final);
# the time in the name is the END of the accumulation
_NAME = re.compile(r"RAD_NL25_RAC_(RT|MFBS_01H)_(\d{12})")


def end_time(filename: str) -> datetime:
    m = _NAME.search(filename)
    if not m:
        raise ValueError(f"not a KNMI radar file name: {filename}")
    s = m.group(2)
    if s.endswith("2400"):  # the climatological files write midnight as 24:00 of the day before
        return datetime.strptime(s[:8], "%Y%m%d").replace(tzinfo=UTC) + timedelta(days=1)
    return datetime.strptime(s, "%Y%m%d%H%M").replace(tzinfo=UTC)


def _calibration(attrs) -> tuple[float, float]:
    """GEO = a * PV + b from the calibration formula string."""
    s = attrs["calibration_formulas"]
    s = s.decode() if isinstance(s, bytes) else str(s)
    m = re.search(r"GEO\s*=\s*([-+0-9.eE]+)\s*\*\s*PV\s*\+?\s*([-+0-9.eE]+)", s)
    if not m:
        raise ValueError(f"unknown calibration formula: {s}")
    return float(m.group(1)), float(m.group(2))


def read_mm(data: bytes) -> np.ndarray:
    """The precipitation accumulation in mm (765 x 700, float32, NaN = missing or outside the image)."""
    with h5py.File(io.BytesIO(data), "r") as f:
        raw = f["image1/image_data"][...]
        cal = f["image1/calibration"].attrs
        a, b = _calibration(cal)
        bad = set()
        for k in ("calibration_missing_data", "calibration_out_of_image"):
            if k in cal:
                bad.update(int(v) for v in np.atleast_1d(cal[k]))
    mm = raw.astype(np.float32) * a + b
    for v in bad:
        mm[raw == v] = np.nan
    return mm
