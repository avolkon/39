"""Постобработка предсказаний (физически осмысленные ограничения)."""

from __future__ import annotations

import numpy as np
import pandas as pd

IC50_CC50_FLOOR = 1e-8


def postprocess(
    predictions: pd.DataFrame,
    *,
    si_domain_blend: float = 0.0,
) -> pd.DataFrame:
    """Обрезка отрицательных IC50/CC50 и SI; опциональная доменная смесь SI с CC50/IC50."""
    out = predictions.copy()
    if "IC50" in out.columns:
        out["IC50"] = out["IC50"].clip(lower=IC50_CC50_FLOOR)
    if "CC50" in out.columns:
        out["CC50"] = out["CC50"].clip(lower=IC50_CC50_FLOOR)
    if "SI" in out.columns:
        out["SI"] = out["SI"].clip(lower=0.0)

    w = float(si_domain_blend)
    if w > 0.0 and {"IC50", "CC50", "SI"}.issubset(out.columns):
        ic = out["IC50"].to_numpy(dtype=np.float64)
        cc = out["CC50"].to_numpy(dtype=np.float64)
        si = out["SI"].to_numpy(dtype=np.float64)
        ratio = np.clip(cc, IC50_CC50_FLOOR, None) / np.clip(ic, IC50_CC50_FLOOR, None)
        out["SI"] = w * si + (1.0 - w) * ratio

    return out
