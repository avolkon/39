"""Постобработка предсказаний (физически осмысленные ограничения)."""

from __future__ import annotations

import pandas as pd

IC50_CC50_FLOOR = 1e-8


def postprocess(predictions: pd.DataFrame) -> pd.DataFrame:
    """Обрезка отрицательных IC50/CC50 и SI; копия входного DataFrame."""
    out = predictions.copy()
    if "IC50" in out.columns:
        out["IC50"] = out["IC50"].clip(lower=IC50_CC50_FLOOR)
    if "CC50" in out.columns:
        out["CC50"] = out["CC50"].clip(lower=IC50_CC50_FLOOR)
    if "SI" in out.columns:
        out["SI"] = out["SI"].clip(lower=0.0)
    return out
