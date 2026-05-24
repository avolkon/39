"""Доменные химические признаки поверх 214 дескрипторов."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-6


def add_chem_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки; отсутствующие исходные колонки пропускаются."""
    out = df.copy()

    if "MolLogP" in out.columns and "TPSA" in out.columns:
        out["LogP_TPSA"] = out["MolLogP"] / (out["TPSA"] + EPS)
    else:
        logger.warning("LogP_TPSA: нет MolLogP и/или TPSA")

    arom = None
    if "NumAromaticRings" in out.columns:
        arom = out["NumAromaticRings"]
    elif "fr_benzene" in out.columns:
        arom = out["fr_benzene"]

    if arom is not None and "HeavyAtomCount" in out.columns:
        out["Arom_Heavy_ratio"] = arom / (out["HeavyAtomCount"] + EPS)
    else:
        logger.warning("Arom_Heavy_ratio: недостаточно колонок")

    if "MaxPartialCharge" in out.columns and "MinPartialCharge" in out.columns:
        out["Charge_sum"] = out["MaxPartialCharge"] + out["MinPartialCharge"]
    else:
        logger.warning("Charge_sum: нет зарядовых колонок")

    if "fr_imide" in out.columns:
        out["fr_imide_flag"] = (out["fr_imide"] > 0).astype(np.float64)
    else:
        logger.warning("fr_imide_flag: нет fr_imide")

    if "fr_sulfone" in out.columns:
        out["fr_sulfone_flag"] = (out["fr_sulfone"] > 0).astype(np.float64)
    else:
        logger.warning("fr_sulfone_flag: нет fr_sulfone")

    if "RingCount" in out.columns and "MolLogP" in out.columns:
        out["Ring_LogP"] = out["RingCount"] * out["MolLogP"]
    else:
        logger.warning("Ring_LogP: нет RingCount и/или MolLogP")

    return out
