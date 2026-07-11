#!/usr/bin/env python
"""Verify notebook submission path: 2 independent author4 runs + EDA metrics."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ml" / "data"
REF = ROOT / "docs" / "submissions" / "submission2_a4.csv"
AUTHOR4 = ROOT / "tools" / "author4_high_priority_submissions.py"
RUNS_DIR = ROOT / "notebooks" / "_verify_runs"
TARGETS = ("IC50", "CC50", "SI")
RANDOM_STATE = 42


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_author4(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CHEM_DATA_DIR"] = str(DATA_DIR)
    env["CHEM_SUBMISSIONS_DIR"] = str(out_dir)
    env["CHEM_RANDOM_SEED"] = "42"
    proc = subprocess.run(
        [sys.executable, str(AUTHOR4), "--only", "2"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"author4 failed for {out_dir}")
    out = out_dir / "submission2_a4.csv"
    if not out.is_file():
        raise FileNotFoundError(out)
    return out


def collect_eda() -> dict:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    feature_cols = [c for c in train_df.columns if c not in ("index", *TARGETS)]
    ratio = train_df["CC50"] / train_df["IC50"].clip(lower=1e-8)
    si_max_err = float((train_df["SI"] - ratio).abs().max())

    num = train_df[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num)
    pca = PCA(random_state=RANDOM_STATE).fit(X_scaled)
    cum2 = float(np.cumsum(pca.explained_variance_ratio_)[1])

    ica = FastICA(n_components=3, random_state=RANDOM_STATE, max_iter=500)
    ic = ica.fit_transform(X_scaled)

    sys.path.insert(0, str(ROOT / "ml" / "src"))
    from chemai.features.build_features import add_chem_features
    from chemai.models.candidate_models import build_default_candidates

    x = add_chem_features(
        train_df.drop(columns=list(TARGETS)).drop(columns=["index"], errors="ignore")
    )
    new_cols = [c for c in x.columns if c not in feature_cols]
    candidates = build_default_candidates(RANDOM_STATE)

    return {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "n_features": len(feature_cols),
        "targets_skew": {t: float(train_df[t].skew()) for t in TARGETS},
        "si_identity_max_err": si_max_err,
        "pca_pc1_pc2_cum_variance": cum2,
        "ica_n_components": ic.shape[1],
        "domain_features_added": new_cols,
        "n_candidates": len(candidates),
        "candidate_names": [c.name for c in candidates],
    }


def main() -> int:
    if not REF.is_file():
        raise FileNotFoundError(f"Reference missing: {REF}")

    ref_hash = md5_file(REF)
    print("Reference submission2_a4 md5:", ref_hash)

    paths = []
    for i in (1, 2):
        p = run_author4(RUNS_DIR / f"run_{i}")
        h = md5_file(p)
        print(f"Run {i} md5:", h)
        paths.append((p, h))

    stable_bytes = paths[0][1] == paths[1][1] == ref_hash
    import pandas as pd

    ref_df = pd.read_csv(REF)
    r1_df = pd.read_csv(paths[0][0])
    r2_df = pd.read_csv(paths[1][0])
    max_ref_r1 = float((ref_df[["IC50", "CC50", "SI"]] - r1_df[["IC50", "CC50", "SI"]]).abs().max().max())
    max_r1_r2 = float((r1_df[["IC50", "CC50", "SI"]] - r2_df[["IC50", "CC50", "SI"]]).abs().max().max())
    stable_numeric = max_ref_r1 < 1e-6 and max_r1_r2 < 1e-6

    print("STABLE (bytes/md5):", stable_bytes)
    print("STABLE (numeric rtol 1e-6):", stable_numeric)
    print("max |Δ| ref vs run1:", max_ref_r1)
    print("max |Δ| run1 vs run2:", max_r1_r2)

    sidecar = RUNS_DIR / "run_2" / "submission2_a4.metrics_sidecar.json"
    stack_metrics = json.loads(sidecar.read_text(encoding="utf-8"))
    eda = collect_eda()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_file": str(REF),
        "reference_md5": ref_hash,
        "run1_md5": paths[0][1],
        "run2_md5": paths[1][1],
        "stable_across_runs_bytes": stable_bytes,
        "stable_across_runs_numeric": stable_numeric,
        "max_abs_diff_ref_run1": max_ref_r1,
        "max_abs_diff_run1_run2": max_r1_r2,
        "stacking_oof": stack_metrics,
        "eda": eda,
        "public_lb_historical": 349.30995,
    }

    out_json = ROOT / "notebooks" / "_verify_runs" / "verification_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Report:", out_json)
    return 0 if stable_numeric else 1


if __name__ == "__main__":
    raise SystemExit(main())
