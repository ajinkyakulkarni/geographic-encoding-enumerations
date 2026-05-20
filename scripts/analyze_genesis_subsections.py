#!/usr/bin/env python3
"""
Genesis 10 within-subsection analysis.

The Table of Nations is a genealogy in three branches — the descendants
of Japheth, Ham, and Shem. The flat-textual test (analyze_corpus.py on
the whole list) finds only a weak global signal; the paper's claim is
that each branch is internally latitudinally ordered, and that the
three branches run in different directions and cancel when concatenated.

This script computes the per-branch Kendall's tau and writes
results/per_corpus/genesis10_subsections.json. It is the reproducible
source for the three sub-section numbers reported in the paper's
Section 3.4 / Table 1 (Japheth, Ham, Shem latitudinal tau).

Sub-section boundaries are the canonical text-position ranges of
Genesis 10:2-5 (Japheth), 10:6-20 (Ham), 10:21-31 (Shem), as recorded
in the `position` column of corpora/genesis10/nations.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REPO_ROOT = Path(__file__).resolve().parent.parent
NATIONS_CSV = REPO_ROOT / "corpora" / "genesis10" / "nations.csv"
OUT_JSON = REPO_ROOT / "results" / "per_corpus" / "genesis10_subsections.json"

# Canonical text-position ranges (inclusive) of the three Noachic branches.
SUBSECTIONS = {
    "Japheth": (1, 14),    # Genesis 10:2-5
    "Ham":     (15, 52),   # Genesis 10:6-20
    "Shem":    (53, 78),   # Genesis 10:21-31
}


def analyse_branch(df: pd.DataFrame, lo: int, hi: int) -> dict:
    """Kendall's tau for one branch: keep identified items in the text-
    position window, re-rank them 1..N in textual order, correlate the
    rank against longitude and latitude.
    """
    sub = df[(df["position"] >= lo) & (df["position"] <= hi)].copy()
    sub = sub.dropna(subset=["centroid_lon", "centroid_lat"])
    sub = sub.sort_values("position").reset_index(drop=True)
    n = len(sub)
    rank = np.arange(1, n + 1)
    t_lon, p_lon = kendalltau(rank, sub["centroid_lon"].to_numpy())
    t_lat, p_lat = kendalltau(rank, sub["centroid_lat"].to_numpy())
    return {
        "n_identified": int(n),
        "tau_longitude": float(t_lon),
        "p_longitude": float(p_lon),
        "tau_latitude": float(t_lat),
        "p_latitude": float(p_lat),
    }


def main() -> int:
    df = pd.read_csv(NATIONS_CSV)
    out = {
        "corpus": "Genesis 10 — within-subsection (three Noachic branches)",
        "source_csv": str(NATIONS_CSV.relative_to(REPO_ROOT)),
        "subsections": {},
    }
    print("Genesis 10 within-subsection analysis")
    print("=" * 60)
    for name, (lo, hi) in SUBSECTIONS.items():
        res = analyse_branch(df, lo, hi)
        out["subsections"][name] = {"text_positions": [lo, hi], **res}
        print(f"  {name:8s} (n={res['n_identified']:2d})  "
              f"tau_lat = {res['tau_latitude']:+.3f}  p = {res['p_latitude']:.4f}   "
              f"tau_lon = {res['tau_longitude']:+.3f}  p = {res['p_longitude']:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
