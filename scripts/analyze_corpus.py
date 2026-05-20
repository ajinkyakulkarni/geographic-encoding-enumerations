#!/usr/bin/env python3
"""
Corpus-generic analyzer — runs the full method battery (Kendall's tau,
Procrustes, Mantel, Bayes factors) on any enumerative corpus given as
a CSV.

The corpus CSV must have columns:
    position             1..n verse/list position (integer)
    name (or equivalent) display name of the item
    centroid_lon         longitude (decimal degrees)
    centroid_lat         latitude (decimal degrees)

Optional columns (used if present):
    syllable_count       if absent, constrained-permutation null is skipped
    confidence           informational only

Usage:
    python3 scripts/analyze_corpus.py <corpus_csv> <results_json>

Example:
    python3 scripts/analyze_corpus.py \
        corpora/avesta_fargard1/lands.csv \
        results/per_corpus/avesta_fargard1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# Allow `from methods import ...` regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import (
    bayes_factors,
    constrained_perm_null,
    mantel_test,
    procrustes_test,
)

SEED = 20260518
N_PERM = 10_000


def kendall_with_perm(verse_rank: np.ndarray, geo: np.ndarray,
                      rng: np.random.Generator, n_perm: int = N_PERM) -> dict:
    tau_obs, _ = kendalltau(verse_rank, geo)
    n = len(verse_rank)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        t, _ = kendalltau(perm, geo)
        null[i] = t
    return {
        "tau": float(tau_obs),
        "p_two_sided": float(np.mean(np.abs(null) >= abs(tau_obs))),
        "n_perm": n_perm,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus_csv")
    ap.add_argument("results_json")
    ap.add_argument("--no-meter", action="store_true",
                    help="Skip the constrained-permutation (meter) null. "
                         "Use for prose corpora where syllable shape does "
                         "not constrain ordering.")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.corpus_csv).sort_values("position").reset_index(drop=True)
    if "centroid_lon" not in df.columns or "centroid_lat" not in df.columns:
        sys.stderr.write(
            "Corpus CSV must have centroid_lon / centroid_lat columns.\n"
        )
        return 1

    # Drop rows lacking coordinates (unidentified items) and re-rank the
    # remaining items 1..N preserving original textual order. This is the
    # right move for both Kendall τ (rank-based, unaffected) and Procrustes
    # (which interprets verse positions as 1D coordinates and is sensitive
    # to gaps).
    n_total_in_csv = len(df)
    df = df.dropna(subset=["centroid_lon", "centroid_lat"]).reset_index(drop=True)
    n_identified = len(df)
    if n_identified < 4:
        sys.stderr.write(
            f"Only {n_identified} identified items remain — Kendall τ is "
            f"undefined or trivial for n < 4. Aborting.\n"
        )
        return 1
    if n_total_in_csv != n_identified:
        sys.stderr.write(
            f"Filtered {n_total_in_csv - n_identified} rows lacking "
            f"coordinates; analyzing {n_identified} identified items.\n"
        )
    # Re-rank the textual positions to consecutive integers 1..N.
    df["position"] = np.arange(1, n_identified + 1)

    verse_rank = df["position"].to_numpy()
    lons = df["centroid_lon"].to_numpy()
    lats = df["centroid_lat"].to_numpy()
    coords = np.column_stack([lons, lats])
    n = len(df)

    rng = np.random.default_rng(SEED)

    # ---- Kendall τ vs each axis ----
    kendall_lon = kendall_with_perm(verse_rank, lons, rng)
    kendall_lat = kendall_with_perm(verse_rank, lats, rng)
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = centered @ eigvecs[:, -1]
    kendall_pc1 = kendall_with_perm(verse_rank, pc1, rng)

    # ---- Procrustes (uses both axes simultaneously) ----
    procrustes_res = procrustes_test(verse_rank, coords, rng)

    # ---- Mantel test (handles non-monotonic spatial coherence) ----
    mantel_res = mantel_test(verse_rank, coords, rng)

    # ---- Constrained-permutation null (skipped for prose corpora) ----
    constrained_res = None
    if "syllable_count" in df.columns and not args.no_meter:
        shapes = df["syllable_count"].astype(int).tolist()
        constrained_res = constrained_perm_null(
            verse_rank, lons, shapes=shapes, rng=rng,
        )

    # ---- Bayes factors ----
    # If --no-meter, set all shapes equal so the meter model collapses to
    # uniform and BF(geog vs meter) becomes BF(geog vs uniform). The
    # bayes_factors function returns both ratios; downstream consumers can
    # ignore BF(geog vs meter) when meter doesn't apply.
    if args.no_meter or "syllable_count" not in df.columns:
        shapes = [1] * n
    else:
        shapes = df["syllable_count"].astype(int).tolist()
    bf_res = bayes_factors(verse_rank, lons, shapes=shapes)
    bf_res["meter_constraint_applies"] = not args.no_meter and "syllable_count" in df.columns

    out = {
        "corpus_csv": args.corpus_csv,
        "n_items": n,
        "metadata": {
            "seed": SEED,
            "n_perm": N_PERM,
            "scipy_kendalltau": "scipy.stats.kendalltau (tau-b)",
        },
        "kendall_axis": {
            "longitude": kendall_lon,
            "latitude": kendall_lat,
            "pc1": kendall_pc1,
        },
        "procrustes": procrustes_res,
        "mantel": mantel_res,
        "constrained_perm_null": constrained_res,
        "bayes_factors": bf_res,
    }

    Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_json).write_text(json.dumps(out, indent=2))

    # ---- Human-readable summary ----
    print("=" * 64)
    print(f"Corpus: {args.corpus_csv}  (n = {n})")
    print("=" * 64)
    print(f"\nKendall τ vs longitude: τ = {kendall_lon['tau']:+.4f}   "
          f"p = {kendall_lon['p_two_sided']:.4f}")
    print(f"Kendall τ vs latitude:  τ = {kendall_lat['tau']:+.4f}   "
          f"p = {kendall_lat['p_two_sided']:.4f}")
    print(f"Kendall τ vs PC1:       τ = {kendall_pc1['tau']:+.4f}   "
          f"p = {kendall_pc1['p_two_sided']:.4f}")

    print(f"\nProcrustes: m² = {procrustes_res['m2']:.4f}   "
          f"p (one-sided) = {procrustes_res['p_one_sided']:.4f}")

    print(f"\nMantel: r = {mantel_res['mantel_r']:+.4f}   "
          f"p (one-sided) = {mantel_res['p_one_sided']:.4f}")

    if constrained_res is not None:
        print(f"\nConstrained-perm null (meter-admissible):")
        print(f"  N admissible = {constrained_res['n_admissible']}   "
              f"p_constrained = {constrained_res['p_constrained']:.4f}")
    else:
        print(f"\nConstrained-perm null: skipped (no syllable_count column).")

    print(f"\nBayes factors  (σ₀ orientation: {bf_res['sigma0_orientation']}, "
          f"Kendall dist = {bf_res['kendall_distance_observed']}/{n*(n-1)//2})")
    print(f"  BF(geog vs uniform) range: "
          f"{min(bf_res['BF_geog_vs_uniform'].values()):.2e} → "
          f"{max(bf_res['BF_geog_vs_uniform'].values()):.2e}")
    print(f"  BF(geog vs meter) range:   "
          f"{min(bf_res['BF_geog_vs_meter'].values()):.2f} → "
          f"{max(bf_res['BF_geog_vs_meter'].values()):.2f}")

    print(f"\nWrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
