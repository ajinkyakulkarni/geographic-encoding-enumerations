#!/usr/bin/env python3
"""
Test whether the order of rivers in Ṛgveda 10.75.5 is independent of their
geographic position. Reports Kendall's τ between verse-order rank and
longitude (primary) plus four sensitivity analyses, writing every reported
number to results/results.json.

Reproducibility
---------------
    Seed:               20260518     (today's date, fixed in paper)
    Permutation null:   10,000 draws (primary test)
    Perturbation runs:  1,000        (sensitivity iv, σ = 0.5°)

Inputs
------
    data/river_coordinates.csv      modern centroids, 10 rows
    data/paleocourse_coords.csv     paleo-Sarasvatī, paleo-Sutlej

Outputs
-------
    results/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# Allow `from methods import ...` when run from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import (
    constrained_perm_null,
    mantel_test,
    procrustes_test,
    profile_likelihood_ratios,
)

SEED = 20260518
N_PERM = 10_000
N_PERTURB = 1_000
PERTURB_SIGMA_DEG = 0.5

REPO_ROOT = Path(__file__).resolve().parent.parent
COORDS_CSV = REPO_ROOT / "data" / "river_coordinates.csv"
PALEO_CSV = REPO_ROOT / "data" / "paleocourse_coords.csv"
RESULTS_JSON = REPO_ROOT / "results" / "results.json"


def tau_and_perm_p(verse_rank: np.ndarray, geo_value: np.ndarray,
                   rng: np.random.Generator,
                   n_perm: int = N_PERM) -> dict:
    """Kendall τ between two arrays of equal length, with two p-values.

    `p_two_sided` is the permutation p-value: shuffle the verse-order
    labels, hold the geography fixed, and take the share of permuted
    |τ| ≥ |observed τ|. `p_exact` is the exact two-sided Kendall-τ
    p-value returned by scipy, available here because every RV variant
    has n ≤ 10 with no ties; for n this small scipy enumerates the exact
    null rather than approximating it. The two agree up to the
    resolution floor of the permutation grid.
    """
    tau_obs, p_exact = kendalltau(verse_rank, geo_value)
    n = len(verse_rank)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        t, _ = kendalltau(perm, geo_value)
        null[i] = t
    p_two_sided = float(np.mean(np.abs(null) >= abs(tau_obs)))
    return {
        "tau": float(tau_obs),
        "p_two_sided": p_two_sided,
        "p_exact": float(p_exact),
        "n_perm": n_perm,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "null_p025": float(np.quantile(null, 0.025)),
        "null_p975": float(np.quantile(null, 0.975)),
        "null_samples": null,  # caller may drop before serializing
    }


def pc1(coords: np.ndarray) -> np.ndarray:
    """Project (lon, lat) points onto their first principal component.

    Centers the data, computes the 2×2 covariance matrix, takes the
    eigenvector with the largest eigenvalue, projects each point onto it.
    Result is rotation-invariant in the sense that PC1 picks the axis of
    maximum variance regardless of cardinal direction.
    """
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    pc1_axis = eigvecs[:, -1]               # largest eigenvalue
    return centered @ pc1_axis


def main() -> int:
    df = pd.read_csv(COORDS_CSV).sort_values("position").reset_index(drop=True)
    assert len(df) == 10, f"expected 10 rivers, found {len(df)}"
    paleo = pd.read_csv(PALEO_CSV)

    verse_rank = df["position"].to_numpy()
    lons = df["centroid_lon"].to_numpy()
    lats = df["centroid_lat"].to_numpy()

    rng = np.random.default_rng(SEED)

    # ---------- Primary test: verse order vs longitude ----------
    primary = tau_and_perm_p(verse_rank, lons, rng)
    null_samples = primary.pop("null_samples")
    primary["seed"] = SEED

    # ---------- Sensitivity (iii): alternative geographic axes ----------
    lat_res = tau_and_perm_p(verse_rank, lats, rng)
    lat_res.pop("null_samples")
    pc1_proj = pc1(np.column_stack([lons, lats]))
    pc1_res = tau_and_perm_p(verse_rank, pc1_proj, rng)
    pc1_res.pop("null_samples")
    sens_axis = {
        "latitude": {"tau": lat_res["tau"],
                     "p_two_sided": lat_res["p_two_sided"],
                     "p_exact": lat_res["p_exact"]},
        "pc1": {"tau": pc1_res["tau"],
                "p_two_sided": pc1_res["p_two_sided"],
                "p_exact": pc1_res["p_exact"]},
    }

    # ---------- Sensitivity (ii): paleocourse substitution ----------
    paleo_lons = lons.copy()
    paleo_lats = lats.copy()
    sub_log = []
    for _, prow in paleo.iterrows():
        match = df.index[df["vedic_name"] == prow["vedic_name"]]
        if len(match) == 0:
            continue
        idx = match[0]
        paleo_lons[idx] = prow["centroid_lon"]
        paleo_lats[idx] = prow["centroid_lat"]
        sub_log.append({
            "vedic_name": prow["vedic_name"],
            "modern": [float(lons[idx]), float(lats[idx])],
            "paleo": [float(prow["centroid_lon"]), float(prow["centroid_lat"])],
        })
    paleo_res = tau_and_perm_p(verse_rank, paleo_lons, rng)
    paleo_res.pop("null_samples")
    sens_paleo = {
        "tau": paleo_res["tau"],
        "p_two_sided": paleo_res["p_two_sided"],
        "p_exact": paleo_res["p_exact"],
        "substitutions": sub_log,
    }

    # ---------- Sensitivity (i): alternative river IDs ----------
    # Candidate alternatives for Ārjīkīyā (pos 9) and Marudvṛdhā (pos 7),
    # both contested identifications; we test reasonable alternatives plus
    # a drop-position-7 variant (n=9) for Marudvṛdhā.
    arjikiya_alts = {
        "Haro (primary)":        lons[8],   # pos 9, idx 8
        "Soan (eastern alt)":    76.0,      # eastern alternative longitude
        "upper Indus tributary": 71.5,      # westward alternative
    }
    marudvridha_alts = {
        "Chenab–Jhelum doab (primary)": lons[6],   # pos 7, idx 6
        "eastern alternative":   74.0,      # slightly east of the doab
    }

    id_results = []
    for arj_label, arj_lon in arjikiya_alts.items():
        for mar_label, mar_lon in marudvridha_alts.items():
            v = lons.copy()
            v[8] = arj_lon
            v[6] = mar_lon
            r = tau_and_perm_p(verse_rank, v, rng)
            r.pop("null_samples")
            id_results.append({
                "arjikiya": arj_label,
                "marudvridha": mar_label,
                "tau": r["tau"],
                "p_two_sided": r["p_two_sided"],
                "p_exact": r["p_exact"],
            })

    # Drop-position-7 variant (n=9): remove the contested Marudvṛdhā.
    drop_mask = verse_rank != 7
    # Renumber verse ranks 1..9 after the drop to preserve ordinality.
    vr9 = np.argsort(np.argsort(verse_rank[drop_mask])) + 1
    drop_res = tau_and_perm_p(vr9, lons[drop_mask], rng)
    drop_res.pop("null_samples")
    id_results.append({
        "arjikiya": "Haro (primary)",
        "marudvridha": "DROPPED (n=9)",
        "tau": drop_res["tau"],
        "p_two_sided": drop_res["p_two_sided"],
        "p_exact": drop_res["p_exact"],
    })

    # ---------- Sensitivity (iv): coordinate perturbation ----------
    taus = np.empty(N_PERTURB)
    for i in range(N_PERTURB):
        noise = rng.normal(0.0, PERTURB_SIGMA_DEG, size=lons.shape)
        t, _ = kendalltau(verse_rank, lons + noise)
        taus[i] = t
    sens_pert = {
        "n_iter": N_PERTURB,
        "sigma_deg": PERTURB_SIGMA_DEG,
        "tau_p025": float(np.quantile(taus, 0.025)),
        "tau_p500": float(np.quantile(taus, 0.500)),
        "tau_p975": float(np.quantile(taus, 0.975)),
        "tau_mean": float(taus.mean()),
        "tau_std":  float(taus.std(ddof=1)),
    }

    # ---------- Procrustes, Mantel, constrained null, likelihood ratios ----------
    coords = np.column_stack([lons, lats])
    procrustes_res = procrustes_test(verse_rank, coords, rng)
    mantel_res = mantel_test(verse_rank, coords, rng)
    constrained_res = constrained_perm_null(verse_rank, lons)
    lr_res = profile_likelihood_ratios(verse_rank, lons)

    # ---------- Assemble ----------
    out = {
        "metadata": {
            "seed": SEED,
            "n_perm_primary": N_PERM,
            "n_perturbation": N_PERTURB,
            "perturbation_sigma_deg": PERTURB_SIGMA_DEG,
            "scipy_kendalltau": "scipy.stats.kendalltau (tau-b)",
        },
        "primary": primary,
        "sensitivity_axis": sens_axis,
        "sensitivity_paleo": sens_paleo,
        "sensitivity_ids": id_results,
        "sensitivity_perturbation": sens_pert,
        "procrustes": procrustes_res,
        "mantel": mantel_res,
        "constrained_perm_null": constrained_res,
        "likelihood_ratios": lr_res,
        "null_distribution_samples": null_samples.tolist(),
    }

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(out, indent=2))

    # ---------- Human-readable summary ----------
    print("=" * 60)
    print("Nadīstuti-sūkta rank-correlation test  (RV 10.75.5)")
    print("=" * 60)
    print(f"\nPrimary: verse order vs longitude")
    print(f"  τ = {primary['tau']:+.4f}   p (10K perm) = {primary['p_two_sided']:.4f}"
          f"   p (exact) = {primary['p_exact']:.3e}")
    print(f"  null mean = {primary['null_mean']:+.4f}   null std = {primary['null_std']:.4f}")
    print(f"  null 95% CI = [{primary['null_p025']:+.4f}, {primary['null_p975']:+.4f}]")

    print(f"\nSensitivity (iii) — alternative axes")
    print(f"  latitude:  τ = {sens_axis['latitude']['tau']:+.4f}  p = {sens_axis['latitude']['p_two_sided']:.4f}")
    print(f"  PC1:       τ = {sens_axis['pc1']['tau']:+.4f}  p = {sens_axis['pc1']['p_two_sided']:.4f}")

    print(f"\nSensitivity (ii) — paleocourse substitution")
    print(f"  τ = {sens_paleo['tau']:+.4f}   p = {sens_paleo['p_two_sided']:.4f}")

    print(f"\nSensitivity (i) — alternative IDs ({len(id_results)} variants)")
    for r in id_results:
        print(f"  arj={r['arjikiya']:30s} mar={r['marudvridha']:30s}  τ={r['tau']:+.4f}  p={r['p_two_sided']:.4f}")

    print(f"\nSensitivity (iv) — coordinate perturbation (σ=0.5°, n=1000)")
    print(f"  τ 95% CI = [{sens_pert['tau_p025']:+.4f}, {sens_pert['tau_p975']:+.4f}]")
    print(f"  τ median = {sens_pert['tau_p500']:+.4f}")

    print()
    print("=" * 60)
    print("Procrustes / Mantel / constrained-permutation / likelihood ratios")
    print("=" * 60)
    print(f"\nProcrustes / PROTEST")
    print(f"  m² = {procrustes_res['m2']:.4f}   p (one-sided) = {procrustes_res['p_one_sided']:.4f}")
    print(f"  null mean = {procrustes_res['null_mean']:.4f}   std = {procrustes_res['null_std']:.4f}")

    print(f"\nMantel (distance-matrix correlation, great-circle)")
    print(f"  r = {mantel_res['mantel_r']:+.4f}   p (one-sided) = {mantel_res['p_one_sided']:.4f}")

    print(f"\nConstrained permutation null (meter-admissible orderings only)")
    print(f"  observed τ = {constrained_res['tau_obs']:+.4f}")
    print(f"  N admissible = {constrained_res['n_admissible']}   N |τ|≥obs = {constrained_res['n_extreme_geq_observed']}")
    print(f"  p_constrained = {constrained_res['p_constrained']:.4f}")

    print(f"\nProfile likelihood ratios  (σ₀ orientation: {lr_res['sigma0_orientation']})")
    print(f"  Kendall distance to optimal sort = {lr_res['kendall_distance_observed']} / 45")
    print(f"  log P(observed) under each model:")
    print(f"    uniform:  {lr_res['log_p_uniform']:.3f}")
    print(f"    meter:    {lr_res['log_p_meter']:.3f}  (N = {lr_res['n_admissible']})")
    for k, v in lr_res['log_p_geog'].items():
        print(f"    geog({k}): {v:.3f}")
    print(f"  LR(meter vs uniform)  = {lr_res['LR_meter_vs_uniform']:.1f}")
    print(f"  LR(geog vs uniform):")
    for k, v in lr_res['LR_geog_vs_uniform'].items():
        print(f"    {k}: {v:.2e}")
    print(f"  LR(geog vs meter):")
    for k, v in lr_res['LR_geog_vs_meter'].items():
        print(f"    {k}: {v:.2f}")

    print(f"\nWrote {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
