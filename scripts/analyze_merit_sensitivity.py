#!/usr/bin/env python3
"""
Sensitivity check: re-run the statistical pipeline using MERIT-Hydro-
derived centroids instead of OpenStreetMap polyline centroids, and
report whether the headline τ is robust.

Inputs:
  data/river_coordinates.csv         OSM-derived (committed)
  data/river_coordinates_merit.csv   MERIT-Hydro-derived (run
                                     fetch_river_coords_merit.py first)
  data/paleocourse_coords.csv        (kept for Sarasvatī paleo variant)

Outputs:
  results/per_corpus/rv_merit.json     results on MERIT centroids
  results/merit_sensitivity_table.txt  side-by-side OSM/MERIT table

It builds an input set using MERIT longitudes/latitudes for the nine
rivers MERIT provides, plus the OSM coordinate for Marudvṛdhā (which
is hand-curated and never had an OSM polyline), and runs the same
Kendall / Procrustes / Mantel battery used for the other corpora.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OSM_CSV = REPO_ROOT / "data" / "river_coordinates.csv"
MERIT_CSV = REPO_ROOT / "data" / "river_coordinates_merit.csv"
COMBINED_CSV = REPO_ROOT / "data" / "river_coordinates_merit_combined.csv"
OUT_JSON = REPO_ROOT / "results" / "per_corpus" / "rv_merit.json"
SUMMARY_TXT = REPO_ROOT / "results" / "merit_sensitivity_table.txt"


def main() -> int:
    if not MERIT_CSV.exists():
        sys.stderr.write(
            f"{MERIT_CSV} not found. Run scripts/fetch_river_coords_merit.py "
            f"first (and see docs/merit-hydro.md for the manual "
            f"download step).\n"
        )
        return 1

    osm = pd.read_csv(OSM_CSV).sort_values("position").reset_index(drop=True)
    merit = pd.read_csv(MERIT_CSV)
    # Build a combined CSV with MERIT coords where available, OSM coords
    # otherwise (e.g. Marudvṛdhā has no OSM polyline anyway).
    merit_lookup = {r["position"]: r for _, r in merit.iterrows()}
    rows = []
    for _, r in osm.iterrows():
        pos = int(r["position"])
        if pos in merit_lookup:
            m = merit_lookup[pos]
            lon, lat = m["merit_lon"], m["merit_lat"]
            source = "MERIT-Hydro (unweighted centroid)"
        else:
            lon, lat = r["centroid_lon"], r["centroid_lat"]
            source = "OSM (no MERIT for this position)"
        rows.append({
            "position": pos,
            "vedic_name": r["vedic_name"],
            "identified_river": r["identified_river"],
            "centroid_lon": lon,
            "centroid_lat": lat,
            "source": source,
        })
    df = pd.DataFrame(rows)
    df.to_csv(COMBINED_CSV, index=False)

    # Run the same Kendall / Procrustes / Mantel battery.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from methods import procrustes_test, mantel_test
    from scipy.stats import kendalltau

    verse_rank = df["position"].to_numpy()
    lons = df["centroid_lon"].to_numpy()
    lats = df["centroid_lat"].to_numpy()
    coords = np.column_stack([lons, lats])
    rng = np.random.default_rng(20260518)

    tau_lon, _ = kendalltau(verse_rank, lons)
    # Permutation p
    nperm = 10_000
    null = np.empty(nperm)
    for i in range(nperm):
        perm = rng.permutation(len(verse_rank))
        null[i], _ = kendalltau(perm, lons)
    p_lon = float(np.mean(np.abs(null) >= abs(tau_lon)))

    procrustes_res = procrustes_test(verse_rank, coords, rng)
    mantel_res = mantel_test(verse_rank, coords, rng)

    out = {
        "n_items": len(df),
        "kendall_lon": {"tau": float(tau_lon), "p_two_sided": p_lon, "n_perm": nperm},
        "procrustes": procrustes_res,
        "mantel": mantel_res,
        "source_csv": str(COMBINED_CSV.relative_to(REPO_ROOT)),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # Load OSM baseline for comparison.
    rv_osm = json.loads((REPO_ROOT / "results" / "results.json").read_text())
    osm_primary = rv_osm["primary"]
    osm_proc = rv_osm["procrustes"]
    osm_mantel = rv_osm["mantel"]

    lines = [
        "MERIT-Hydro vs. OSM sensitivity (RV 10.75.5)",
        "=" * 60,
        f"  Test               OSM baseline        MERIT-Hydro",
        f"  -------------------------------------------------",
        f"  Kendall τ vs lon   {osm_primary['tau']:+.4f} (p={osm_primary['p_two_sided']:.4f})   "
        f"{tau_lon:+.4f} (p={p_lon:.4f})",
        f"  Procrustes m²      {osm_proc['m2']:.4f} (p={osm_proc['p_one_sided']:.4f})   "
        f"{procrustes_res['m2']:.4f} (p={procrustes_res['p_one_sided']:.4f})",
        f"  Mantel r           {osm_mantel['mantel_r']:+.4f} (p={osm_mantel['p_one_sided']:.4f})   "
        f"{mantel_res['mantel_r']:+.4f} (p={mantel_res['p_one_sided']:.4f})",
        "",
        "Mean centroid shift (km): "
        + f"{pd.read_csv(MERIT_CSV)['distance_km'].mean():.1f} "
        + f"(max {pd.read_csv(MERIT_CSV)['distance_km'].max():.1f})",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {SUMMARY_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
