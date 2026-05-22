#!/usr/bin/env python3
"""
Derive river centroids for the Ṛgveda 10.75.5 rivers from MERIT-Hydro
upstream-drainage-area (upa) rasters. MERIT-Hydro is a global
hydrography product built on satellite elevation data, including
NASA's SRTM. This supports the sensitivity check that the headline
result is robust to switching the coordinate source from OpenStreetMap
polylines to an independent satellite-derived dataset.

For each river, the script:
  1. Reads the upa GeoTIFF tiles covering the bounding box of the
     OSM-derived polyline (with a 0.5° buffer).
  2. Selects pixels whose upa exceeds a per-river river-network
     threshold (≈1000 km² for the major Punjab rivers; smaller for
     the minor tributaries).
  3. Of those, keeps only pixels within 0.2° of an OSM polyline node,
     which disambiguates the target river from its neighbours.
  4. Takes the *unweighted* centroid (mean of the kept pixels). Not
     upa-weighted: weighting by drainage area would pull the centroid
     of a long river downstream toward its high-accumulation lower
     reaches, which is not the "where is this river" summary the test
     needs.

Output: data/river_coordinates_merit.csv with columns
  position, vedic_name, identified_river, merit_lon, merit_lat,
  osm_lon, osm_lat, distance_km, n_pixels, threshold_km2

The committed data/river_coordinates_merit.csv lets the downstream
sensitivity check run with no download. Regenerating it from the raw
tiles needs the MERIT-Hydro registration and download described in
docs/merit-hydro.md; in brief:

  1. Register at https://global-hydrodynamics.github.io/MERIT_Hydro/
     and accept the CC-BY-NC 4.0 licence.
  2. Download the upstream-drainage-area (upa) tar packages covering
     0°–60°N, 60°–90°E and extract them into data/merit_hydro/ (tiles
     unpack into per-package subdirectories).
  3. Run:  python3 scripts/fetch_river_coords_merit.py

License: MERIT-Hydro is CC-BY-NC 4.0 (Yamazaki et al. 2019). The
derived CSV inherits the non-commercial restriction; cite Yamazaki
et al. 2019 in any downstream use.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.windows import from_bounds
except ImportError:
    sys.stderr.write(
        "rasterio is required for this script. Install with:\n"
        "    pip install rasterio\n"
    )
    raise SystemExit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
MERIT_DIR = REPO_ROOT / "data" / "merit_hydro"
OSM_CACHE = REPO_ROOT / "data" / "osm_cache"
OUT_CSV = REPO_ROOT / "data" / "river_coordinates_merit.csv"

# Per-river upa threshold (km²) and the OSM-cache file to disambiguate
# from neighboring rivers. Larger threshold = require the pixel to lie
# on a major channel; smaller = include tributaries.
RIVER_SPECS = [
    # (position, vedic_name, identified, osm_cache_basename, upa_threshold_km2)
    (1,  "Gaṅgā",        "Ganges",          "ganga.json",   10_000),
    (2,  "Yamunā",       "Yamuna",          "yamuna.json",   5_000),
    (3,  "Sarasvatī",    "Ghaggar-Hakra",   "ghaggar.json",  1_000),
    (4,  "Śutudrī",      "Sutlej",          "sutlej.json",   2_000),
    (5,  "Paruṣṇī",      "Ravi",            "ravi.json",     1_000),
    (6,  "Asiknī",       "Chenab",          "chenab.json",   3_000),
    # Marudvṛdhā: hand-curated in OSM analysis; we skip MERIT here.
    (8,  "Vitastā",      "Jhelum",          "jhelum.json",   2_000),
    (9,  "Ārjīkīyā",     "Haro",   "haro.json",     500),
    (10, "Suṣomā",       "Soan (Sawan)",    "sawan.json",    500),
]


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance between two points in km."""
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def find_upa_tiles_for_bbox(min_lon, min_lat, max_lon, max_lat):
    """Yield paths to MERIT-Hydro upa tiles intersecting the bbox.

    MERIT-Hydro tiles are 5°×5° GeoTIFFs named n{lat}e{lon}_upa.tif
    (where (lat, lon) is the SW corner rounded down to a multiple of 5),
    bundled in 30°×30° tar packages that extract into subdirectories
    like upa_n30e060/ and upa_n00e060/. We search those subdirectories
    recursively.
    """
    lat0 = (int(math.floor(min_lat)) // 5) * 5
    lat1 = (int(math.floor(max_lat)) // 5) * 5
    lon0 = (int(math.floor(min_lon)) // 5) * 5
    lon1 = (int(math.floor(max_lon)) // 5) * 5
    for la in range(lat0, lat1 + 1, 5):
        for lo in range(lon0, lon1 + 1, 5):
            ns = "n" if la >= 0 else "s"
            ew = "e" if lo >= 0 else "w"
            fname = f"{ns}{abs(la):02d}{ew}{abs(lo):03d}_upa.tif"
            # Search recursively: tiles live in upa_n??e??? subdirs.
            matches = list(MERIT_DIR.rglob(fname))
            for m in matches:
                yield m
                break  # one match per (la, lo) is enough


def osm_polyline_nodes(cache_file: Path) -> list[tuple[float, float]]:
    """Extract all (lon, lat) nodes from a cached OSM Overpass response."""
    if not cache_file.exists():
        return []
    payload = json.loads(cache_file.read_text())
    coords = []
    for el in payload.get("elements", []):
        if "geometry" in el:
            for pt in el["geometry"]:
                coords.append((pt["lon"], pt["lat"]))
        for m in el.get("members", []):
            if "geometry" in m:
                for pt in m["geometry"]:
                    coords.append((pt["lon"], pt["lat"]))
    return coords


def derive_centroid(spec, threshold_km2):
    position, vedic, identified, cache_name, _ = spec
    osm_nodes = osm_polyline_nodes(OSM_CACHE / cache_name)
    if not osm_nodes:
        sys.stderr.write(f"  pos {position}: no OSM nodes cached; skipping\n")
        return None

    lons = [c[0] for c in osm_nodes]
    lats = [c[1] for c in osm_nodes]
    bbox = (min(lons) - 0.5, min(lats) - 0.5, max(lons) + 0.5, max(lats) + 0.5)
    tiles = list(find_upa_tiles_for_bbox(*bbox))
    if not tiles:
        sys.stderr.write(
            f"  pos {position}: no MERIT tiles found for bbox {bbox}; "
            f"have you extracted the tars into {MERIT_DIR}?\n"
        )
        return None

    # Build a KD-tree of OSM nodes for fast proximity queries.
    osm_arr = np.array(osm_nodes)

    # Collect the (lon, lat) of every MERIT pixel that (a) exceeds the
    # upa river-threshold and (b) lies within 0.2° of any OSM polyline
    # node. We compute the *unweighted* centroid — the geometric mean of
    # qualifying pixels — to mirror what OSM does with its polyline nodes,
    # and to avoid the downstream bias that upa-weighting would introduce
    # for long rivers like the Ganga (whose lower reaches dominate the
    # drainage area and would skew the centroid toward the delta).
    matched_lons: list[float] = []
    matched_lats: list[float] = []

    for tile_path in tiles:
        with rasterio.open(tile_path) as src:
            arr = src.read(1)  # km²
            transform = src.transform
            cols = np.arange(arr.shape[1])
            rows = np.arange(arr.shape[0])
            tile_lons = transform.c + (cols + 0.5) * transform.a
            tile_lats = transform.f + (rows + 0.5) * transform.e
            mask = arr >= threshold_km2
            if not mask.any():
                continue
            row_idx, col_idx = np.where(mask)
            px_lons = tile_lons[col_idx]
            px_lats = tile_lats[row_idx]
            # Vectorized proximity: for each MERIT pixel, find min squared
            # distance to OSM polyline. Done in chunks to avoid OOM on
            # the Ganga (which has ~15K OSM nodes × ~50K river pixels).
            CHUNK = 2000
            for k in range(0, len(px_lons), CHUNK):
                chk_lons = px_lons[k:k + CHUNK]
                chk_lats = px_lats[k:k + CHUNK]
                # pairwise distance: (chunk, osm) → take min over osm
                dlon = chk_lons[:, None] - osm_arr[None, :, 0]
                dlat = chk_lats[:, None] - osm_arr[None, :, 1]
                d2_min = (dlon ** 2 + dlat ** 2).min(axis=1)
                close_mask = d2_min < 0.2 ** 2
                matched_lons.extend(chk_lons[close_mask].tolist())
                matched_lats.extend(chk_lats[close_mask].tolist())

    n_pixels = len(matched_lons)
    if n_pixels == 0:
        sys.stderr.write(
            f"  pos {position}: no MERIT pixels matched (threshold "
            f"{threshold_km2} km², OSM-proximity 0.2°)\n"
        )
        return None

    merit_lon = float(np.mean(matched_lons))
    merit_lat = float(np.mean(matched_lats))

    # Compute distance to OSM centroid (mean of OSM nodes).
    osm_lon = float(np.mean(osm_arr[:, 0]))
    osm_lat = float(np.mean(osm_arr[:, 1]))
    dist_km = haversine_km(osm_lon, osm_lat, merit_lon, merit_lat)

    return {
        "position": position,
        "vedic_name": vedic,
        "identified_river": identified,
        "merit_lon": round(merit_lon, 4),
        "merit_lat": round(merit_lat, 4),
        "osm_lon": round(osm_lon, 4),
        "osm_lat": round(osm_lat, 4),
        "distance_km": round(dist_km, 2),
        "n_pixels": n_pixels,
        "threshold_km2": threshold_km2,
    }


def main() -> int:
    if not MERIT_DIR.is_dir() or not any(MERIT_DIR.rglob("*_upa.tif")):
        sys.stderr.write(
            f"No MERIT-Hydro upa tiles found under {MERIT_DIR}.\n"
            f"See docs/merit-hydro.md for download instructions.\n"
        )
        return 1

    rows = []
    for spec in RIVER_SPECS:
        print(f"  pos {spec[0]:2d} {spec[1]} ({spec[2]}) ...", file=sys.stderr)
        row = derive_centroid(spec, spec[4])
        if row is None:
            continue
        rows.append(row)
        print(
            f"    MERIT ({row['merit_lon']}, {row['merit_lat']}) "
            f"vs OSM ({row['osm_lon']}, {row['osm_lat']})  "
            f"Δ = {row['distance_km']} km   n_pixels = {row['n_pixels']}",
            file=sys.stderr,
        )

    if not rows:
        sys.stderr.write("No rows produced.\n")
        return 1

    fields = ["position", "vedic_name", "identified_river",
              "merit_lon", "merit_lat", "osm_lon", "osm_lat",
              "distance_km", "n_pixels", "threshold_km2"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\nWrote {OUT_CSV} ({len(rows)} rivers).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
