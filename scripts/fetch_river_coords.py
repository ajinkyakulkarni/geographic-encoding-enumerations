#!/usr/bin/env python3
"""
Fetch centroid coordinates for the ten rivers of Ṛgveda 10.75.5 via the
OpenStreetMap Overpass API.

Verse order follows the standard text of the Aufrecht edition, confirmed
against van Nooten & Holland (1994):

    imaṃ me gaṅge yamune sarasvati śutudri stomaṃ sacatā paruṣṇyā |
    asiknyā marudvṛdhe vitastayārjīkīye śṛṇuhyā suṣomayā ||

Outputs:
    data/river_coordinates.csv   — primary identifications (Witzel 1995)
    data/osm_cache/<slug>.json   — raw Overpass responses, for offline reruns

Identifications follow Witzel (1995) and Macdonell & Keith (1912).
The most-contested identification (Marudvṛdhā) is documented inline and
exercised in the sensitivity analysis.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Bounding box covering the Punjab + Ganges plain + upper Indus.
# (south, west, north, east). Generous: includes the full Ganges course
# down to the Sundarbans (≈22 °N) and the Indus mouth (≈24 °N, 67 °E).
REGION_BBOX = (22.0, 65.0, 37.0, 90.0)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "osm_cache"
OUT_CSV = DATA_DIR / "river_coordinates.csv"


@dataclass
class RiverSpec:
    position: int                  # verse position (1..10)
    vedic_name: str                # IAST transliteration
    identification: str            # modern identified river
    osm_names: list[str] | None    # candidate OSM names to try (regex-joined)
    hand_lon: float | None = None  # used when osm_names is None
    hand_lat: float | None = None
    note: str = ""


# Ten rivers in correct verse order (positions 6, 7 corrected from draft).
# OSM name candidates are joined into a single case-insensitive regex against
# the `name` and `name:en` tags. Multiple candidates handle the common
# pattern where OSM uses a vernacular spelling as primary name and the
# English form as `name:en` — or vice versa.
RIVERS: list[RiverSpec] = [
    RiverSpec(1,  "Gaṅgā",      "Ganges",   ["Ganga", "Ganges"]),
    RiverSpec(2,  "Yamunā",     "Yamuna",   ["Yamuna", "Jamuna"]),
    RiverSpec(3,  "Sarasvatī",  "Ghaggar-Hakra",
              ["Ghaggar", "Ghaggar-Hakra", "Ghaggar River"]),
    RiverSpec(4,  "Śutudrī",    "Sutlej",   ["Sutlej", "Satluj"]),
    RiverSpec(5,  "Paruṣṇī",    "Ravi",     ["Ravi", "Ravi River"]),
    RiverSpec(6,  "Asiknī",     "Chenab",   ["Chenab"]),
    RiverSpec(7,  "Marudvṛdhā", "uncertain (Maruvardhā / Chenab-area tributary)",
              None,
              hand_lon=73.10, hand_lat=32.20,
              note="Most contested ID. Hand-curated to the Chenab-Jhelum doab "
                   "(Trimmu junction region) as a conservative placement "
                   "between Asiknī (pos. 6) and Vitastā (pos. 8). Witzel 1995 "
                   "treats the referent as uncertain. Sensitivity analysis "
                   "(i) tests alternative IDs and a drop-position-7 variant."),
    RiverSpec(8,  "Vitastā",    "Jhelum",   ["Jhelum"]),
    RiverSpec(9,  "Ārjīkīyā",   "Haro (Witzel)", ["Haro", "Haro River"]),
    RiverSpec(10, "Suṣomā",     "Soan (Sohan / Sawan)",
              ["Sawan", "Suwaan", "Swaan"]),
]
# Note: OSM tags the Pakistani Soan/Sohan as "Sawan River" (Punjabi
# دریائے سواں). The exact-match "Soan" only finds an unrelated minor
# Indian Punjab stream.


def overpass_query(names: list[str]) -> dict:
    """Query Overpass for ways+relations matching any of `names` (case-insensitive)
    against either the `name` or `name:en` tag, restricted to waterway=river.
    """
    s, w, n, e = REGION_BBOX
    # Allow optional " River" / " river" suffix (very common in OSM tagging).
    pattern = "^(" + "|".join(
        nm.replace('"', '\\"') for nm in names
    ) + ")( [Rr]iver)?$"
    query = f"""
[out:json][timeout:120][bbox:{s},{w},{n},{e}];
(
  way[waterway=river]["name"~"{pattern}",i];
  way[waterway=river]["name:en"~"{pattern}",i];
  relation[waterway=river]["name"~"{pattern}",i];
  relation[waterway=river]["name:en"~"{pattern}",i];
);
out geom;
"""
    headers = {
        "User-Agent": "nadistuti-paper-build/0.1 "
                      "(academic research; contact: ajinkya.kulkarni@uah.edu)"
    }
    r = requests.post(OVERPASS_URL, data={"data": query},
                      headers=headers, timeout=180)
    r.raise_for_status()
    return r.json()


def extract_coords(payload: dict) -> list[tuple[float, float]]:
    """Pull (lon, lat) pairs from all geometry nodes in the payload."""
    coords: list[tuple[float, float]] = []
    for el in payload.get("elements", []):
        # Ways carry a top-level `geometry` list.
        if "geometry" in el:
            for pt in el["geometry"]:
                coords.append((pt["lon"], pt["lat"]))
        # Relations carry per-member `geometry` lists.
        for m in el.get("members", []):
            if "geometry" in m:
                for pt in m["geometry"]:
                    coords.append((pt["lon"], pt["lat"]))
    return coords


def centroid(coords: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(coords)
    if n == 0:
        raise ValueError("no coordinates to centroid")
    lon = sum(c[0] for c in coords) / n
    lat = sum(c[1] for c in coords) / n
    return lon, lat


def slugify(name: str) -> str:
    return (name.lower()
            .replace(" ", "_")
            .replace("(", "").replace(")", "")
            .replace("/", "_"))


def fetch_one(spec: RiverSpec) -> dict:
    """Fetch + centroid one river. Returns row dict for CSV."""
    if spec.osm_names is None:
        # Hand-curated coordinate (e.g. Marudvṛdhā).
        return {
            "position": spec.position,
            "vedic_name": spec.vedic_name,
            "identified_river": spec.identification,
            "centroid_lon": spec.hand_lon,
            "centroid_lat": spec.hand_lat,
            "n_nodes": 0,
            "osm_id": "",
            "provenance": "hand-curated",
            "note": spec.note,
        }

    slug = slugify(spec.osm_names[0])
    cache_path = CACHE_DIR / f"{slug}.json"
    if cache_path.exists():
        print(f"  [cache] {spec.osm_names[0]}", file=sys.stderr)
        payload = json.loads(cache_path.read_text())
    else:
        print(f"  [fetch] {'|'.join(spec.osm_names)}", file=sys.stderr)
        payload = overpass_query(spec.osm_names)
        cache_path.write_text(json.dumps(payload))
        time.sleep(2)  # be polite to public Overpass instance

    coords = extract_coords(payload)
    if not coords:
        raise RuntimeError(
            f"Overpass returned no geometry for {spec.osm_names!r}. "
            f"Inspect {cache_path} and add a hand-curated fallback."
        )

    lon, lat = centroid(coords)
    osm_ids = sorted({str(el["id"]) for el in payload.get("elements", [])
                      if el.get("type") in ("way", "relation")})

    return {
        "position": spec.position,
        "vedic_name": spec.vedic_name,
        "identified_river": spec.identification,
        "centroid_lon": round(lon, 4),
        "centroid_lat": round(lat, 4),
        "n_nodes": len(coords),
        "osm_id": ";".join(osm_ids),
        "provenance": f"OSM Overpass; name in [{', '.join(spec.osm_names)}]",
        "note": spec.note,
    }


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in RIVERS:
        try:
            row = fetch_one(spec)
        except Exception as exc:
            print(f"ERROR on position {spec.position} ({spec.vedic_name}): "
                  f"{exc}", file=sys.stderr)
            return 1
        rows.append(row)
        print(f"  pos {row['position']:2d} {row['vedic_name']:12s} "
              f"→ ({row['centroid_lon']}, {row['centroid_lat']}) "
              f"n={row['n_nodes']}", file=sys.stderr)

    fields = ["position", "vedic_name", "identified_river",
              "centroid_lon", "centroid_lat",
              "n_nodes", "osm_id", "provenance", "note"]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
