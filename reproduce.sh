#!/usr/bin/env bash
# =====================================================================
# reproduce.sh — regenerate every numerical result in the paper from
# scratch, then verify each one against the paper's stated values.
#
# Runs fully offline: river coordinates come from the cached OSM
# Overpass responses in data/osm_cache/. The only step that needs an
# external download is the MERIT-Hydro re-derivation
# (scripts/fetch_river_coords_merit.py) — that is NOT run here; the
# resulting data/river_coordinates_merit.csv is provided, and the
# MERIT sensitivity check below consumes it. See docs/merit-hydro.md
# to regenerate that file from the raw NASA-SRTM tiles.
#
# Usage:  ./reproduce.sh
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/8] River coordinates for RV 10.75.5 (from cached OSM responses)"
python3 scripts/fetch_river_coords.py

echo
echo "[2/8] RV 10.75.5 analysis  (Kendall, Procrustes, Mantel,"
echo "      constrained-permutation null, Bayes factors)"
python3 scripts/analyze_nadistuti.py

echo
echo "[3/8] Per-corpus analyses  (Mahajanapadas, Avesta, Bisutun, Genesis 10)"
python3 scripts/analyze_corpus.py corpora/mahajanapadas/janapadas.csv \
        results/per_corpus/mahajanapadas.json --no-meter
python3 scripts/analyze_corpus.py corpora/avesta_fargard1/lands.csv \
        results/per_corpus/avesta_fargard1.json --no-meter
python3 scripts/analyze_corpus.py corpora/bisutun/lands.csv \
        results/per_corpus/bisutun.json --no-meter
python3 scripts/analyze_corpus.py corpora/genesis10/nations.csv \
        results/per_corpus/genesis10.json --no-meter

echo
echo "[4/8] Genesis 10 within-subsection analysis  (Japheth / Ham / Shem)"
python3 scripts/analyze_genesis_subsections.py

echo
echo "[5/8] MERIT-Hydro sensitivity check  (NASA-SRTM-derived centroids)"
python3 scripts/analyze_merit_sensitivity.py

echo
echo "[6/8] Figure 1  (RV null distribution)  +  Figure 2  (corpus comparison)"
python3 scripts/make_figure.py
python3 scripts/make_comparative_figure.py

echo
echo "[7/8] Sensitivity-summary table fragment"
python3 scripts/make_table.py

echo
echo "[8/8] Verifying every paper claim against the regenerated results"
python3 verify_claims.py

echo
echo "Done. Every figure, table, and number in the paper has been"
echo "regenerated and cross-checked."
