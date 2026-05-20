# Geographic encoding in premodern enumerations

Replication materials for the paper **_Geographic encoding in premodern
enumerations: a five-corpus quantitative test_** (Ajinkya Kulkarni, 2026).

Every number, figure, and table in the paper is regenerated from raw
inputs by the scripts here, and every numerical claim is
cross-checked automatically against the paper's stated values.

## The result in one paragraph

Five canonical enumerations are tested for geographic ordering — the
ten rivers of Ṛgveda 10.75.5, the sixteen Mahājanapadas of Anguttara
Nikāya I.213, the sixteen lands of Vendīdād Fargard 1, the twenty-three
subject lands of Darius I's Bīsutūn inscription, and the Table of
Nations in Genesis 10. The two Indic lists are strongly ordered
east-to-west (Kendall τ = −0.911 and −0.783, both *p* < 10⁻⁴); Bīsutūn
is ordered loosely by satrapal bloc; Genesis 10 is ordered within its
three Noachic sub-sections but not across them; Vendīdād Fargard 1
shows no geographic order at all. Genre predicts the outcome better
than language family or date.

## Reproduce everything in one command

```bash
pip install -r requirements.txt
./reproduce.sh
```

`reproduce.sh` runs the full pipeline offline — river coordinates come
from the cached OSM responses in `data/osm_cache/` — and finishes by
running `verify_claims.py`, which checks all 35 numerical claims in the
paper against the freshly regenerated results and prints a PASS/FAIL
table. It exits non-zero if any number fails to reproduce.

To check the committed results without rerunning the pipeline:

```bash
python3 verify_claims.py
```

## How each claim is verified

`verify_claims.py` is the cross-verification harness. Each row pins one
number in the paper to (a) its location — abstract, Table 1, Section
3.4, or Appendix A — and (b) the results file and field that produces
it. Running it prints, for every claim, the value stated in the paper
beside the value just computed, and whether they match. The current
state: **35/35 claims reproduce.**

The chain for any single number is: a raw input file →
an analysis script → a JSON results file → a row in `verify_claims.py`.
For example, the headline τ = −0.911:

| Step | Artifact |
|---|---|
| Raw input | `data/river_coordinates.csv` (from `data/osm_cache/`) |
| Script | `scripts/analyze_nadistuti.py` |
| Output | `results/results.json` → `primary.tau` |
| Checked by | `verify_claims.py`, row "RV 10.75.5 Kendall tau vs longitude" |

## Pipeline

| Step | Script | Produces |
|---|---|---|
| 1 | `scripts/fetch_river_coords.py` | `data/river_coordinates.csv` from cached OSM Overpass responses |
| 2 | `scripts/analyze_nadistuti.py` | `results/results.json` — RV 10.75.5: Kendall τ, permutation null, Procrustes, Mantel, constrained-permutation null, Bayes factors |
| 3 | `scripts/analyze_corpus.py` | `results/per_corpus/*.json` — the same battery for each of the other four corpora |
| 4 | `scripts/analyze_genesis_subsections.py` | `results/per_corpus/genesis10_subsections.json` — the Japheth/Ham/Shem within-branch analysis |
| 5 | `scripts/analyze_merit_sensitivity.py` | `results/per_corpus/rv_merit.json` — RV result re-checked on NASA-SRTM-derived MERIT-Hydro centroids |
| 6 | `scripts/make_figure.py`, `make_comparative_figure.py` | `figures/*.pdf` |
| 7 | `scripts/make_table.py` | `tables/sensitivity_table.tex` |
| 8 | `verify_claims.py` | PASS/FAIL table over all 35 paper claims |

`scripts/methods.py` holds the shared method implementations
(Procrustes/PROTEST, Mantel, constrained-permutation null, Bayes
factors with a Mallows model).

## Layout

```
reproduce.sh                 one-command full pipeline + verification
verify_claims.py             cross-checks all 35 paper numbers
requirements.txt             numpy, scipy, pandas, matplotlib, requests, rasterio
paper/
  paper.tex, references.bib  the manuscript
  paper.pdf                  compiled paper (read the claims here)
scripts/                     analysis pipeline (see table above)
corpora/                     identification + coordinate CSVs for the
                             non-river corpora, with provenance columns
data/
  river_coordinates.csv      RV 10.75.5 river centroids (OSM-derived)
  river_coordinates_merit.csv  the same, from NASA-SRTM MERIT-Hydro
  paleocourse_coords.csv     paleo-Sarasvatī / paleo-Sutlej coordinates
  osm_cache/                 cached OSM Overpass responses (offline reruns)
results/
  results.json               RV 10.75.5 — every reported number
  per_corpus/*.json          the other corpora + subsections + MERIT check
figures/, tables/            regenerated figure and table artifacts
docs/merit-hydro.md      how to regenerate the MERIT-Hydro centroids
```

## Data sources

- **River coordinates** — OpenStreetMap, via the Overpass API. Raw
  responses are cached in `data/osm_cache/` so the pipeline runs
  offline and is reproducible without re-querying OSM. OSM data is
  © OpenStreetMap contributors, ODbL.
- **MERIT-Hydro centroids** — `data/river_coordinates_merit.csv` is
  derived from MERIT-Hydro (Yamazaki et al. 2019), a NASA-SRTM-derived
  global hydrography product distributed under CC-BY-NC 4.0. The
  derived file inherits the non-commercial restriction; regenerating
  it from the raw tiles needs the steps in `docs/merit-hydro.md`.
- **Corpus identifications** — every place/people identification in
  `corpora/` and in the paleocourse file carries a per-row `source`
  or `provenance` column citing the scholarly work it comes from.

## License

Paper text and figures: CC-BY-4.0. Analysis code: MIT. The
MERIT-Hydro-derived data file is CC-BY-NC-4.0 (see above). Full terms
in `LICENSE`.

## Citation

```bibtex
@misc{kulkarni2026enumerations,
  author = {Kulkarni, Ajinkya},
  title  = {Geographic encoding in premodern enumerations:
            a five-corpus quantitative test},
  year   = {2026},
  note   = {Preprint and replication materials.}
}
```
