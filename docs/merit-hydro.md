# Regenerating the MERIT-Hydro centroids

`data/river_coordinates_merit.csv` is provided in this repository, so
the MERIT-Hydro sensitivity check (`scripts/analyze_merit_sensitivity.py`,
step 5 of `reproduce.sh`) runs offline with no download. This file
documents how that CSV was produced, for anyone who wants to
regenerate it from the raw MERIT-Hydro tiles.

## What the check found

Re-deriving the nine OSM-traced river centroids from MERIT-Hydro
(an independent satellite-derived hydrography dataset) gives a mean
centroid shift of 75 km
from the OpenStreetMap polyline centroids. On the MERIT-Hydro
centroids the headline result is essentially unchanged:
τ = −0.867 (p = 0.0001), versus τ = −0.911 on the OSM centroids.
The geographic ordering of RV 10.75.5 does not depend on the
coordinate source.

## Regenerating from raw tiles

1. **Register for MERIT-Hydro.** The product is distributed by the
   Yamazaki Lab (Institute of Industrial Science, University of Tokyo)
   under a CC-BY-NC 4.0 licence, after a one-time registration and
   licence-agreement form:

   > https://global-hydrodynamics.github.io/MERIT_Hydro/

   Registration returns a download password by email. The direct
   download URLs are deliberately **not** reproduced here — obtain
   them yourself through the registration step, so the licence
   agreement is accepted by each user.

2. **Download the upstream-drainage-area (`upa`) tiles** covering
   0°–60°N, 60°–90°E — the two 30°×30° tar packages `upa_n00e060.tar`
   and `upa_n30e060.tar`.

3. **Extract** both tars into `data/merit_hydro/`. They unpack into
   per-package subdirectories of 5°×5° GeoTIFF tiles named like
   `n25e070_upa.tif` (~3 GB extracted).

4. **Run** the derivation script:

   ```bash
   python3 scripts/fetch_river_coords_merit.py
   ```

   For each river it reads the relevant `upa` tiles, thresholds to
   river-network pixels, restricts to pixels within 0.2° of the OSM
   polyline, and writes the unweighted centroid to
   `data/river_coordinates_merit.csv`.

5. **Verify** the sensitivity result is unchanged:

   ```bash
   python3 scripts/analyze_merit_sensitivity.py
   ```

## Licence note

MERIT-Hydro is CC-BY-NC 4.0 (Yamazaki et al. 2019, *Water Resources
Research* 55:5053–5073). The derived file
`data/river_coordinates_merit.csv` inherits the non-commercial
restriction; cite Yamazaki et al. 2019 in any downstream use. The raw
`.tif` tiles are not redistributed in this repository.
