#!/usr/bin/env python3
"""
Statistical methods for the geographic-encoding analysis.

This module holds the four methods used alongside the plain Kendall's
τ permutation test:

  1. procrustes_test          Procrustes / PROTEST superimposition with a
                              permutation null. Eliminates the lat/lon
                              redundancy of testing Kendall τ against
                              longitude and latitude separately.

  2. constrained_perm_null    Permutation null restricted to orderings that
                              match the syllable-shape pattern of the verse's
                              ten river names. Defeats the strawman-null
                              critique that "uniform over 10! is implausible
                              because metrical constraints limit re-ordering."

  3. bayes_factors            Likelihoods of the observed verse under three
                              generative models (uniform, meter-admissible,
                              Mallows-around-longitude-sort) and the
                              corresponding Bayes factors.

The verse-position river names with their inflected syllable counts (per the
padapāṭha at https://xn--j2b3a4c.com/rigveda/10/75/5 and the agent-4 prosodic
analysis) are:

    pos 1   gaṅge          2 syll
    pos 2   yamune         3 syll
    pos 3   sarasvati      4 syll
    pos 4   śutudri        3 syll
    pos 5   paruṣṇyā       4 syll
    pos 6   asiknyā        3 syll
    pos 7   marudvṛdhe     4 syll
    pos 8   vitastayā      4 syll
    pos 9   ārjīkīye       4 syll
    pos 10  suṣomayā       4 syll

Syllable-shape buckets: {2:[1], 3:[2,4,6], 4:[3,5,7,8,9,10]}.
Meter-admissible orderings: 1! × 3! × 6! = 4320.
"""

from __future__ import annotations

from itertools import permutations
from math import factorial, log

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import kendalltau


# Syllable shapes per verse position (1-indexed → 0-indexed for arrays).
SYLLABLE_SHAPES = [2, 3, 4, 3, 4, 3, 4, 4, 4, 4]


# ---------------------------------------------------------------------------
# (1) Procrustes / PROTEST
# ---------------------------------------------------------------------------

def _procrustes_m2(verse_rank: np.ndarray, geo_coords: np.ndarray) -> float:
    """Return the Procrustes disparity m² between a 1D verse-rank embedding
    and the 2D geographic configuration.

    Embed verse-order rank as 2D points (rank, 0) so scipy.spatial.procrustes
    (which requires same-shape inputs) can superimpose them onto (lon, lat).
    Procrustes optimally rotates / scales / translates to minimize residuals;
    m² is the residual sum of squared distances after the fit. Smaller m² =
    better alignment.
    """
    verse_2d = np.column_stack([verse_rank, np.zeros_like(verse_rank)]).astype(float)
    geo_2d = geo_coords.astype(float)
    _, _, disparity = procrustes(geo_2d, verse_2d)
    return float(disparity)


def procrustes_test(verse_rank: np.ndarray, geo_coords: np.ndarray,
                    rng: np.random.Generator, n_perm: int = 10_000) -> dict:
    """Procrustes superimposition + permutation null.

    Permutes verse-rank labels and recomputes m² under the null that there is
    no systematic alignment between verse position and 2D geography. Returns
    observed m², two-sided permutation p-value (proportion of permuted m² ≤
    observed), and null distribution summary.
    """
    m2_obs = _procrustes_m2(verse_rank, geo_coords)
    n = len(verse_rank)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = _procrustes_m2(perm, geo_coords)
    p_val = float(np.mean(null <= m2_obs))
    return {
        "m2": m2_obs,
        "p_one_sided": p_val,
        "n_perm": n_perm,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "null_p025": float(np.quantile(null, 0.025)),
        "null_p975": float(np.quantile(null, 0.975)),
    }


# ---------------------------------------------------------------------------
# (2) Constrained permutation null
# ---------------------------------------------------------------------------

def _meter_admissible_perms(shapes: list[int]) -> list[tuple[int, ...]]:
    """Enumerate all permutations of indices 0..n-1 that preserve syllable
    shape position-by-position. I.e. position i in the permuted ordering must
    have a river with the same shape as position i in the original.

    For shapes [2,3,4,3,4,3,4,4,4,4] this gives 1! × 3! × 6! = 4320 orderings.
    """
    n = len(shapes)
    # Group original indices by their shape value.
    buckets: dict[int, list[int]] = {}
    for i, s in enumerate(shapes):
        buckets.setdefault(s, []).append(i)

    # For each bucket, enumerate all permutations of its members. Then take
    # the Cartesian product across buckets and reassemble into full
    # permutations of length n.
    bucket_keys = list(buckets.keys())
    bucket_perms = [list(permutations(buckets[k])) for k in bucket_keys]

    results: list[tuple[int, ...]] = []
    # Iterate over the Cartesian product manually to avoid a heavy import.
    from itertools import product
    for combo in product(*bucket_perms):
        # combo is a tuple of one permutation-tuple per bucket. Reassemble:
        # for each position 0..n-1, look up which bucket it belongs to and
        # pick the next element from that bucket's permutation.
        out = [0] * n
        bucket_pos = {k: 0 for k in bucket_keys}
        for pos, s in enumerate(shapes):
            perm_for_bucket = combo[bucket_keys.index(s)]
            out[pos] = perm_for_bucket[bucket_pos[s]]
            bucket_pos[s] += 1
        results.append(tuple(out))
    return results


MAX_ENUM = 50_000


def _count_admissible(shapes: list[int]) -> int:
    """Closed-form count of position-by-position shape-preserving permutations."""
    from collections import Counter
    bucket_sizes = Counter(shapes).values()
    prod = 1
    for s in bucket_sizes:
        prod *= factorial(s)
    return prod


def _sample_admissible_perm(shapes: list[int],
                            rng: np.random.Generator) -> tuple[int, ...]:
    """Sample one shape-preserving permutation of indices 0..n-1 uniformly.

    Identical in distribution to picking from `_meter_admissible_perms`
    uniformly, but avoids enumerating the full set.
    """
    from collections import defaultdict
    n = len(shapes)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(shapes):
        buckets[s].append(i)
    # Shuffle the indices within each bucket independently.
    shuffled = {k: list(rng.permutation(v)) for k, v in buckets.items()}
    out = [0] * n
    pos: dict[int, int] = defaultdict(int)
    for i, s in enumerate(shapes):
        out[i] = shuffled[s][pos[s]]
        pos[s] += 1
    return tuple(out)


def constrained_perm_null(verse_rank: np.ndarray, geo_value: np.ndarray,
                          shapes: list[int] = SYLLABLE_SHAPES,
                          rng: np.random.Generator | None = None) -> dict:
    """Compute τ-against-`geo_value` for every meter-admissible permutation
    of the n names (or, if there are too many, a Monte Carlo sample of them).

    Behavior:
      * If the admissible set fits in `MAX_ENUM`, enumerate exhaustively
        (deterministic; the exact small-sample tail).
      * Otherwise, draw `MAX_ENUM` shape-preserving permutations uniformly
        at random (need an rng).
      * If all shapes are equal, the "constraint" reduces to the uniform
        permutation set — equivalent to the plain Kendall permutation
        null — and we return `applicable: False` to signal this is not a
        meaningful constraint for the corpus.
    """
    n_admissible = _count_admissible(shapes)
    tau_obs, _ = kendalltau(verse_rank, geo_value)

    # Check applicability: if all syllable shapes are the same, the
    # constraint collapses to "any ordering" — no metrical leverage.
    if len(set(shapes)) <= 1:
        return {
            "applicable": False,
            "reason": "all shapes equal — constraint reduces to uniform null",
            "tau_obs": float(tau_obs),
            "n_admissible": n_admissible,
        }

    if n_admissible <= MAX_ENUM:
        admissible = _meter_admissible_perms(shapes)
        taus = np.empty(len(admissible))
        for i, perm in enumerate(admissible):
            reordered = geo_value[list(perm)]
            t, _ = kendalltau(np.arange(1, len(perm) + 1), reordered)
            taus[i] = t
        method = "exhaustive_enumeration"
        n_samples = len(admissible)
    else:
        if rng is None:
            rng = np.random.default_rng(20260518)
        taus = np.empty(MAX_ENUM)
        for i in range(MAX_ENUM):
            perm = _sample_admissible_perm(shapes, rng)
            reordered = geo_value[list(perm)]
            t, _ = kendalltau(np.arange(1, len(perm) + 1), reordered)
            taus[i] = t
        method = "monte_carlo_sampling"
        n_samples = MAX_ENUM

    n_extreme = int(np.sum(np.abs(taus) >= abs(tau_obs)))
    return {
        "applicable": True,
        "tau_obs": float(tau_obs),
        "n_admissible": n_admissible,
        "method": method,
        "n_samples": n_samples,
        "n_extreme_geq_observed": n_extreme,
        "p_constrained": n_extreme / n_samples,
        "null_mean": float(taus.mean()),
        "null_std": float(taus.std(ddof=1)),
        "null_p025": float(np.quantile(taus, 0.025)),
        "null_p975": float(np.quantile(taus, 0.975)),
    }


# ---------------------------------------------------------------------------
# (3) Mantel test (distance-matrix correlation)
# ---------------------------------------------------------------------------

def _great_circle_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Symmetric n×n matrix of great-circle distances (radians of arc).
    Vectorized haversine. Returns distances in arc-radians; absolute scale
    doesn't matter for Mantel since it's correlation-based.
    """
    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2)
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * np.arcsin(np.sqrt(a))


def mantel_test(verse_rank: np.ndarray, geo_coords: np.ndarray,
                rng: np.random.Generator, n_perm: int = 10_000) -> dict:
    """Mantel test for correlation between pairwise verse-order distance
    and pairwise great-circle geographic distance.

    Computes Pearson correlation between the upper triangles of two n×n
    distance matrices, and generates a null distribution by permuting the
    rows+columns of one matrix. One-tailed p (positive Mantel r means
    "items close in the verse are close on the ground").

    Unlike Kendall's τ, Mantel does not assume monotonic relationship —
    so a non-monotonic but spatially coherent ordering (e.g., a list that
    wanders across a region in any pattern, but with adjacent items
    consistently nearby) will register here.
    """
    n = len(verse_rank)
    pos_dist = np.abs(verse_rank[:, None] - verse_rank[None, :]).astype(float)
    geo_dist = _great_circle_distance_matrix(geo_coords)

    iu = np.triu_indices(n, k=1)
    vec_pos = pos_dist[iu]
    vec_geo = geo_dist[iu]

    # Pearson correlation on the flattened upper triangles.
    r_obs = float(np.corrcoef(vec_pos, vec_geo)[0, 1])

    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        permuted = geo_dist[perm][:, perm]
        null[i] = np.corrcoef(vec_pos, permuted[iu])[0, 1]

    p_one_sided = float(np.mean(null >= r_obs))
    return {
        "mantel_r": r_obs,
        "p_one_sided": p_one_sided,
        "n_perm": n_perm,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "null_p025": float(np.quantile(null, 0.025)),
        "null_p975": float(np.quantile(null, 0.975)),
    }


# ---------------------------------------------------------------------------
# (4) Bayes factor model comparison
# ---------------------------------------------------------------------------

def _kendall_inversions(sigma: np.ndarray, sigma0: np.ndarray) -> int:
    """Number of pair inversions (Kendall distance) between σ and σ₀.

    Equivalent to the number of pairs (i, j) with i < j that are ordered
    one way in σ and the other in σ₀.
    """
    rank0 = np.argsort(sigma0)  # rank-of-each-element in σ₀
    permuted = rank0[sigma]
    inv = 0
    n = len(permuted)
    for i in range(n):
        for j in range(i + 1, n):
            if permuted[i] > permuted[j]:
                inv += 1
    return inv


def _mallows_log_z(theta: float, n: int) -> float:
    """log of the Mallows normalizing constant Z(θ, n) under Kendall distance.

    Z(θ, n) = ∏_{k=1..n-1} (1 - exp(-(k+1) θ)) / (1 - exp(-θ))
            = ∏_{k=1..n} (1 - exp(-k θ)) / (1 - exp(-θ))^n   [equivalent forms]

    Closed form (Fligner & Verducci 1986): Z = ∏_{j=1..n-1} (1 - e^{-(j+1)θ}) / (1 - e^{-θ}).
    """
    if theta <= 0:
        # Uniform limit: Z = n!
        return log(float(factorial(n)))
    log_z = 0.0
    one_minus_e_theta = 1.0 - np.exp(-theta)
    for j in range(1, n):
        log_z += log(1.0 - np.exp(-(j + 1) * theta)) - log(one_minus_e_theta)
    return log_z


def _log_mallows(sigma: np.ndarray, sigma0: np.ndarray, theta: float) -> float:
    """log P(σ | σ₀, θ) under the Mallows model with Kendall distance."""
    d = _kendall_inversions(sigma, sigma0)
    n = len(sigma)
    return -theta * d - _mallows_log_z(theta, n)


def bayes_factors(verse_rank: np.ndarray, geo_value: np.ndarray,
                  shapes: list[int] = SYLLABLE_SHAPES,
                  theta_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)) -> dict:
    """Compute log-likelihoods of the observed verse ordering under three
    generative models, and report Bayes factors.

    Models:
      M_uniform     : σ ∼ Uniform({0,...,n-1}!)
      M_meter       : σ ∼ Uniform(meter-admissible orderings)
      M_geog(θ)     : σ ∼ Mallows(σ₀ = argsort(geo_value), θ)  for several θ

    The observed σ is the identity permutation (0..n-1), because verse_rank
    is itself the position labels and `geo_value` is in verse order. In other
    words, "did the composer place rivers in the order that happens to
    geographically sort them?" The relevant "permutation" is how the
    geographic values are ordered relative to verse position.

    Returns log-likelihoods and Bayes factors (M_geog vs M_uniform,
    M_geog vs M_meter).
    """
    n = len(verse_rank)
    # σ_observed is the identity permutation (verse order itself).
    # σ₀ should be the geographic reference ordering that the verse aligns
    # with, i.e. whichever of ascending or descending sort yields the
    # smaller Kendall distance. The verse runs east-to-west, so longitudes
    # are roughly descending in verse order; descending-sort is the relevant
    # reference. Test both and pick the closer one.
    sigma_obs = np.arange(n)
    sigma0_asc = np.argsort(geo_value)
    sigma0_desc = np.argsort(geo_value)[::-1]
    d_asc = _kendall_inversions(sigma_obs, sigma0_asc)
    d_desc = _kendall_inversions(sigma_obs, sigma0_desc)
    if d_desc <= d_asc:
        sigma0 = sigma0_desc
        sigma0_orientation = "descending (east-to-west)"
    else:
        sigma0 = sigma0_asc
        sigma0_orientation = "ascending (west-to-east)"

    # M_uniform: log P = -log(n!)
    log_p_uniform = -log(float(factorial(n)))

    # M_meter: log P = -log(N_admissible) if observed in admissible set;
    # observed verse ordering IS admissible by construction (shapes match
    # the actual verse). Use the closed-form count to avoid enumerating
    # 16!+ orderings for non-versified corpora.
    n_admissible = _count_admissible(shapes)
    log_p_meter = -log(float(n_admissible))

    # M_geog(θ) for each θ in the grid.
    log_p_geog = {}
    for theta in theta_grid:
        log_p_geog[float(theta)] = _log_mallows(sigma_obs, sigma0, theta)

    # Bayes factors. BF(M_a vs M_b) = exp(log_p_a - log_p_b).
    bf_geog_vs_uniform = {
        f"theta={th}": float(np.exp(lp - log_p_uniform))
        for th, lp in log_p_geog.items()
    }
    bf_geog_vs_meter = {
        f"theta={th}": float(np.exp(lp - log_p_meter))
        for th, lp in log_p_geog.items()
    }
    bf_meter_vs_uniform = float(np.exp(log_p_meter - log_p_uniform))

    return {
        "log_p_uniform": log_p_uniform,
        "log_p_meter": log_p_meter,
        "log_p_geog": {f"theta={th}": float(lp) for th, lp in log_p_geog.items()},
        "n_admissible": n_admissible,
        "kendall_distance_observed": _kendall_inversions(sigma_obs, sigma0),
        "sigma0_orientation": sigma0_orientation,
        "BF_geog_vs_uniform": bf_geog_vs_uniform,
        "BF_geog_vs_meter": bf_geog_vs_meter,
        "BF_meter_vs_uniform": bf_meter_vs_uniform,
        "interpretation": (
            "BF > 100: 'decisive' evidence (Jeffreys 1961 / Kass-Raftery 1995). "
            "BF 10–100: 'strong'. BF 3–10: 'moderate'. BF 1–3: 'weak / no preference'."
        ),
    }
