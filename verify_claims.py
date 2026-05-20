#!/usr/bin/env python3
"""
verify_claims.py — cross-check every numerical claim in the paper
against the regenerated analysis outputs.

Each row of CLAIMS below pins one number that appears in the paper to
its location (section, table, or abstract) and to the results-file path
that produces it. The script loads the regenerated JSON outputs, looks
up each value, and prints a PASS/FAIL table. It exits 0 only if every
claim matches; non-zero otherwise.

Run `./reproduce.sh` first to regenerate the results from scratch, or
run this directly to check the committed results files.

    python3 verify_claims.py

A claim is checked one of two ways:
  - "approx": |actual - expected| must be <= tol
  - "below":  actual must be < threshold   (for "p < 1e-4"-style claims)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"


# ---------------------------------------------------------------------------
# Load every regenerated results file once.
# ---------------------------------------------------------------------------

def load(path: Path):
    if not path.exists():
        sys.stderr.write(f"MISSING results file: {path}\n"
                         f"Run ./reproduce.sh first.\n")
        sys.exit(2)
    return json.loads(path.read_text())


R = {
    "rv":        load(RESULTS / "results.json"),
    "mahaj":     load(RESULTS / "per_corpus" / "mahajanapadas.json"),
    "avesta":    load(RESULTS / "per_corpus" / "avesta_fargard1.json"),
    "bisutun":   load(RESULTS / "per_corpus" / "bisutun.json"),
    "genesis":   load(RESULTS / "per_corpus" / "genesis10.json"),
    "gen_subs":  load(RESULTS / "per_corpus" / "genesis10_subsections.json"),
    "rv_merit":  load(RESULTS / "per_corpus" / "rv_merit.json"),
}


def drop_pos7_tau() -> float:
    """The drop-position-7 (n=9) variant lives in the sensitivity_ids list."""
    for row in R["rv"]["sensitivity_ids"]:
        if "DROPPED" in str(row.get("marudvridha", "")):
            return row["tau"]
    raise KeyError("drop-position-7 variant not found in sensitivity_ids")


# ---------------------------------------------------------------------------
# CLAIMS: (paper location, description, actual-value, check-kind, target, tol)
#   check-kind "approx": pass if |actual - target| <= tol
#   check-kind "below":  pass if actual < target   (tol ignored)
# ---------------------------------------------------------------------------

CLAIMS = [
    # ---- RV 10.75.5 (Abstract, Table 1, Appendix A) ----
    ("Abstract/Table 1", "RV 10.75.5  Kendall tau vs longitude",
     R["rv"]["primary"]["tau"], "approx", -0.911, 0.0015),
    ("Table 1", "RV 10.75.5  primary permutation p",
     R["rv"]["primary"]["p_two_sided"], "below", 1e-4, None),
    ("Appendix A", "RV 10.75.5  null distribution mean",
     R["rv"]["primary"]["null_mean"], "approx", -0.001, 0.01),
    ("Appendix A", "RV 10.75.5  null distribution std",
     R["rv"]["primary"]["null_std"], "approx", 0.251, 0.01),
    ("Table 1", "RV 10.75.5  Procrustes m^2",
     R["rv"]["procrustes"]["m2"], "approx", 0.18, 0.006),
    ("Table 1", "RV 10.75.5  Procrustes p",
     R["rv"]["procrustes"]["p_one_sided"], "below", 1e-4, None),
    ("Table 1", "RV 10.75.5  Mantel r",
     R["rv"]["mantel"]["mantel_r"], "approx", 0.767, 0.0015),
    ("Table 1", "RV 10.75.5  Mantel p",
     R["rv"]["mantel"]["p_one_sided"], "below", 1e-4, None),
    ("Appendix A", "RV 10.75.5  constrained-perm: admissible orderings",
     R["rv"]["constrained_perm_null"]["n_admissible"], "approx", 4320, 0),
    ("Appendix A", "RV 10.75.5  constrained-perm: orderings >= observed",
     R["rv"]["constrained_perm_null"]["n_extreme_geq_observed"], "approx", 6, 0),
    ("Appendix A", "RV 10.75.5  constrained-perm p",
     R["rv"]["constrained_perm_null"]["p_constrained"], "approx", 0.0014, 0.0005),

    # ---- RV 10.75.5 sensitivity analyses (Appendix A) ----
    ("Appendix A", "RV sensitivity: tau vs latitude",
     R["rv"]["sensitivity_axis"]["latitude"]["tau"], "approx", 0.956, 0.0015),
    ("Appendix A", "RV sensitivity: tau vs PC1",
     R["rv"]["sensitivity_axis"]["pc1"]["tau"], "approx", 0.911, 0.0015),
    ("Appendix A", "RV sensitivity: paleocourse tau",
     R["rv"]["sensitivity_paleo"]["tau"], "approx", -0.778, 0.0015),
    ("Appendix A", "RV sensitivity: paleocourse p",
     R["rv"]["sensitivity_paleo"]["p_two_sided"], "approx", 0.001, 0.001),
    ("Appendix A", "RV sensitivity: drop-position-7 (n=9) tau",
     drop_pos7_tau(), "approx", -0.944, 0.0015),
    ("Appendix A", "RV sensitivity: perturbation 2.5% quantile",
     R["rv"]["sensitivity_perturbation"]["tau_p025"], "approx", -0.956, 0.01),
    ("Appendix A", "RV sensitivity: perturbation 97.5% quantile",
     R["rv"]["sensitivity_perturbation"]["tau_p975"], "approx", -0.644, 0.01),

    # ---- RV 10.75.5 MERIT-Hydro cross-check (Appendix A) ----
    ("Appendix A", "RV MERIT-Hydro centroids: Kendall tau",
     R["rv_merit"]["kendall_lon"]["tau"], "approx", -0.867, 0.002),

    # ---- Mahajanapadas (Abstract, Table 1) ----
    ("Abstract/Table 1", "Mahajanapadas  Kendall tau vs longitude",
     R["mahaj"]["kendall_axis"]["longitude"]["tau"], "approx", -0.783, 0.0015),
    ("Table 1", "Mahajanapadas  Procrustes m^2",
     R["mahaj"]["procrustes"]["m2"], "approx", 0.43, 0.006),
    ("Table 1", "Mahajanapadas  Mantel r",
     R["mahaj"]["mantel"]["mantel_r"], "approx", 0.617, 0.0015),

    # ---- Bisutun (Abstract, Table 1) ----
    ("Abstract/Table 1", "Bisutun dahyava  Kendall tau vs longitude",
     R["bisutun"]["kendall_axis"]["longitude"]["tau"], "approx", 0.472, 0.0015),
    ("Table 1", "Bisutun dahyava  Procrustes m^2",
     R["bisutun"]["procrustes"]["m2"], "approx", 0.51, 0.006),
    ("Table 1", "Bisutun dahyava  Mantel r",
     R["bisutun"]["mantel"]["mantel_r"], "approx", 0.505, 0.0015),

    # ---- Vendidad Fargard 1 (Abstract, Table 1) ----
    ("Abstract/Table 1", "Vendidad Fargard 1  Kendall tau vs longitude",
     R["avesta"]["kendall_axis"]["longitude"]["tau"], "approx", -0.117, 0.0015),
    ("Table 1", "Vendidad Fargard 1  Procrustes m^2",
     R["avesta"]["procrustes"]["m2"], "approx", 0.98, 0.006),
    ("Table 1", "Vendidad Fargard 1  Mantel r",
     R["avesta"]["mantel"]["mantel_r"], "approx", 0.112, 0.0015),

    # ---- Genesis 10, flat (Abstract, Table 1) ----
    ("Abstract/Table 1", "Genesis 10 flat  Kendall tau vs longitude",
     R["genesis"]["kendall_axis"]["longitude"]["tau"], "approx", 0.174, 0.0015),
    ("Table 1", "Genesis 10 flat  Procrustes m^2",
     R["genesis"]["procrustes"]["m2"], "approx", 0.91, 0.006),
    ("Table 1", "Genesis 10 flat  Mantel r",
     R["genesis"]["mantel"]["mantel_r"], "approx", 0.172, 0.0015),
    ("Table 1", "Genesis 10 flat  Mantel p",
     R["genesis"]["mantel"]["p_one_sided"], "approx", 0.0004, 0.0005),

    # ---- Genesis 10 within-subsection (Section 3.4, Table 1) ----
    ("Sec 3.4/Table 1", "Genesis 10  Japheth  tau vs latitude",
     R["gen_subs"]["subsections"]["Japheth"]["tau_latitude"], "approx", -0.42, 0.01),
    ("Sec 3.4/Table 1", "Genesis 10  Ham  tau vs latitude",
     R["gen_subs"]["subsections"]["Ham"]["tau_latitude"], "approx", 0.42, 0.01),
    ("Sec 3.4/Table 1", "Genesis 10  Shem  tau vs latitude",
     R["gen_subs"]["subsections"]["Shem"]["tau_latitude"], "approx", -0.56, 0.01),
]


def check(actual, kind, target, tol):
    if kind == "approx":
        return abs(actual - target) <= tol
    if kind == "below":
        return actual < target
    raise ValueError(f"unknown check kind {kind!r}")


def main() -> int:
    print("=" * 78)
    print("Verifying every numerical claim in the paper against regenerated "
          "results")
    print("=" * 78)
    print(f"{'':2}{'Location':18}{'Claim':46}{'paper':>9}  {'actual':>9}  ok")
    print("-" * 78)

    n_pass = 0
    n_fail = 0
    for location, desc, actual, kind, target, tol in CLAIMS:
        ok = check(actual, kind, target, tol)
        n_pass += ok
        n_fail += not ok
        if kind == "below":
            paper_str = f"<{target:g}"
        else:
            paper_str = f"{target:+.4g}" if isinstance(target, float) else f"{target}"
        actual_str = (f"{actual:+.4g}" if isinstance(actual, float)
                      else f"{actual}")
        mark = "PASS" if ok else "**FAIL**"
        print(f"  {location:18}{desc:46}{paper_str:>9}  {actual_str:>9}  {mark}")

    print("-" * 78)
    total = n_pass + n_fail
    print(f"  {n_pass}/{total} claims verified", end="")
    if n_fail:
        print(f"  --  {n_fail} FAILED")
        print("\nFAIL: regenerated results do not match the paper. "
              "Investigate before submitting.")
        return 1
    print("  --  all paper numbers reproduce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
