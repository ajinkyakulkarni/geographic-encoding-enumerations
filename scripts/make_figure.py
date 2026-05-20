#!/usr/bin/env python3
"""
Figure 1 — null distribution of Kendall's τ between verse-order rank and
longitude, with the observed τ overlaid.

Inputs:  results/results.json
Outputs: figures/fig1_null_distribution.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO_ROOT / "results" / "results.json"
OUT_PDF = REPO_ROOT / "figures" / "fig1_null_distribution.pdf"


def main() -> int:
    data = json.loads(RESULTS_JSON.read_text())
    null = np.array(data["null_distribution_samples"])
    tau_obs = data["primary"]["tau"]
    p025 = data["primary"]["null_p025"]
    p975 = data["primary"]["null_p975"]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.hist(null, bins=40, color="0.78", edgecolor="0.45", linewidth=0.4)
    ax.axvline(tau_obs, color="#8c1d28", linewidth=1.8, label=f"observed τ = {tau_obs:.3f}")
    ax.axvline(p025, color="0.25", linewidth=0.9, linestyle="--",
               label="null 2.5 / 97.5 % quantiles")
    ax.axvline(p975, color="0.25", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Kendall's τ (verse-order rank vs longitude)")
    ax.set_ylabel("Count (out of 10,000 permutations)")
    ax.set_xlim(-1.05, 1.05)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    print(f"Wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
