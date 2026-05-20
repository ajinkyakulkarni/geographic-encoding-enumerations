#!/usr/bin/env python3
"""
Figure 2 — five-panel comparative summary of geographic-encoding
strength, one panel per corpus, laid out on a 2x3 grid (the sixth
cell is left blank).

Each panel: sequence-position vs longitude scatter, a linear fit, and
the corpus's Kendall τ and Procrustes m² annotated.

Outputs: figures/fig2_corpus_comparison.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PDF = REPO_ROOT / "figures" / "fig2_corpus_comparison.pdf"

CORPORA = [
    {
        "label": "RV 10.75.5 (n=10)",
        "csv": REPO_ROOT / "data" / "river_coordinates.csv",
        "results": REPO_ROOT / "results" / "results.json",
        "results_key_primary": "primary",
        "names_col": "vedic_name",
    },
    {
        "label": "Mahājanapadas (n=16)",
        "csv": REPO_ROOT / "corpora" / "mahajanapadas" / "janapadas.csv",
        "results": REPO_ROOT / "results" / "per_corpus" / "mahajanapadas.json",
        "results_key_primary": None,
        "names_col": "name",
    },
    {
        "label": "Bīsutūn dahyāva (n=23)",
        "csv": REPO_ROOT / "corpora" / "bisutun" / "lands.csv",
        "results": REPO_ROOT / "results" / "per_corpus" / "bisutun.json",
        "results_key_primary": None,
        "names_col": "op_name",
    },
    {
        "label": "Avesta Fargard 1 (n=16)",
        "csv": REPO_ROOT / "corpora" / "avesta_fargard1" / "lands.csv",
        "results": REPO_ROOT / "results" / "per_corpus" / "avesta_fargard1.json",
        "results_key_primary": None,
        "names_col": "avestan_name",
    },
    {
        "label": "Genesis 10 (n=62 identified)",
        "csv": REPO_ROOT / "corpora" / "genesis10" / "nations.csv",
        "results": REPO_ROOT / "results" / "per_corpus" / "genesis10.json",
        "results_key_primary": None,
        "names_col": "hebrew_name",
    },
]


def load_corpus(c: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    df = pd.read_csv(c["csv"])
    df = df.dropna(subset=["centroid_lon"]).sort_values("position").reset_index(drop=True)
    # Re-rank to 1..N for comparison with the dropped-rows analysis.
    rank = np.arange(1, len(df) + 1)
    lons = df["centroid_lon"].to_numpy()

    res = json.loads(c["results"].read_text())
    if c["results_key_primary"]:
        # results.json schema (RV 10.75.5): Kendall under a top-level key,
        # Procrustes under "procrustes".
        prim = res[c["results_key_primary"]]
        tau = prim["tau"]
        p = prim["p_two_sided"]
        proc = res.get("procrustes", {})
        m2 = proc.get("m2")
        p_proc = proc.get("p_one_sided")
    else:
        # per-corpus schema: Kendall under kendall_axis.longitude.
        ka = res["kendall_axis"]["longitude"]
        tau = ka["tau"]
        p = ka["p_two_sided"]
        proc = res.get("procrustes", {})
        m2 = proc.get("m2")
        p_proc = proc.get("p_one_sided")

    summary = {"tau": tau, "p": p, "m2": m2, "p_proc": p_proc}
    return rank, lons, summary


def main() -> int:
    # Five corpora on a 2x3 grid; the sixth cell is left blank.
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6))
    axes = axes.flatten()

    for ax, c in zip(axes, CORPORA):
        rank, lons, s = load_corpus(c)
        ax.scatter(rank, lons, s=22, alpha=0.85, color="#8c1d28",
                   edgecolor="black", linewidth=0.4)
        # Best-fit line for visual reference.
        if len(rank) >= 2:
            coef = np.polyfit(rank, lons, 1)
            xs = np.array([rank.min(), rank.max()])
            ax.plot(xs, np.polyval(coef, xs), color="0.35", lw=0.8, linestyle="--")
        ax.set_title(c["label"], fontsize=10)
        ax.set_xlabel("position in text/verse", fontsize=8.5)
        ax.set_ylabel("longitude (°E)", fontsize=8.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8)
        # Annotate τ and Procrustes m². A permutation p of 0.0 means no
        # permuted statistic reached the observed value in 10,000 draws;
        # report it as the resolution floor, "<1e-4", not as "0".
        def fmt_p(pv: float) -> str:
            return "<1e-4" if pv is not None and pv < 1e-4 else f"{pv:.3g}"

        ann = f"τ = {s['tau']:+.3f}   p = {fmt_p(s['p'])}"
        if s["m2"] is not None:
            ann += f"\nProcrustes m² = {s['m2']:.3f}   p = {fmt_p(s['p_proc'])}"
        ax.text(0.97, 0.04, ann, transform=ax.transAxes, fontsize=8.5,
                ha="right", va="bottom", family="monospace",
                bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.3"))

    # Blank any unused cells (the 6th, with five corpora).
    for ax in axes[len(CORPORA):]:
        ax.axis("off")

    fig.suptitle("Geographic encoding across five premodern enumerations",
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    print(f"Wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
