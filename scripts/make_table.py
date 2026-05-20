#!/usr/bin/env python3
"""
Build Table 2 — sensitivity analysis summary — as a LaTeX fragment.

Inputs:  results/results.json
Outputs: tables/sensitivity_table.tex
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO_ROOT / "results" / "results.json"
OUT_TEX = REPO_ROOT / "tables" / "sensitivity_table.tex"


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"${p:.4f}$"


def fmt_tau(t: float) -> str:
    return f"${t:+.3f}$"


def main() -> int:
    d = json.loads(RESULTS_JSON.read_text())

    rows: list[tuple[str, float, float, str]] = []
    p = d["primary"]
    rows.append(("Primary: longitude",
                 p["tau"], p["p_two_sided"],
                 "modern centroids, 10\\,000 permutations"))
    rows.append(("Axis: latitude",
                 d["sensitivity_axis"]["latitude"]["tau"],
                 d["sensitivity_axis"]["latitude"]["p_two_sided"],
                 "rivers also trend N-S"))
    rows.append(("Axis: PC1",
                 d["sensitivity_axis"]["pc1"]["tau"],
                 d["sensitivity_axis"]["pc1"]["p_two_sided"],
                 "principal component of $(\\mathrm{lon},\\mathrm{lat})$"))
    rows.append(("Paleocourse substitution",
                 d["sensitivity_paleo"]["tau"],
                 d["sensitivity_paleo"]["p_two_sided"],
                 "paleo-Sarasvat\\={\\i}, paleo-Sutlej"))
    for r in d["sensitivity_ids"]:
        short_arj = r['arjikiya'].split(' (')[0]
        short_mar = r['marudvridha'].split(' (')[0]
        rows.append((f"ID: \\={{A}}rj.={short_arj}; Mar.={short_mar}",
                     r["tau"], r["p_two_sided"],
                     ""))
    pert = d["sensitivity_perturbation"]
    rows.append((f"Coord. perturb. ($\\sigma={pert['sigma_deg']}^\\circ$, n=1000)",
                 pert["tau_p500"], -1.0,
                 f"95\\% CI [{pert['tau_p025']:+.3f}, {pert['tau_p975']:+.3f}]"))

    lines: list[str] = []
    lines.append(r"\begin{tabular}{lrrl}")
    lines.append(r"\toprule")
    lines.append(r"Analysis & $\tau$ & $p$ (two-sided) & Note \\")
    lines.append(r"\midrule")
    for label, tau, pval, note in rows:
        p_str = "---" if pval < 0 else fmt_p(pval)
        # Escape underscores/ampersands in note that latex would choke on.
        safe_note = note.replace("&", "\\&")
        # Latin-escape Sanskrit diacritics already present as Unicode are
        # passed through (the paper's preamble loads inputenc + lmodern).
        safe_label = label.replace("&", "\\&")
        lines.append(f"{safe_label} & {fmt_tau(tau)} & {p_str} & {safe_note} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
