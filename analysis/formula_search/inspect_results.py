#!/usr/bin/env python3
"""Quick inspection of Phase 3 results.

Prints:
  - per-pair top-10 by mae_total
  - per-pair top-5 by BIC
  - distribution of best k and best form_kind
  - parsimonious recommendation: best form with n_params <= median
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "analysis" / "formula_search" / "results"


def main() -> None:
    pq = RESULTS / "all_results.parquet"
    if not pq.exists():
        print(f"Not found: {pq} -- has Phase 3 finished?")
        return
    df = pd.read_parquet(pq)
    print(f"Loaded {len(df):,} evaluated forms over {df.pair.nunique()} pairs\n")

    for pair, sub in df.groupby("pair"):
        print("=" * 80)
        print(f"PAIR: {pair}")
        print("=" * 80)
        print(f"  n_forms_evaluated: {len(sub):,}")
        print(f"  best mae_total:    {sub.mae_total.min():.5g}")
        print(f"  median mae_total:  {sub.mae_total.median():.5g}")
        print()

        print("  TOP 10 by mae_total:")
        cols = ["k", "n_params", "form_kind", "descriptors", "label",
                "mae_red_cv", "mae_ox_cv", "mae_total", "r2_red_cv", "r2_ox_cv"]
        top = sub.nsmallest(10, "mae_total")[cols]
        print(top.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print()

        print("  TOP 5 by BIC (parsimony):")
        top_bic = sub.nsmallest(5, "bic_total")[cols + ["bic_total"]]
        print(top_bic.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print()


if __name__ == "__main__":
    main()
