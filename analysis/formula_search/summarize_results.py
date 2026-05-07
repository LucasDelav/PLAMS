#!/usr/bin/env python3
"""Summary report across all 4 targets and 3 pipeline stages.

Reads:
    results/step1_freq_top50_<target>.csv     (Step 1: brute force Acetonitrile)
    results/step2_fukui_<target>.csv          (Step 2: Fukui correction)
    results/step3_solvent_<target>.csv        (Step 3: multi-solvent ε term)
    results/step3_final_form_<target>.json    (final retained model)

Prints a consolidated comparison table vs the rapport baselines, and writes
results/summary_freq_pipeline.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "analysis" / "formula_search" / "results"

TARGETS = ["ddU_red", "ddU_ox", "TdS_red", "TdS_ox"]
BASELINE = {
    "ddU_red": (0.811, 0.957, 4),
    "ddU_ox":  (0.957, 0.957, 4),
    "TdS_red": (1.392, 1.392, 3),
    "TdS_ox":  (1.859, 1.859, 3),
}  # we only use the first MAE field per target


def best_step1(target: str) -> dict:
    f = RESULTS / f"step1_freq_top50_{target}.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    best = df.iloc[0]
    return {"mae": float(best["mae_cv"]), "n": int(best["n_params"]),
            "label": best["label"], "fukui": "none"}


def best_step2(target: str) -> dict:
    f = RESULTS / f"step2_fukui_{target}.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    best = df.iloc[0]
    return {"mae": float(best["mae_cv"]), "n": int(best["n_params"]),
            "label": best["label"], "fukui": best["fukui"]}


def best_step3(target: str) -> dict:
    f = RESULTS / f"step3_final_form_{target}.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return {"mae": float(d["mae_cv"]),
            "rmse": float(d["rmse_cv"]),
            "r2": float(d["r2_cv"]),
            "n": int(d["n_params_total"]),
            "family": d["solvent_family"],
            "fukui": d["fukui"],
            "label": d["base_label"]}


def fmt(d: dict, key: str = "mae") -> str:
    if not d:
        return "—"
    if key in d:
        return f"{d[key]:.3f} (k={d['n']})"
    return "—"


def main():
    rows = []
    for t in TARGETS:
        s1 = best_step1(t)
        s2 = best_step2(t)
        s3 = best_step3(t)
        baseline_mae = BASELINE[t][0]
        rows.append({
            "target": t,
            "baseline_rapport_mae": baseline_mae,
            "baseline_rapport_k": BASELINE[t][2],
            "step1_mae": s1.get("mae"),
            "step1_k": s1.get("n"),
            "step2_mae": s2.get("mae"),
            "step2_k": s2.get("n"),
            "step2_fukui": s2.get("fukui"),
            "step3_mae": s3.get("mae"),
            "step3_k": s3.get("n"),
            "step3_family": s3.get("family"),
            "step3_r2": s3.get("r2"),
            "improvement_step3_vs_baseline": (
                f"{(s3['mae'] - baseline_mae) / baseline_mae * 100:+.1f}%"
                if s3 else None),
        })

    df = pd.DataFrame(rows)
    print("=" * 100)
    print("SUMMARY — frequency-descriptor pipeline vs rapport baselines")
    print("=" * 100)
    print(df.to_string(index=False, float_format="%.3f"))
    print()

    # Markdown
    lines = ["# Frequency-descriptor pipeline — summary report", "",
             "Pipeline 3 étapes : Step 1 (brute force Acetonitrile) →",
             "Step 2 (correction Fukui) → Step 3 (Padé/Onsager solvant 10 solvants).",
             "",
             "## Comparaison cible par cible (MAE en kJ/mol)", "",
             "| Cible | Baseline rapport | Step 1 (Acetonitrile) | Step 2 (+Fukui) | Step 3 (multi-solvant) | Δ vs baseline |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| **{r['target']}** | "
            f"{r['baseline_rapport_mae']:.3f} (k={r['baseline_rapport_k']}) | "
            f"{r['step1_mae']:.3f} (k={r['step1_k']}) | "
            f"{r['step2_mae']:.3f} (k={r['step2_k']}, {r['step2_fukui']}) | "
            f"**{r['step3_mae']:.3f}** (k={r['step3_k']}, {r['step3_family']}) | "
            f"{r['improvement_step3_vs_baseline']} |"
        )
    lines += ["", "## Forme retenue par cible", ""]
    for t in TARGETS:
        s3 = best_step3(t)
        if not s3:
            continue
        lines += [f"### {t}",
                  f"- **MAE** = {s3['mae']:.4f} kJ/mol, **R²** = {s3['r2']:.4f}, k = {s3['n']}",
                  f"- Forme de base : `{s3['label']}`",
                  f"- Correction Fukui : `{s3['fukui']}`",
                  f"- Famille solvant : `{s3['family']}`",
                  ""]
    out = RESULTS / "summary_freq_pipeline.md"
    out.write_text("\n".join(lines))
    print(f"+ Saved {out}")


if __name__ == "__main__":
    main()
