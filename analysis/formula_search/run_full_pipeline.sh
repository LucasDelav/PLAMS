#!/bin/bash
###############################################################################
# Run the complete frequency-descriptor pipeline for the 4 targets:
#   ddU_red, ddU_ox, TdS_red, TdS_ox
#
# Phase A: Step 1a (correlations) + Step 1b (brute-force k=1..K_MAX top_K pool)
# Phase B: Step 2 (Fukui correction)
# Phase C: Step 3 (multi-solvent Padé/Onsager/Born + parity plots)
#
# Usage:
#   ./run_full_pipeline.sh                # K_MAX=3, TOP_K=30 (default)
#   K_MAX=5 TOP_K=50 ./run_full_pipeline.sh  # production
###############################################################################
set -eu

PY="${PY:-/home/cav/Documents/Ecole/Fac/M2/plams/PlamsV5/analysis/formula_search/.venv/bin/python}"
WORKERS="${FORMULA_SEARCH_WORKERS:-14}"
TOP_K="${TOP_K:-30}"
TOP_N="${TOP_N:-50}"
K_MAX="${K_MAX:-3}"

cd "$(dirname "$0")"

echo "================================================================================"
echo "FREQUENCY-DESCRIPTOR SEARCH PIPELINE"
echo "================================================================================"
echo "  Python:      $PY"
echo "  Workers:     $WORKERS"
echo "  TOP_K:       $TOP_K"
echo "  TOP_N:       $TOP_N"
echo "  K_MAX:       $K_MAX"
echo "================================================================================"
echo

# ----- PHASE A1: Correlations (only if missing) ----------------------------
if [ ! -f results/step1_freq_top50_ddU_red.txt ]; then
    echo "[A1] computing Step 1a correlations ..."
    "$PY" step1_freq_correlations.py
    echo
fi

# ----- PHASE A2: Brute-force search per target ------------------------------
for tgt in ddU_red ddU_ox TdS_red TdS_ox; do
    echo "[A2] $tgt — brute force k=1..$K_MAX TOP_K=$TOP_K"
    FORMULA_SEARCH_WORKERS=$WORKERS \
    "$PY" step1_freq_search.py --target "$tgt" --top-k "$TOP_K" \
                                --top-n "$TOP_N" --k-max "$K_MAX"
    echo
done

# ----- PHASE B: Fukui correction --------------------------------------------
echo "[B] Step 2 Fukui correction (all 4 targets)"
"$PY" step2_fukui_correction.py --top-m 20 --top-n 50
echo

# ----- PHASE C: Multi-solvent generalisation --------------------------------
echo "[C] Step 3 Padé solvant + parity plots (all 4 targets)"
"$PY" step3_solvent_pade.py
echo

echo "================================================================================"
echo "PIPELINE DONE"
echo "================================================================================"
echo "Results:"
ls -lh results/step1_freq_top${TOP_N}_*.csv 2>/dev/null
ls -lh results/step2_fukui_*.csv 2>/dev/null
ls -lh results/step3_solvent_*.csv 2>/dev/null
ls -lh results/step3_parity_*.pdf 2>/dev/null
ls -lh results/step3_final_form_*.json 2>/dev/null
