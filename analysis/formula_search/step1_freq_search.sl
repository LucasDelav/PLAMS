#!/bin/bash
#SBATCH -p long
#SBATCH -N 1
#SBATCH -n 192
#SBATCH --time 24:00:00
#SBATCH --mem=0
#SBATCH --output formula_search.%j.out
#SBATCH --error  formula_search.%j.err

###############################################################################
# Brute-force frequency-descriptor search on CRIANN (Cobra/AMS partition).
#
# Layout expected on CRIANN ($SLURM_SUBMIT_DIR):
#   formula_search/.venv                # python 3.12 venv with pandas, scipy, sklearn, dcor
#   formula_search/data/*.parquet       # merged_dataset, ddu_delta, freq_descriptors_full
#   formula_search/results/             # output directory (top-N CSVs land here)
#   formula_search/*.py                 # search modules
#
# Run mode (default): full brute-force k=1..5 on top-K=50 pool per target.
###############################################################################

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment)
# -----------------------------------------------------------------------------
TOP_K="${TOP_K:-50}"
TOP_N="${TOP_N:-50}"
K_MAX="${K_MAX:-5}"
TARGETS_DEFAULT="ddU_red ddU_ox TdS_red TdS_ox"
TARGETS="${TARGETS:-$TARGETS_DEFAULT}"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
module purge
module load py_env/python/3.11.7

WORKDIR="$SLURM_SUBMIT_DIR/formula_search"
cd "$WORKDIR"

if [ ! -x ".venv/bin/python" ]; then
    echo "x ERROR: $WORKDIR/.venv/bin/python not found"
    echo "  Setup the venv with: python -m venv .venv && .venv/bin/pip install pandas scipy scikit-learn dcor pyarrow"
    exit 1
fi

PY=.venv/bin/python
echo "Python: $($PY --version)"
echo "Workdir: $WORKDIR"
echo "Targets: $TARGETS"
echo "TOP_K=$TOP_K  TOP_N=$TOP_N  K_MAX=$K_MAX"
echo

# -----------------------------------------------------------------------------
# Step 1a (correlations) — only if missing
# -----------------------------------------------------------------------------
need_corrs=0
for t in $TARGETS; do
    if [ ! -f "results/step1_freq_top50_${t}.txt" ]; then
        need_corrs=1
        break
    fi
done

if [ "$need_corrs" -eq 1 ]; then
    echo "[Step 1a] computing correlations for missing targets ..."
    $PY step1_freq_correlations.py
fi

# -----------------------------------------------------------------------------
# Step 1b (brute-force) — one target at a time, full SLURM cores
# -----------------------------------------------------------------------------
export FORMULA_SEARCH_WORKERS=$SLURM_NTASKS_PER_NODE
export FORMULA_SEARCH_WORKERS=${FORMULA_SEARCH_WORKERS:-$SLURM_NPROCS}
export FORMULA_SEARCH_WORKERS=${FORMULA_SEARCH_WORKERS:-192}

for tgt in $TARGETS; do
    echo "================================================================================"
    echo "[Step 1b] $tgt — brute-force k=1..$K_MAX with TOP_K=$TOP_K (workers=$FORMULA_SEARCH_WORKERS)"
    echo "================================================================================"
    t0=$(date +%s)
    $PY step1_freq_search.py --target "$tgt" --top-k "$TOP_K" --top-n "$TOP_N" --k-max "$K_MAX"
    t1=$(date +%s)
    echo "  [$tgt] elapsed: $((t1 - t0))s"
done

echo
echo "All targets done."
echo "Results: $WORKDIR/results/step1_freq_top${TOP_N}_<target>.csv"
