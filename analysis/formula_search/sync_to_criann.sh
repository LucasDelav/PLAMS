#!/bin/bash
###############################################################################
# Sync the formula_search/ directory to CRIANN for the production brute-force
# k=1..5 search.
#
# Pushes:
#   - python modules (forms.py, search.py, freq_descriptors_lib.py, step1*,
#     step2*, step3*, solvent_correction.py)
#   - the SLURM script (step1_freq_search.sl)
#   - the data parquets (~5 MB total: merged_dataset, ddu_delta, freq_descriptors_full)
#   - the existing top-50 correlation rankings (so step 1a is skipped on cluster)
#
# Submits the SLURM job: top_K=50, k_max=5, all 4 targets, ~24h walltime cap.
#
# Usage:
#   ./sync_to_criann.sh           # sync only
#   ./sync_to_criann.sh --submit  # sync and submit job
###############################################################################
set -eu

HOST="${HOST:-criann}"
REMOTE_BASE="${REMOTE_BASE:-PlamsV5}"
REMOTE_DIR="${REMOTE_BASE}/formula_search"

cd "$(dirname "$0")"

echo "Syncing formula_search/ to ${HOST}:${REMOTE_DIR}/ ..."

# Create remote dir
ssh "$HOST" "mkdir -p ${REMOTE_DIR}/data ${REMOTE_DIR}/results"

# Python modules
rsync -avz --progress \
    forms.py search.py solvent_correction.py \
    freq_descriptors_lib.py build_freq_dataset.py \
    step1_freq_correlations.py step1_freq_search.py \
    step2_fukui_correction.py step3_solvent_pade.py \
    step1_freq_search.sl \
    "${HOST}:${REMOTE_DIR}/"

# Data parquets
rsync -avz --progress \
    data/merged_dataset.parquet \
    data/ddu_delta_descriptors.parquet \
    data/freq_descriptors_full.parquet \
    "${HOST}:${REMOTE_DIR}/data/"

# Existing top-50 correlation rankings (skip Step 1a on cluster)
if ls results/step1_freq_top50_*.txt 1>/dev/null 2>&1; then
    rsync -avz \
        results/step1_freq_top50_*.txt \
        results/step1_freq_correlations.csv \
        "${HOST}:${REMOTE_DIR}/results/"
fi

echo
echo "+ Sync complete."
echo "Remote dir: ${HOST}:~/${REMOTE_DIR}/"
echo

# Setup the venv on remote if missing
ssh "$HOST" "cd ${REMOTE_DIR} && [ -x .venv/bin/python ] || (
    echo 'Setting up Python venv on CRIANN ...'
    module purge
    module load py_env/python/3.11.7
    python -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q pandas scipy scikit-learn dcor pyarrow numpy
    echo '+ venv ready'
)"

# Submit if requested
if [ "${1:-}" = "--submit" ]; then
    echo
    echo "Submitting SLURM job ..."
    JOBID=$(ssh "$HOST" "cd ${REMOTE_BASE} && sbatch --parsable formula_search/step1_freq_search.sl")
    echo "+ Submitted job ${JOBID}"
    echo "  Watch:  ssh ${HOST} squeue -j ${JOBID}"
    echo "  Logs:   ssh ${HOST} tail -f ${REMOTE_BASE}/formula_search.${JOBID}.out"
fi
