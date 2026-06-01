#!/bin/bash
################################################################################
# Launch 16 backfill-friendly Water re-run jobs on partition 2tcourt.
#   - 1 chunk (2 molecules) per job, 1 full node each, 12h walltime.
#   - Small molecules (~5-6h each) => 2 per 12h job, with margin.
#   - Workdirs saved per-molecule => 12h timeout is resume-safe.
#
# Usage:  ./launch_water_2tcourt.sh
#         (run from ~/PlamsV5 on CRIANN; chunk files live in water_chunks/)
################################################################################

set -eu

SOLVENT="${SOLVENT:-Water}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/dft_water_2tcourt.sl"
SUBMIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHUNK_DIR="$SUBMIT_DIR/water_chunks"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: $SCRIPT_PATH not found"
    exit 1
fi

echo "================================================================================"
echo "LAUNCHING WATER 2tcourt BATCH (16 jobs x 1 node, 12h each)"
echo "================================================================================"
echo "  Solvent:     $SOLVENT"
echo "  Partition:   2tcourt"
echo "  Walltime:    12h per job"
echo "  Chunks dir:  $CHUNK_DIR"
echo ""

submitted=0
for chunk in "$CHUNK_DIR"/input_w2t_*.txt; do
    base="$(basename "$chunk")"
    tag="${base#input_w2t_}"; tag="${tag%.txt}"
    JOBNAME="Water_w2t_${tag}"

    echo "-> Submitting $JOBNAME (input: water_chunks/$base)"
    sbatch \
        --job-name="$JOBNAME" \
        --export="ALL,INPUT_FILE=water_chunks/$base,SOLVENT=$SOLVENT" \
        "$SCRIPT_PATH"
    submitted=$((submitted + 1))
done

echo ""
echo "================================================================================"
echo "Submitted $submitted job(s). Check with: squeue -u \$USER"
echo "================================================================================"
