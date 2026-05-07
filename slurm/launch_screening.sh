#!/bin/bash
################################################################################
# Launch jobs for all solvents
# Usage: ./launch_screening.sh <dft|prediction> [job_prefix]
# Example: ./launch_screening.sh prediction Catechol
#   → submits 10 jobs: Catechol_Hexane_prediction, ...
################################################################################

if [ -z "$1" ]; then
    echo "Usage: $0 <dft|prediction> [job_prefix]"
    echo ""
    echo "  dft         Full redox pipeline (TZ2P+DZP, 2 nodes)"
    echo "  prediction  Screening predict-dg (3 SP, 1 node)"
    echo ""
    echo "Examples:"
    echo "  $0 prediction Catechol"
    echo "  $0 dft Batch1"
    exit 1
fi

JOB_TYPE="$1"
PREFIX="${2:-plams}"

# Validate job type
case "$JOB_TYPE" in
    dft)
        SCRIPT="dft.sl"
        ;;
    prediction)
        SCRIPT="prediction.sl"
        ;;
    *)
        echo "Error: unknown job type '$JOB_TYPE'"
        echo "Valid types: dft, prediction"
        exit 1
        ;;
esac

SOLVENTS=(
    Hexane
    Toluene
    Chloroform
    Tetrahydrofuran
    Dichloromethane
    Acetone
    Acetonitrile
    DMSO
    Methanol
    Water
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: script not found: $SCRIPT_PATH"
    exit 1
fi

echo "Launching ${#SOLVENTS[@]} $JOB_TYPE jobs for prefix: $PREFIX"
echo "  Script: $SCRIPT"
echo ""

for SOLVENT in "${SOLVENTS[@]}"; do
    JOB_NAME="${PREFIX}_${SOLVENT}_${JOB_TYPE}"
    echo "  Submitting $JOB_NAME ..."
    sbatch --export=ALL,SOLVENT="$SOLVENT" \
           -J "$JOB_NAME" \
           "$SCRIPT_PATH"
done

echo ""
echo "All jobs submitted. Check with: squeue -u \$USER"
