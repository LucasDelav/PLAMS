#!/bin/bash
################################################################################
# Launch 4 parallel Buchard re-run jobs (1 full node each = 4 nodes total)
#   - Mol_3_ox  (retry oxidized side, SCF previously failed)
#   - Mol_4_ox  (verify suspect IP value)
#   - Mol_7_red (from scratch)
#   - Mol_7_ox  (from scratch)
#
# Usage:  ./launch_buchard.sh
################################################################################

set -eu

SOLVENT="${SOLVENT:-Acetonitrile}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/dft_buchard.sl"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: $SCRIPT_PATH not found"
    exit 1
fi

MOLECULES=(
    Mol_3_ox
    Mol_4_ox
    Mol_7_red
    Mol_7_ox
)

echo "================================================================================"
echo "LAUNCHING BUCHARD BATCH (4 jobs x 1 node = 4 nodes parallel)"
echo "================================================================================"
echo "  Solvent:     $SOLVENT"
echo "  Partition:   2tcourt"
echo "  Walltime:    12h per job"
echo "  Molecules:   ${MOLECULES[*]}"
echo ""

for mol in "${MOLECULES[@]}"; do
    INPUT="input_${mol}.txt"
    JOBNAME="Buchard_${mol}"

    if [ ! -f "../$INPUT" ] && [ ! -f "$INPUT" ]; then
        echo "x Skipping $mol: input file $INPUT not found"
        continue
    fi

    echo "-> Submitting $JOBNAME (input: $INPUT)"
    sbatch \
        --job-name="$JOBNAME" \
        --export="ALL,INPUT_FILE=$INPUT,SOLVENT=$SOLVENT" \
        "$SCRIPT_PATH"
done

echo ""
echo "================================================================================"
echo "All jobs submitted. Check with: squeue -u \$USER"
echo "================================================================================"
