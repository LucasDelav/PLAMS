#!/bin/bash
################################################################################
# launch_missing.sh
# Soumet la relance des systemes manquants pour CHAQUE solvant.
# Lit input_missing_<SOLVENT>.txt (genere par le script local de matrice).
#
# Mapping partition / walltime :
#   - DMSO            -> court   48h  (7 mols)
#   - tous les autres -> tcourt  24h  (1 mol)
#
# Usage : ./slurm/launch_missing.sh
#         ./slurm/launch_missing.sh DMSO Water     # subset
################################################################################
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DFT_SL="$SCRIPT_DIR/dft.sl"

if [ ! -f "$DFT_SL" ]; then
  echo "Error: $DFT_SL not found"
  exit 1
fi

# solvant -> "partition walltime"
declare -A CONF=(
  [Hexane]="tcourt 24:00:00"
  [Toluene]="tcourt 24:00:00"
  [Chloroform]="tcourt 24:00:00"
  [Tetrahydrofuran]="tcourt 24:00:00"
  [Dichloromethane]="tcourt 24:00:00"
  [Acetone]="tcourt 24:00:00"
  [Acetonitrile]="tcourt 24:00:00"
  [Methanol]="tcourt 24:00:00"
  [Water]="tcourt 24:00:00"
  [DMSO]="court 48:00:00"
)

if [ $# -gt 0 ]; then
  SOLVENTS=("$@")
else
  SOLVENTS=(Hexane Toluene Chloroform Tetrahydrofuran Dichloromethane \
            Acetone Acetonitrile Methanol Water DMSO)
fi

echo "Submitting missing-systems relaunch jobs"
echo "  Submit dir : $SUBMIT_DIR"
echo "  Script     : $DFT_SL"
echo ""

for SOLVENT in "${SOLVENTS[@]}"; do
  CFG="${CONF[$SOLVENT]:-}"
  if [ -z "$CFG" ]; then
    echo "!! No partition mapping for $SOLVENT, skipping"
    continue
  fi
  PART=$(echo "$CFG" | awk '{print $1}')
  TIME=$(echo "$CFG" | awk '{print $2}')

  INPUT_FILE="input_missing_${SOLVENT}.txt"
  if [ ! -f "$SUBMIT_DIR/$INPUT_FILE" ]; then
    echo "!! Missing $INPUT_FILE for $SOLVENT, skipping"
    continue
  fi
  N=$(grep -cv '^[[:space:]]*\(#\|$\)' "$SUBMIT_DIR/$INPUT_FILE")

  JOB_NAME="Missing_${SOLVENT}"
  echo "  -> $JOB_NAME  | $PART  | $TIME  | $N mols  | $INPUT_FILE"

  sbatch \
    -J "$JOB_NAME" \
    -p "$PART" \
    --time "$TIME" \
    --export=ALL,SOLVENT="$SOLVENT",INPUT_FILE="$INPUT_FILE" \
    "$DFT_SL"
done

echo ""
echo "Done. Check status with: squeue -u \$USER"
