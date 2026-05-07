#!/bin/bash
################################################################################
# launch_predict_dg_missing.sh
# Relance predict-dg pour les molécules non finies du premier batch.
# Lit input_missing_predict_<SOLVENT>.txt pour chaque solvant.
#
# Partition tcourt 24h : ~15-17 mols × ~50 min = ~15h, marge OK.
#
# Usage : ./slurm/launch_predict_dg_missing.sh
#         ./slurm/launch_predict_dg_missing.sh DMSO Water   # subset
################################################################################
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PREDICT_SL="$SCRIPT_DIR/predict_dg.sl"

if [ ! -f "$PREDICT_SL" ]; then
  echo "Error: $PREDICT_SL not found"
  exit 1
fi

CONF_DIR="$SUBMIT_DIR/conformers_data"
if [ ! -d "$CONF_DIR" ]; then
  echo "Error: $CONF_DIR n'existe pas !"
  echo "Lancez d'abord ./sync_conformers_to_criann.sh depuis votre PC"
  exit 1
fi

SOLVENTS=(Hexane Toluene Chloroform Tetrahydrofuran Dichloromethane \
          Acetone Acetonitrile Methanol Water DMSO)

if [ $# -gt 0 ]; then
  SOLVENTS=("$@")
fi

echo "================================================================================"
echo "LANCEMENT PREDICT-DG MISSING POUR ${#SOLVENTS[@]} SOLVANTS"
echo "================================================================================"

for SOLVENT in "${SOLVENTS[@]}"; do
  INPUT_FILE="input_missing_predict_${SOLVENT}.txt"
  if [ ! -f "$SUBMIT_DIR/$INPUT_FILE" ]; then
    echo "  !! $SOLVENT: $INPUT_FILE manquant, skip"
    continue
  fi

  n_mols=$(grep -cv '^[[:space:]]*\(#\|$\)' "$SUBMIT_DIR/$INPUT_FILE")
  if [ "$n_mols" -eq 0 ]; then
    echo "  $SOLVENT: 0 missing mols, skip"
    continue
  fi

  JOB_NAME="PredDGmiss_${SOLVENT}"
  echo "  -> $JOB_NAME  | tcourt 24h  | $n_mols molecules  | $INPUT_FILE"

  sbatch \
    -J "$JOB_NAME" \
    -p tcourt \
    --time 24:00:00 \
    --export=ALL,SOLVENT="$SOLVENT",INPUT_FILE="$INPUT_FILE" \
    "$PREDICT_SL"
done

echo ""
echo "Monitoring : squeue -u \$USER"
echo "================================================================================"
