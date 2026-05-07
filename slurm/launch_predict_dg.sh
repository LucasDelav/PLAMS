#!/bin/bash
################################################################################
# launch_predict_dg.sh
# Soumet les calculs predict-dg pour les 10 solvants.
#
# Prérequis : avoir lancé sync_conformers_to_criann.sh depuis le PC local
#             pour pousser les conformères dans ~/PlamsV5/conformers_data/
#
# Usage : ./slurm/launch_predict_dg.sh
#         ./slurm/launch_predict_dg.sh Acetonitrile DMSO   # subset
################################################################################
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PREDICT_SL="$SCRIPT_DIR/predict_dg.sl"

if [ ! -f "$PREDICT_SL" ]; then
  echo "Error: $PREDICT_SL not found"
  exit 1
fi

# Vérifier que les conformères ont été sync
CONF_DIR="$SUBMIT_DIR/conformers_data"
if [ ! -d "$CONF_DIR" ]; then
  echo "Error: $CONF_DIR n'existe pas !"
  echo "Lancez d'abord depuis votre PC :"
  echo "  ./sync_conformers_to_criann.sh"
  exit 1
fi

SOLVENTS=(Hexane Toluene Chloroform Tetrahydrofuran Dichloromethane \
          Acetone Acetonitrile Methanol Water DMSO)

if [ $# -gt 0 ]; then
  SOLVENTS=("$@")
fi

echo "================================================================================"
echo "LANCEMENT PREDICT-DG POUR ${#SOLVENTS[@]} SOLVANTS"
echo "================================================================================"
echo "  Submit dir     : $SUBMIT_DIR"
echo "  Script         : $PREDICT_SL"
echo "  Conformers dir : $CONF_DIR"
echo "  Input          : input.txt"
echo ""

for SOLVENT in "${SOLVENTS[@]}"; do
  # Vérifier que le solvant a des données
  n_wk=$(ls -d "$CONF_DIR/$SOLVENT"/*_workdir 2>/dev/null | wc -l)
  if [ "$n_wk" -eq 0 ]; then
    echo "  !! $SOLVENT: aucun workdir dans conformers_data/$SOLVENT, skip"
    continue
  fi

  JOB_NAME="PredDG_${SOLVENT}"
  # 3 SP+NM × ~2.5 conf × 73 mols × ~5 min = ~15-20h → court (48h) suffit
  echo "  -> $JOB_NAME  | court 48h  | $n_wk molecules"

  sbatch \
    -J "$JOB_NAME" \
    -p court \
    --time 48:00:00 \
    --export=ALL,SOLVENT="$SOLVENT" \
    "$PREDICT_SL"
done

echo ""
echo "================================================================================"
echo "Tous les jobs soumis."
echo ""
echo "Monitoring : squeue -u \$USER"
echo "Résultats  : PredDG_<Solvent>_<JobID>/dataset/"
echo "================================================================================"
