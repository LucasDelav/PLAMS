#!/bin/bash
################################################################################
# sync_conformers_to_criann.sh
# Pousse les données conformères minimales vers le CRIANN pour predict-dg.
#
# Données transférées par molécule :
#   - conformers/results/       (conf_*.xyz, neutral_rkf_paths.json, summary)
#   - conformers/calculations/03_dft_opt/  (contient les adf.rkf neutres)
#
# ~3 GB par solvant, ~30 GB total.
#
# Usage :
#   ./sync_conformers_to_criann.sh                    # tous les solvants
#   ./sync_conformers_to_criann.sh Acetonitrile DMSO  # subset
################################################################################
set -eu

HOST="${HOST:-criann}"
REMOTE_DIR="PlamsV5/conformers_data"

cd "$(dirname "$0")"

SOLVENTS=(Hexane Toluene Chloroform Tetrahydrofuran Dichloromethane \
          Acetone Acetonitrile Methanol Water DMSO)

if [ $# -gt 0 ]; then
  SOLVENTS=("$@")
fi

echo "================================================================================"
echo "SYNC CONFORMERES VERS CRIANN"
echo "================================================================================"
echo "  Host:       $HOST"
echo "  Remote dir: ~/$REMOTE_DIR/"
echo "  Solvants:   ${#SOLVENTS[@]}"
echo ""

# Créer le répertoire distant
ssh "$HOST" "mkdir -p $REMOTE_DIR"

for SOLV in "${SOLVENTS[@]}"; do
  echo "=== $SOLV ==="

  if [ ! -d "data/$SOLV" ]; then
    echo "  ! data/$SOLV n'existe pas, skip"
    continue
  fi

  # Compter les molécules
  n_mols=$(ls -d data/$SOLV/*_workdir 2>/dev/null | wc -l)
  echo "  $n_mols molecules"

  # rsync : on envoie uniquement les sous-dossiers nécessaires
  # --include/--exclude pour ne prendre que results/ et 03_dft_opt/
  rsync -av --info=progress2 \
    --include='*_workdir/' \
    --include='*_workdir/conformers/' \
    --include='*_workdir/conformers/results/' \
    --include='*_workdir/conformers/results/**' \
    --include='*_workdir/conformers/calculations/' \
    --include='*_workdir/conformers/calculations/03_dft_opt/' \
    --include='*_workdir/conformers/calculations/03_dft_opt/**' \
    --include='*_workdir/conformers/calculations/boltzmann_filter_summary.txt' \
    --exclude='*' \
    "data/$SOLV/" "$HOST:$REMOTE_DIR/$SOLV/"

  echo ""
done

echo "================================================================================"
echo "Sync terminé."
echo "Données sur le CRIANN : ~/$REMOTE_DIR/<Solvant>/<mol>_workdir/conformers/"
echo "================================================================================"
