#!/bin/bash
# Rapatriement chirurgical des molecules manquantes.
#
# Mode DFT (defaut) : pull des workdirs /dlocal/run/<JobID>/<mol>_workdir/
#                     vers data/<Solvant>/ (pour re-extraction locale).
#
# Mode predict-dg   : pull du dataset/ deja extrait par le job SLURM depuis
#                     ~/PlamsV5/PredDGmiss_<Solv>_<JobID>/dataset/<Solv>/
#                     vers dataset_predict_dg/<Solv>/.
#
# Usage :
#   HOST=ldelav02@criann ./fetch_missing.sh                      # DFT, tout
#   HOST=ldelav02@criann ./fetch_missing.sh Acetone              # DFT, un solvant
#   HOST=ldelav02@criann ./fetch_missing.sh --predict-dg         # predict-dg, tout
#   HOST=ldelav02@criann ./fetch_missing.sh --predict-dg Water   # predict-dg, un solvant
#
set -eu

HOST="${HOST:-criann}"
cd "$(dirname "$0")"

# ---- Parse mode flag --------------------------------------------------------
MODE="dft"
if [ "${1:-}" = "--predict-dg" ]; then
  MODE="predict"
  shift
fi

# ---- Mappings JobID ---------------------------------------------------------
# DFT : jobs 'Missing_<Solv>' (2026-04-20)
declare -A JOB_DFT=(
  [Hexane]=1957440
  [Toluene]=1957441
  [Chloroform]=1957442
  [Tetrahydrofuran]=1957443
  [Dichloromethane]=1957444
  [Acetone]=1957445
  [Acetonitrile]=1957446
  [Methanol]=1957447
  [Water]=1957448
  [DMSO]=1957449
)

# predict-dg : jobs 'PredDGmiss_<Solv>' (2026-04-20, 5 jobs utiles uniquement)
declare -A JOB_PRED=(
  [Hexane]=1957450
  [Toluene]=1957451
  [Chloroform]=1957452
  [Acetone]=1957455
  [Water]=1957458
)

# ---- Selection mode ---------------------------------------------------------
if [ "$MODE" = "dft" ]; then
  declare -n JOB=JOB_DFT
  JOB_PREFIX="Missing"
else
  declare -n JOB=JOB_PRED
  JOB_PREFIX="PredDGmiss"
fi

# Liste cible : dernier champ de chaque ligne non-commentee de input.txt
mapfile -t ALL_MOLS < <(awk '!/^[[:space:]]*#/ && NF>=2 {print $NF}' input.txt)

# Solvants a traiter (argument optionnel)
if [ $# -gt 0 ]; then
  SOLVENTS=("$@")
else
  SOLVENTS=("${!JOB[@]}")
fi

TOTAL_FETCHED=0

echo "Mode : $MODE  (HOST=$HOST)"

for SOLV in "${SOLVENTS[@]}"; do
  ID="${JOB[$SOLV]:-}"
  if [ -z "$ID" ]; then
    echo "!! Pas de JobID pour $SOLV ($MODE), ignore"
    continue
  fi

  echo
  echo "=== $SOLV  (job $ID, $MODE) ==="

  if [ "$MODE" = "dft" ]; then
    # -------------------- MODE DFT ------------------------------------------
    DST="data/$SOLV"
    mkdir -p "$DST"

    # 1. Molecules manquantes localement (critere : redox_summary.csv absent)
    MISSING=()
    for M in "${ALL_MOLS[@]}"; do
      MDIR="${M//,/_}"
      csv="$DST/${MDIR}_workdir/redox/results/${MDIR}_redox_summary.csv"
      [ -s "$csv" ] || MISSING+=("$MDIR")
    done
    if [ ${#MISSING[@]} -eq 0 ]; then
      echo "  Complet en local, rien a rapatrier"
      continue
    fi
    echo "  Manquant local : ${#MISSING[@]}"

    # 2. Quels workdirs sont prets sur le cluster (csv present)
    REMOTE_LIST="${MISSING[*]}"
    READY=$(ssh "$HOST" "cd /dlocal/run/$ID 2>/dev/null || exit 0; for m in $REMOTE_LIST; do [ -s \"\${m}_workdir/redox/results/\${m}_redox_summary.csv\" ] && echo \"\$m\"; done; true")

    if [ -z "$READY" ]; then
      echo "  Aucun pret sur le cluster pour ce job"
      continue
    fi
    N_READY=$(echo "$READY" | wc -l)
    echo "  Pret a rapatrier : $N_READY"
    echo "$READY" | sed 's/^/    - /'

    # 3. rsync un workdir a la fois
    for m in $READY; do
      [ -d "$DST/${m}_workdir" ] && [ -z "$(ls -A "$DST/${m}_workdir" 2>/dev/null)" ] && rmdir "$DST/${m}_workdir"
      rsync -a --info=progress2 "$HOST:/dlocal/run/$ID/${m}_workdir" "$DST/"
    done

    TOTAL_FETCHED=$((TOTAL_FETCHED + N_READY))

  else
    # -------------------- MODE predict-dg -----------------------------------
    # Les JSON sont deja extraits sur CRIANN dans
    # ~/PlamsV5/${JOB_PREFIX}_<Solv>_<JobID>/dataset/<Solv>/<mol>/*.json
    REMOTE_DIR="~/PlamsV5/${JOB_PREFIX}_${SOLV}_${ID}/dataset/${SOLV}"
    DST="dataset_predict_dg/$SOLV"
    mkdir -p "$DST"

    # Verifie que le dossier distant existe (job termine avec succes)
    if ! ssh "$HOST" "[ -d $REMOTE_DIR ]"; then
      echo "  Dossier distant absent : $REMOTE_DIR"
      echo "  (job pas encore termine ou echoue avant extraction)"
      continue
    fi

    # Liste des molecules extraites
    READY=$(ssh "$HOST" "ls $REMOTE_DIR 2>/dev/null" | tr -d '\r')
    if [ -z "$READY" ]; then
      echo "  Aucune molecule extraite sur le cluster"
      continue
    fi
    N_READY=$(echo "$READY" | wc -l)
    echo "  Pret a rapatrier : $N_READY"
    echo "$READY" | sed 's/^/    - /'

    # rsync dossier JSON complet par molecule
    for m in $READY; do
      rsync -a --info=progress2 "$HOST:$REMOTE_DIR/$m/" "$DST/$m/"
    done

    TOTAL_FETCHED=$((TOTAL_FETCHED + N_READY))
  fi
done

echo
echo "=========================================="
echo "Total rapatrie : $TOTAL_FETCHED systemes ($MODE)"
echo "=========================================="
