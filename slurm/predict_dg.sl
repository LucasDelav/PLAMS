#!/bin/bash

#SBATCH -p long
#SBATCH -N 1
#SBATCH -n 192
#SBATCH --time 72:00:00
#SBATCH --mem=0

#SBATCH --output summary.%j
#SBATCH --error error.%j

################################################################################
# PLAMS workflow Predict-DG
#
# Pour chaque molécule :
#   1. Copie les conformères pré-calculés depuis CONFORMERS_DIR
#   2. Fixe les chemins dans neutral_rkf_paths.json
#   3. Lance redox_calc.py --predict-dg (3 SP+NormalModes + ConceptualDFT + QTAIM)
#
# Les données conformères doivent être pré-chargées sur le CRIANN via
# sync_conformers_to_criann.sh AVANT de soumettre ce job.
#
# Variables d'environnement :
#   SOLVENT        - Nom du solvant (défaut: Acetonitrile)
#   INPUT_FILE     - Fichier d'entrée (défaut: input.txt)
#   CONFORMERS_DIR - Répertoire contenant les <mol>_workdir/conformers/
#                    (défaut: $SLURM_SUBMIT_DIR/conformers_data/$SOLVENT)
################################################################################

# =============================================================================
# CONFIGURATION
# =============================================================================
SOLVENT="${SOLVENT:-Acetonitrile}"
INPUT_FILE="${INPUT_FILE:-input.txt}"
CONFORMERS_DIR="${CONFORMERS_DIR:-$SLURM_SUBMIT_DIR/conformers_data/$SOLVENT}"

# AMS environment
module purge
module load atomic_simu/cobra-ams/2025.106_amd
module load py_env/python/3.11.7
module list
echo ""

# Disable GPU
export SCM_GPUENABLED="FALSE"

# Logging
SUMMARY_LOG="summary.log"
FAILED_LOG="failed_molecules.txt"
TIMING_LOG="timing.log"

echo "================================================================================"
echo "PLAMS PREDICT-DG - Job ID: $SLURM_JOB_ID"
echo "================================================================================"
echo "Configuration:"
echo "  Solvent:         $SOLVENT"
echo "  Input File:      $INPUT_FILE"
echo "  Conformers dir:  $CONFORMERS_DIR"
echo "  Mode:            predict-dg (3 SP+NormalModes + ConceptualDFT + QTAIM)"
echo "  Defaults:        PBE0/DZP, Boltzmann 0.10"
echo "================================================================================"
echo ""

# =============================================================================
# Vérification préalable : le répertoire conformères existe-t-il ?
# =============================================================================
if [ ! -d "$CONFORMERS_DIR" ]; then
    echo "x ERREUR FATALE: $CONFORMERS_DIR n'existe pas !"
    echo "  Avez-vous lancé sync_conformers_to_criann.sh depuis votre PC ?"
    exit 1
fi

n_workdirs=$(ls -d "$CONFORMERS_DIR"/*_workdir 2>/dev/null | wc -l)
echo "Conformères disponibles: $n_workdirs workdirs dans $CONFORMERS_DIR"
if [ "$n_workdirs" -eq 0 ]; then
    echo "x ERREUR FATALE: aucun *_workdir trouvé dans $CONFORMERS_DIR"
    exit 1
fi
echo ""

# =============================================================================
# Copie du code vers le répertoire de travail local
# =============================================================================
echo "Setting up working directory..."
cp "$SLURM_SUBMIT_DIR/$INPUT_FILE" "$LOCAL_WORK_DIR"
cp "$SLURM_SUBMIT_DIR/conformers_gen.py" "$LOCAL_WORK_DIR"
cp "$SLURM_SUBMIT_DIR/redox_calc.py" "$LOCAL_WORK_DIR"
cp "$SLURM_SUBMIT_DIR/extract_predict_dg.py" "$LOCAL_WORK_DIR"
cp -r "$SLURM_SUBMIT_DIR/common/" "$LOCAL_WORK_DIR"
cp -r "$SLURM_SUBMIT_DIR/qmconf/" "$LOCAL_WORK_DIR"
cp -r "$SLURM_SUBMIT_DIR/qmredox/" "$LOCAL_WORK_DIR"
echo "  Code copié dans $LOCAL_WORK_DIR"
echo ""

# =============================================================================
# Copie des conformères + fixation des chemins
# =============================================================================
echo "================================================================================"
echo "COPIE DES CONFORMERES ET FIXATION DES CHEMINS"
echo "================================================================================"

parse_input_early() {
    grep -v '^#' "$SLURM_SUBMIT_DIR/$INPUT_FILE" | grep -v '^[[:space:]]*$'
}

copied=0
missing=0

while read -r smiles name; do
    workdir="${name}_workdir"
    src="$CONFORMERS_DIR/$workdir/conformers"

    if [ -d "$src/results" ] && [ -f "$src/results/neutral_rkf_paths.json" ]; then
        # Copier la structure conformères
        mkdir -p "$LOCAL_WORK_DIR/$workdir/conformers"
        cp -r "$src/results" "$LOCAL_WORK_DIR/$workdir/conformers/"
        cp -r "$src/calculations" "$LOCAL_WORK_DIR/$workdir/conformers/" 2>/dev/null

        # Fixer les chemins dans neutral_rkf_paths.json
        json="$LOCAL_WORK_DIR/$workdir/conformers/results/neutral_rkf_paths.json"
        # Remplacer tout préfixe absolu avant <name>_workdir/ par $LOCAL_WORK_DIR/
        sed -i "s|[^\"]*${workdir}/|${LOCAL_WORK_DIR}/${workdir}/|g" "$json"

        copied=$((copied + 1))
    else
        echo "  ! $name: conformères non trouvés dans $CONFORMERS_DIR"
        missing=$((missing + 1))
    fi
done < <(parse_input_early)

echo "  Copiés:    $copied"
echo "  Manquants: $missing"

if [ "$missing" -gt 0 ]; then
    echo ""
    echo "  ! ATTENTION: $missing molécules sans conformères."
    echo "    Ces molécules seront SKIPPÉES (pas de fallback)."
    echo "    Vérifiez sync_conformers_to_criann.sh"
fi
echo "================================================================================"
echo ""

# =============================================================================
# Validation : vérifier que les adf.rkf sont bien accessibles
# =============================================================================
echo "Validation des chemins RKF..."
valid=0
invalid=0
invalid_names=""

while read -r smiles name; do
    workdir="${name}_workdir"
    json="$LOCAL_WORK_DIR/$workdir/conformers/results/neutral_rkf_paths.json"

    if [ ! -f "$json" ]; then
        invalid=$((invalid + 1))
        invalid_names="$invalid_names $name"
        continue
    fi

    # Vérifier que chaque adf.rkf pointé existe
    all_ok=true
    while IFS= read -r rkf_path; do
        if [ ! -f "$rkf_path" ]; then
            all_ok=false
            echo "  ! $name: RKF manquant: $rkf_path"
            break
        fi
    done < <(grep -o '"[^"]*adf\.rkf"' "$json" | tr -d '"')

    if $all_ok; then
        valid=$((valid + 1))
    else
        invalid=$((invalid + 1))
        invalid_names="$invalid_names $name"
    fi
done < <(parse_input_early)

echo "  Valides:   $valid"
echo "  Invalides: $invalid"

if [ "$invalid" -gt 0 ]; then
    echo "  Molécules invalides:$invalid_names"
fi

if [ "$valid" -eq 0 ]; then
    echo ""
    echo "x ERREUR FATALE: aucune molécule avec des conformères valides !"
    echo "  Les chemins RKF ne pointent vers aucun fichier existant."
    echo "  Vérifiez que sync_conformers_to_criann.sh a bien copié les 03_dft_opt/"
    exit 1
fi
echo ""

# Move to working directory
cd "$LOCAL_WORK_DIR"

# Initialize log files
echo "# PLAMS Predict-DG - Job $SLURM_JOB_ID" > "$SUMMARY_LOG"
echo "# Solvent: $SOLVENT" >> "$SUMMARY_LOG"
echo "# Conformers source: $CONFORMERS_DIR" >> "$SUMMARY_LOG"
echo "# Valid: $valid / Invalid: $invalid" >> "$SUMMARY_LOG"
echo "# Started: $(date)" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"
> "$FAILED_LOG"
> "$TIMING_LOG"

# Parse input file
parse_input() {
    grep -v '^#' "$INPUT_FILE" | grep -v '^[[:space:]]*$'
}

total_molecules=$(parse_input | wc -l)
echo "Processing $valid/$total_molecules molecules (${invalid} skipped - no conformers)"
echo ""

# Counters
success_count=0
failed_count=0
skipped_count=0
current=0

# Global timer
global_start=$(date +%s)

# =============================================================================
# Main processing loop
# =============================================================================
while read -r smiles name; do
    current=$((current + 1))

    # Skip si conformères invalides
    json="${name}_workdir/conformers/results/neutral_rkf_paths.json"
    if [ ! -f "$json" ]; then
        echo "[$current/$total_molecules] $name — SKIP (pas de conformères)"
        echo "$name: SKIPPED (no conformers)" >> "$SUMMARY_LOG"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    echo "================================================================================"
    echo "[$current/$total_molecules] Processing: $name"
    echo "================================================================================"
    echo "SMILES: $smiles"
    echo "Started: $(date)"
    echo ""

    mol_start=$(date +%s)

    # =========================================================================
    # Predict-DG Calculations (3 SP+NormalModes)
    # =========================================================================
    echo "-> Predict-DG Calculations"
    echo "--------------------------------------------------------------------------------"

    amspython redox_calc.py \
        --predict-dg \
        --input-dir "${name}_workdir/conformers/results" \
        --name "$name" \
        --solvent "$SOLVENT"

    mol_end=$(date +%s)
    mol_elapsed=$((mol_end - mol_start))

    if [ $? -ne 0 ]; then
        echo "x ERROR: Predict-DG failed for $name"
        echo "$name: predict_dg_failed" >> "$FAILED_LOG"
        echo "$name: FAILED (${mol_elapsed}s)" >> "$SUMMARY_LOG"
        failed_count=$((failed_count + 1))
    else
        echo "+ SUCCESS: $name completed in ${mol_elapsed}s"
        echo "$name: SUCCESS (${mol_elapsed}s)" >> "$SUMMARY_LOG"
        echo "$name: ${mol_elapsed}s" >> "$TIMING_LOG"
        success_count=$((success_count + 1))
    fi

    echo "Completed: $(date)"
    echo "================================================================================"
    echo ""

done < <(parse_input)

# =============================================================================
# Extraction JSON
# =============================================================================
echo "================================================================================"
echo "EXTRACTION DES DONNEES -> JSON"
echo "================================================================================"

amspython extract_predict_dg.py \
    --solvent "$SOLVENT" \
    --data-dir . \
    --output-dir dataset

echo ""

# =============================================================================
# Global Summary
# =============================================================================
global_end=$(date +%s)
global_elapsed=$((global_end - global_start))
hours=$((global_elapsed / 3600))
minutes=$(((global_elapsed % 3600) / 60))
seconds=$((global_elapsed % 60))

echo ""
echo "================================================================================"
echo "PREDICT-DG TERMINE"
echo "================================================================================"
echo "Solvent:             $SOLVENT"
echo "Total molecules:     $total_molecules"
echo "Successful:          $success_count"
echo "Failed:              $failed_count"
echo "Skipped:             $skipped_count"
echo "Total time:          ${hours}h ${minutes}m ${seconds}s"
echo "================================================================================"
echo ""

# Append summary to log
echo "" >> "$SUMMARY_LOG"
echo "# Completed: $(date)" >> "$SUMMARY_LOG"
echo "# Total: $total_molecules | Success: $success_count | Failed: $failed_count | Skipped: $skipped_count" >> "$SUMMARY_LOG"
echo "# Total time: ${hours}h ${minutes}m ${seconds}s" >> "$SUMMARY_LOG"

if [ $failed_count -gt 0 ]; then
    echo "Failed molecules:"
    cat "$FAILED_LOG"
    echo ""
fi

# =============================================================================
# Copy Results Back
# =============================================================================
echo "Copying dataset back to submission directory..."
RESULTS_DIR="$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

cp -r dataset/ "$RESULTS_DIR/" 2>/dev/null
cp "$SUMMARY_LOG" "$RESULTS_DIR/"
cp "$FAILED_LOG" "$RESULTS_DIR/" 2>/dev/null
cp "$TIMING_LOG" "$RESULTS_DIR/"

echo "+ Dataset copied to $RESULTS_DIR/"
echo ""

echo "================================================================================"
echo "Job completed"
echo "Results directory: $RESULTS_DIR/"
echo "  dataset/$SOLVENT/  -- JSON par molecule et conformere"
echo "================================================================================"
