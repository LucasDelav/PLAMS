#!/bin/bash

#SBATCH -p long
#SBATCH -N 2
#SBATCH -n 384
#SBATCH --time 72:00:00
#SBATCH --mem=0

#SBATCH --output summary.%j
#SBATCH --error error.%j

################################################################################
# PLAMS workflow DFT: conformers → redox complet (TZ2P+DZP) → extraction JSON
# Reads molecules from input.txt (format: SMILES name)
# Runs full redox pipeline (7 calculations/conformer)
# Extracts all data into JSON dataset
################################################################################

# =============================================================================
# CONFIGURATION
# =============================================================================
SOLVENT="${SOLVENT:-Acetonitrile}"
INPUT_FILE="${INPUT_FILE:-input.txt}"
CLUSTER_CONTINUUM="no"   # "yes" for protic solvents (Water, Methanol)

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
echo "PLAMS DFT COMPLET - Job ID: $SLURM_JOB_ID"
echo "================================================================================"
echo "Configuration:"
echo "  Solvent:         $SOLVENT"
echo "  Input File:      $INPUT_FILE"
echo "  Cluster-cont.:   $CLUSTER_CONTINUUM"
echo "  Mode:            DFT complet (7 calculs/conformere, TZ2P+DZP)"
echo "  Defaults:        PBE0/DZP, refinement TZ2P, Boltzmann 0.10"
echo "================================================================================"
echo ""

# Copy code to local work directory
echo "Setting up working directory..."
cp "$SLURM_SUBMIT_DIR/$INPUT_FILE" "$LOCAL_WORK_DIR"
cp "$SLURM_SUBMIT_DIR/conformers_gen.py" "$LOCAL_WORK_DIR"
cp "$SLURM_SUBMIT_DIR/redox_calc.py" "$LOCAL_WORK_DIR"
cp "$SLURM_SUBMIT_DIR/extract_predict_dg.py" "$LOCAL_WORK_DIR"
cp -r "$SLURM_SUBMIT_DIR/common/" "$LOCAL_WORK_DIR"
cp -r "$SLURM_SUBMIT_DIR/qmconf/" "$LOCAL_WORK_DIR"
cp -r "$SLURM_SUBMIT_DIR/qmredox/" "$LOCAL_WORK_DIR"
echo "  Files copied to $LOCAL_WORK_DIR"
echo ""

# Copy existing workdirs (for resuming interrupted runs)
echo "Copying existing workdirs from submission directory..."
parse_input_early() {
    grep -v '^#' "$SLURM_SUBMIT_DIR/$INPUT_FILE" | grep -v '^[[:space:]]*$'
}

while read -r smiles name; do
    workdir="${name}_workdir"
    if [ -d "$SLURM_SUBMIT_DIR/$workdir" ]; then
        echo "  -> Copying $workdir..."
        cp -r "$SLURM_SUBMIT_DIR/$workdir" "$LOCAL_WORK_DIR/"
    fi
done < <(parse_input_early)
echo "  Workdirs copied"
echo ""

# Move to working directory
cd "$LOCAL_WORK_DIR"

# Initialize log files
echo "# PLAMS DFT Complet - Job $SLURM_JOB_ID" > "$SUMMARY_LOG"
echo "# Solvent: $SOLVENT" >> "$SUMMARY_LOG"
echo "# Mode: DFT complet (TZ2P+DZP)" >> "$SUMMARY_LOG"
echo "# Started: $(date)" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"
> "$FAILED_LOG"
> "$TIMING_LOG"

# Parse input file
parse_input() {
    grep -v '^#' "$INPUT_FILE" | grep -v '^[[:space:]]*$'
}

total_molecules=$(parse_input | wc -l)
echo "Found $total_molecules molecule(s) to process"
echo ""

# Counters
success_count=0
failed_count=0
current=0

# Global timer
global_start=$(date +%s)

# =============================================================================
# Main processing loop
# =============================================================================
while read -r smiles name; do
    current=$((current + 1))

    echo "================================================================================"
    echo "[$current/$total_molecules] Processing: $name"
    echo "================================================================================"
    echo "SMILES: $smiles"
    echo "Started: $(date)"
    echo ""

    mol_start=$(date +%s)
    molecule_success=true

    # =========================================================================
    # STEP 1: Conformers Generation
    # =========================================================================
    echo "-> Step 1/2: Conformers Generation"
    echo "--------------------------------------------------------------------------------"

    CC_FLAG=""
    if [ "$CLUSTER_CONTINUUM" = "yes" ]; then
        CC_FLAG="--cluster-continuum"
    fi

    amspython conformers_gen.py \
        --name "$name" \
        --smiles "$smiles" \
        --solvent "$SOLVENT" \
        $CC_FLAG

    if [ $? -ne 0 ]; then
        echo "x ERROR: Conformers generation failed for $name"
        echo "$name: conformers_failed" >> "$FAILED_LOG"
        molecule_success=false
    else
        echo "+ Conformers generation completed"
    fi
    echo ""

    # =========================================================================
    # STEP 2: Redox Calculations (full TZ2P+DZP composite)
    # =========================================================================
    if [ "$molecule_success" = true ]; then
        echo "-> Step 2/2: Redox Calculations (DFT complet)"
        echo "--------------------------------------------------------------------------------"

        amspython redox_calc.py \
            --input-dir "${name}_workdir/conformers/results" \
            --name "$name" \
            --solvent "$SOLVENT"

        if [ $? -ne 0 ]; then
            echo "x ERROR: Redox calculations failed for $name"
            echo "$name: redox_failed" >> "$FAILED_LOG"
            molecule_success=false
        else
            echo "+ Redox calculations completed"
        fi
        echo ""
    fi

    # =========================================================================
    # Molecule summary
    # =========================================================================
    mol_end=$(date +%s)
    mol_elapsed=$((mol_end - mol_start))

    if [ "$molecule_success" = true ]; then
        echo "+ SUCCESS: $name completed in ${mol_elapsed}s"
        echo "$name: SUCCESS (${mol_elapsed}s)" >> "$SUMMARY_LOG"
        echo "$name: ${mol_elapsed}s" >> "$TIMING_LOG"
        success_count=$((success_count + 1))
    else
        echo "x FAILED: $name (see logs for details)"
        echo "$name: FAILED (${mol_elapsed}s)" >> "$SUMMARY_LOG"
        failed_count=$((failed_count + 1))
    fi

    echo "Completed: $(date)"
    echo "================================================================================"
    echo ""

done < <(parse_input)

# =============================================================================
# STEP 3: Extract all data into JSON dataset
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
echo "DFT COMPLET TERMINE"
echo "================================================================================"
echo "Solvent:             $SOLVENT"
echo "Total molecules:     $total_molecules"
echo "Successful:          $success_count"
echo "Failed:              $failed_count"
echo "Total time:          ${hours}h ${minutes}m ${seconds}s"
echo "================================================================================"
echo ""

# Append summary to log
echo "" >> "$SUMMARY_LOG"
echo "# Completed: $(date)" >> "$SUMMARY_LOG"
echo "# Total: $total_molecules | Success: $success_count | Failed: $failed_count" >> "$SUMMARY_LOG"
echo "# Total time: ${hours}h ${minutes}m ${seconds}s" >> "$SUMMARY_LOG"

if [ $failed_count -gt 0 ]; then
    echo "Failed molecules:"
    cat "$FAILED_LOG"
    echo ""
fi

# =============================================================================
# Copy Results Back (dataset only)
# =============================================================================
echo "Copying dataset back to submission directory..."
RESULTS_DIR="$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Copy JSON dataset
cp -r dataset/ "$RESULTS_DIR/" 2>/dev/null

# Copy logs
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
