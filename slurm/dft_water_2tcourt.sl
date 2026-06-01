#!/bin/bash

#SBATCH -p 2tcourt
#SBATCH -N 1
#SBATCH -n 192
#SBATCH --time 12:00:00
#SBATCH --mem=0

#SBATCH --output summary.%j
#SBATCH --error error.%j

################################################################################
# PlamsV5 DFT (Water re-run, backfill-friendly) : conformers -> redox -> JSON
# One chunk of (typically 2) small molecules per job, 1 full node, 2tcourt 12h.
# Designed to backfill into short gaps rather than wait for a single long block.
# Each molecule's workdir is copied back to the submit dir AS SOON AS it finishes,
# so a 12h timeout loses at most the in-progress molecule (resume-safe on re-run).
################################################################################

# =============================================================================
# CONFIGURATION
# =============================================================================
SOLVENT="${SOLVENT:-Water}"
INPUT_FILE="${INPUT_FILE:-input_w2t_01.txt}"
CLUSTER_CONTINUUM="no"

# AMS environment
module purge
module load atomic_simu/cobra-ams/2025.106_amd
module load py_env/python/3.11.7
module list
echo ""

# Disable GPU
export SCM_GPUENABLED="FALSE"

# Cray MPICH workaround: some nodes (e.g. c23fin*) have CMA locked down, so the
# default single-copy shared-memory transport aborts ams.exe at startup
# ("CMA does not have sufficient permission to operate / process_vm_readv:
# Operation not permitted"). Disabling single-copy mode fixes it everywhere.
export MPICH_SMP_SINGLE_COPY_MODE=NONE

# Logging
SUMMARY_LOG="summary.log"
FAILED_LOG="failed_molecules.txt"
TIMING_LOG="timing.log"

# Per-job result dir in the submit dir
RESULTS_DIR="$SLURM_SUBMIT_DIR/Water_missing_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "================================================================================"
echo "PLAMSV5 DFT WATER RE-RUN (2tcourt backfill) - Job ID: $SLURM_JOB_ID"
echo "================================================================================"
echo "Configuration:"
echo "  Solvent:         $SOLVENT"
echo "  Input File:      $INPUT_FILE"
echo "  Nodes:           1 (192 cores)"
echo "  Partition:       2tcourt (12h)"
echo "  Mode:            DFT complet (7 calculs/conformere, TZ2P+DZP)"
echo "================================================================================"
echo ""

# Save a molecule's workdir back to the submit dir (called per-molecule).
save_workdir() {
    local name="$1"
    local workdir="${name}_workdir"
    if [ -d "$LOCAL_WORK_DIR/$workdir" ]; then
        echo "  -> Saving $workdir to submit dir..."
        rm -rf "$SLURM_SUBMIT_DIR/$workdir"
        cp -r "$LOCAL_WORK_DIR/$workdir" "$SLURM_SUBMIT_DIR/"
    fi
}

# Copy code to local work directory
echo "Setting up working directory..."
cp "$SLURM_SUBMIT_DIR/$INPUT_FILE" "$LOCAL_WORK_DIR/"
cp "$SLURM_SUBMIT_DIR/conformers_gen.py" "$LOCAL_WORK_DIR"
cp "$SLURM_SUBMIT_DIR/redox_calc.py" "$LOCAL_WORK_DIR"
cp "$SLURM_SUBMIT_DIR/extract_predict_dg.py" "$LOCAL_WORK_DIR"
cp -r "$SLURM_SUBMIT_DIR/common/" "$LOCAL_WORK_DIR"
cp -r "$SLURM_SUBMIT_DIR/qmconf/" "$LOCAL_WORK_DIR"
cp -r "$SLURM_SUBMIT_DIR/qmredox/" "$LOCAL_WORK_DIR"
echo "  Files copied to $LOCAL_WORK_DIR"
echo ""

# Copy existing workdir (resume partial runs)
parse_input_early() {
    grep -v '^#' "$SLURM_SUBMIT_DIR/$INPUT_FILE" | grep -v '^[[:space:]]*$'
}

echo "Copying existing workdirs..."
while read -r smiles name; do
    workdir="${name}_workdir"
    if [ -d "$SLURM_SUBMIT_DIR/$workdir" ]; then
        echo "  -> Resuming $workdir..."
        cp -r "$SLURM_SUBMIT_DIR/$workdir" "$LOCAL_WORK_DIR/"
    else
        echo "  -> Fresh start for $name (no existing workdir)"
    fi
done < <(parse_input_early)
echo ""

cd "$LOCAL_WORK_DIR"

# From here on, the input file lives at the top of LOCAL_WORK_DIR (basename only).
INPUT_FILE="$(basename "$INPUT_FILE")"

# Init logs
echo "# PlamsV5 DFT Water re-run (2tcourt) - Job $SLURM_JOB_ID" > "$SUMMARY_LOG"
echo "# Solvent: $SOLVENT | Input: $INPUT_FILE" >> "$SUMMARY_LOG"
echo "# Started: $(date)" >> "$SUMMARY_LOG"
> "$FAILED_LOG"
> "$TIMING_LOG"

parse_input() {
    grep -v '^#' "$INPUT_FILE" | grep -v '^[[:space:]]*$'
}

total_molecules=$(parse_input | wc -l)
echo "Found $total_molecules molecule(s) to process"
echo ""

success_count=0
failed_count=0
current=0
global_start=$(date +%s)

# =============================================================================
# Main loop
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

    # ------------------------------------------------------------------------
    # STEP 1: Conformers (skip if already present)
    # ------------------------------------------------------------------------
    if [ -f "${name}_workdir/conformers/results/neutral_rkf_paths.json" ]; then
        echo "-> Step 1/2: Conformers already present, skipping"
        echo "+ Reusing ${name}_workdir/conformers/results/"
        echo ""
    else
        echo "-> Step 1/2: Conformers Generation"
        amspython conformers_gen.py \
            --name "$name" \
            --smiles "$smiles" \
            --solvent "$SOLVENT"

        if [ $? -ne 0 ]; then
            echo "x ERROR: Conformers generation failed for $name"
            echo "$name: conformers_failed" >> "$FAILED_LOG"
            molecule_success=false
        else
            echo "+ Conformers generation completed"
        fi
        echo ""
    fi

    # ------------------------------------------------------------------------
    # STEP 2: Redox (TZ2P+DZP)
    # ------------------------------------------------------------------------
    if [ "$molecule_success" = true ]; then
        echo "-> Step 2/2: Redox Calculations"
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

    mol_end=$(date +%s)
    mol_elapsed=$((mol_end - mol_start))

    if [ "$molecule_success" = true ]; then
        echo "+ SUCCESS: $name completed in ${mol_elapsed}s"
        echo "$name: SUCCESS (${mol_elapsed}s)" >> "$SUMMARY_LOG"
        echo "$name: ${mol_elapsed}s" >> "$TIMING_LOG"
        success_count=$((success_count + 1))
    else
        echo "x FAILED: $name"
        echo "$name: FAILED (${mol_elapsed}s)" >> "$SUMMARY_LOG"
        failed_count=$((failed_count + 1))
    fi

    # Save this molecule's workdir immediately (timeout-safe).
    save_workdir "$name"

    echo "Completed: $(date)"
    echo "================================================================================"
    echo ""

done < <(parse_input)

# =============================================================================
# STEP 3: Extract JSON (best effort; on timeout, re-run resumes & re-extracts)
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
# Summary
# =============================================================================
global_end=$(date +%s)
global_elapsed=$((global_end - global_start))
hours=$((global_elapsed / 3600))
minutes=$(((global_elapsed % 3600) / 60))
seconds=$((global_elapsed % 60))

echo "================================================================================"
echo "DFT WATER RE-RUN (2tcourt) TERMINE"
echo "================================================================================"
echo "Solvent:             $SOLVENT"
echo "Input:               $INPUT_FILE"
echo "Successful:          $success_count"
echo "Failed:              $failed_count"
echo "Total time:          ${hours}h ${minutes}m ${seconds}s"
echo "================================================================================"
echo ""

echo "" >> "$SUMMARY_LOG"
echo "# Completed: $(date)" >> "$SUMMARY_LOG"
echo "# Success: $success_count | Failed: $failed_count" >> "$SUMMARY_LOG"
echo "# Total: ${hours}h ${minutes}m ${seconds}s" >> "$SUMMARY_LOG"

# =============================================================================
# Copy back: dataset + logs (workdirs already saved per-molecule)
# =============================================================================
echo "Copying results back..."
cp -r dataset/ "$RESULTS_DIR/" 2>/dev/null
cp "$SUMMARY_LOG" "$RESULTS_DIR/"
cp "$FAILED_LOG" "$RESULTS_DIR/" 2>/dev/null
cp "$TIMING_LOG" "$RESULTS_DIR/"

echo "+ Results in $RESULTS_DIR/"
echo "================================================================================"
