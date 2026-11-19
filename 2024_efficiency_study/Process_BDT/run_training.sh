#!/bin/bash

# ============================================================
#              Developed by Bapi Basak
#                    IISER Pune
#                  November 2026
# ============================================================

# set -euo pipefail

# ============================================================
# Basic information
# ============================================================

echo "============================================================"
echo "           XGBoost BDT Training"
echo "============================================================"

echo "Hostname : $(hostname)"
echo "Date     : $(date)"
echo "Job ID   : ${_CONDOR_CLUSTER_ID:-unknown}.${_CONDOR_PROCNO:-unknown}"

BDT_Folder="$1"
Signal_File="$2"
Background_File="$3"
# Mass_Value="$4"
Mass_Value="${4:-None}"

# ============================================================
# Temporary working directory
# ============================================================

WORKDIR="${_CONDOR_SCRATCH_DIR:-$PWD}"

cd "$WORKDIR"

echo "Working directory: $WORKDIR"

# ============================================================
# Setup environment
# ============================================================

source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh

export PYTHONPATH="$HOME/.local/lib/python3.13/site-packages/:${PYTHONPATH:-}"
export PATH="$HOME/.local/bin/:$PATH"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

# ============================================================
# Check environment
# ============================================================

echo "Python:"
which python3
python3 --version

echo "Checking required packages..."

python3 -c "
import ROOT
import uproot
import awkward
import xgboost
import optuna
import sklearn
import pandas
import matplotlib

print('All required packages imported successfully.')
"

# ============================================================
# Input files
# ============================================================

echo "Checking input files..."

ls -lh Merged_Inputs/

test -s Merged_Inputs/merged_signal_allCat_equalized_weight.root
test -s Merged_Inputs/merged_bkg_allCat.root

# ============================================================
# Output directory
# ============================================================

LOCAL_OUTPUT_DIR="${WORKDIR}/Model_output"

mkdir -p "$LOCAL_OUTPUT_DIR"

export BDT_OUTPUT_DIR="$LOCAL_OUTPUT_DIR"

JOB_ID="${_CONDOR_CLUSTER_ID:-0}_${_CONDOR_PROCNO:-0}"

# ============================================================
# Run training
# ============================================================

echo
echo "============================================================"
echo "Starting XGBoost training"
echo "============================================================"

# python3 xgboost_TrainMVA_v2_withWeight_mod_condor.py \
#     --useMC \
#     --optimize \
#     --trials 100 \
#     -m "${LOCAL_OUTPUT_DIR}/${BDT_Folder}.pkl" \
#     --signal_path "$Signal_File" \
#     --background_path "$Background_File" \
#     --mass_value "$Mass_Value"

CMD=(
    python3 xgboost_TrainMVA_v2_withWeight_mod_condor.py
    --useMC
    --optimize
    --trials 100
    -m "${LOCAL_OUTPUT_DIR}/${BDT_Folder}.pkl"
    --signal_path "$Signal_File"
    --background_path "$Background_File"
)

if [[ "$Mass_Value" != "None" ]]; then
    CMD+=(--mass_value "$Mass_Value")
fi

"${CMD[@]}"


echo
echo "Training completed successfully."

# ============================================================
# Check output
# ============================================================

echo "Generated output files:"

find "$LOCAL_OUTPUT_DIR" -maxdepth 2 -type f -printf '%p %s bytes\n'

if ! find "$LOCAL_OUTPUT_DIR" -maxdepth 1 -type f -name "*.pkl" | grep -q .; then
    echo "ERROR: Model pickle file is missing."
    exit 1
fi

# ============================================================
# Copy results to EOS
# ============================================================

EOS_BASE="/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Process_BDT/Model_output"

REMOTE_BASE="root://eosuser.cern.ch/${EOS_BASE}"

REMOTE_OUTPUT="${EOS_BASE}/${BDT_Folder}"
REMOTE_URL="root://eosuser.cern.ch/${REMOTE_OUTPUT}"

echo
echo "Copying training outputs to EOS..."

xrdfs root://eosuser.cern.ch mkdir -p "$REMOTE_OUTPUT"

while IFS= read -r -d '' FILE
do

    RELATIVE_PATH="${FILE#${LOCAL_OUTPUT_DIR}/}"

    REMOTE_DIR="$(dirname "${REMOTE_OUTPUT}/${RELATIVE_PATH}")"

    xrdfs root://eosuser.cern.ch mkdir -p "$REMOTE_DIR"

    echo "Copying: $RELATIVE_PATH"

    xrdcp -f \
        "$FILE" \
        "${REMOTE_URL}/${RELATIVE_PATH}"

done < <(find "$LOCAL_OUTPUT_DIR" -type f -print0)

# ============================================================
# Verify EOS output
# ============================================================

echo
echo "Verifying EOS output..."

xrdfs root://eosuser.cern.ch ls "$REMOTE_OUTPUT"

echo
echo "============================================================"
echo "              TRAINING JOB SUCCESS"
echo "============================================================"

echo "Job ID      : $JOB_ID"
echo "EOS output  : $REMOTE_URL"
echo "Completed   : $(date)"

echo "============================================================"
