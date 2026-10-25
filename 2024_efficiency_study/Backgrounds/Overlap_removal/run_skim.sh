#!/bin/bash

set -e

echo "============================================================"
echo "Starting skim_to_parquet Condor job"
echo "============================================================"

echo "Hostname : $(hostname)"
echo "PWD      : $(pwd)"
echo "Date     : $(date)"

# ------------------------------------------------------------
# CMS / LCG environment
# ------------------------------------------------------------

source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh

# ------------------------------------------------------------
# Python environment
# ------------------------------------------------------------

export PYTHONPATH="$HOME/.local/lib/python3.13/site-packages/:$PYTHONPATH"
export PATH="$HOME/.local/bin/:$PATH"

echo "Python   : $(which python3)"
python3 --version

echo "xrdcp    : $(which xrdcp)"
xrdcp --version || true

# ------------------------------------------------------------
# Arguments
#
# $1 = ROOT input file
# $2 = local parquet output
# $3 = EOS remote parquet output
# $4 = process name
#
# ------------------------------------------------------------

ROOT_FILE="$1"
LOCAL_OUTPUT="$2"
EOS_OUTPUT="$3"
PROCESS="$4"

echo "------------------------------------------------------------"
echo "ROOT file    : ${ROOT_FILE}"
echo "Local output : ${LOCAL_OUTPUT}"
echo "EOS output   : ${EOS_OUTPUT}"
echo "Process      : ${PROCESS}"
echo "------------------------------------------------------------"

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

if [ ! -f "${ROOT_FILE}" ]; then
    echo "ERROR: ROOT file does not exist:"
    echo "${ROOT_FILE}"
    exit 1
fi

# ------------------------------------------------------------
# Run skim_to_parquet.py
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Running skim_to_parquet.py"
echo "------------------------------------------------------------"

python3 skim_to_parquet.py \
    --input-file "${ROOT_FILE}" \
    --output-file "${LOCAL_OUTPUT}" \
    --process "${PROCESS}"

# ------------------------------------------------------------
# Check output
# ------------------------------------------------------------

if [ ! -f "${LOCAL_OUTPUT}" ]; then
    echo "ERROR: skim_to_parquet.py did not produce:"
    echo "${LOCAL_OUTPUT}"
    exit 1
fi

FILE_SIZE=$(stat -c%s "${LOCAL_OUTPUT}")

echo "Output file created:"
echo "${LOCAL_OUTPUT}"
echo "Size: ${FILE_SIZE} bytes"

if [ "${FILE_SIZE}" -eq 0 ]; then
    echo "ERROR: Output parquet is empty"
    exit 1
fi

# ------------------------------------------------------------
# Create EOS destination directory
# ------------------------------------------------------------

EOS_DIR=$(dirname "${EOS_OUTPUT}")

echo "------------------------------------------------------------"
echo "Creating EOS output directory"
echo "${EOS_DIR}"
echo "------------------------------------------------------------"

xrdfs eosuser.cern.ch mkdir -p "${EOS_DIR#root://eosuser.cern.ch/}"

# ------------------------------------------------------------
# Transfer parquet to EOS
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Transferring output to EOS"
echo "------------------------------------------------------------"

xrdcp \
    --force \
    "${LOCAL_OUTPUT}" \
    "${EOS_OUTPUT}"

# ------------------------------------------------------------
# Verify EOS file
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Verifying EOS output"
echo "------------------------------------------------------------"

xrdfs eosuser.cern.ch stat \
    "${EOS_OUTPUT#root://eosuser.cern.ch/}"

echo "============================================================"
echo "JOB COMPLETED SUCCESSFULLY"
echo "EOS output:"
echo "${EOS_OUTPUT}"
echo "============================================================"

exit 0