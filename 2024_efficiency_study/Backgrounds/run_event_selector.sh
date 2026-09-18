#!/bin/bash

set -e

echo "============================================================"
echo "Starting event_selector Condor job"
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
# $1 = file list
# $2 = local parquet output
# $3 = EOS remote parquet output
# ------------------------------------------------------------

FILE_LIST="$1"
LOCAL_OUTPUT="$2"
EOS_OUTPUT="$3"

echo "------------------------------------------------------------"
echo "File list    : ${FILE_LIST}"
echo "Local output : ${LOCAL_OUTPUT}"
echo "EOS output   : ${EOS_OUTPUT}"
echo "------------------------------------------------------------"

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

if [ ! -f "${FILE_LIST}" ]; then
    echo "ERROR: File list does not exist:"
    echo "${FILE_LIST}"
    exit 1
fi

NFILES=$(wc -l < "${FILE_LIST}")

echo "Number of ROOT files in this job: ${NFILES}"

if [ "${NFILES}" -eq 0 ]; then
    echo "ERROR: Empty file list"
    exit 1
fi

# ------------------------------------------------------------
# Run event selector
#
# event_selector.py already supports:
#
# --worker
# --file-list
# --worker-output
#
# ------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Running event_selector.py"
echo "------------------------------------------------------------"

python3 event_selector.py \
    --worker \
    --file-list "${FILE_LIST}" \
    --worker-output "${LOCAL_OUTPUT}"

# ------------------------------------------------------------
# Check output
# ------------------------------------------------------------

if [ ! -f "${LOCAL_OUTPUT}" ]; then
    echo "ERROR: event_selector.py did not produce:"
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
