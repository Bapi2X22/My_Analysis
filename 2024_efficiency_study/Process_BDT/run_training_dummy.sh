#!/bin/bash

echo "======================================"
echo "CONDOR DIAGNOSTIC TEST"
echo "======================================"

echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "PWD: $PWD"

echo "Cluster ID: ${_CONDOR_CLUSTER_ID:-undefined}"
echo "Process ID: ${_CONDOR_PROCNO:-undefined}"
echo "Scratch: ${_CONDOR_SCRATCH_DIR:-undefined}"

echo ""
echo "Environment variables:"
env | sort

echo ""
echo "Files in working directory:"
ls -lah

echo ""
echo "Testing Python:"
which python3
python3 --version

echo ""
echo "Testing successful execution..."

sleep 5

echo "CONDOR DIAGNOSTIC TEST COMPLETED"

exit 0

