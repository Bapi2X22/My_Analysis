#!/bin/bash

# List of correction configs
configs=(
 "FNUF"
 "Material"
 "jec_jet_syst"
 "ScaleEB2G_IJazZ"
 "ScaleEE2G_IJazZ"
 "Smearing2G_IJazZ"
 "Pileup"
 "SF_photon_ID"
 "ElectronVetoSF"
 "PreselSF"
 "TriggerSF"
 "energyErrShift"
 "PartonShower"
)

# Base paths
JSON_PREFIX="my_runner_WH_2024"
DUMP_BASE="../../NTuples_WH_2024_single_correction_check_HDNA"

# Loop over configs
for cfg in "${configs[@]}"; do
  echo "Running for config: $cfg"

    run_analysis.py \
    --json-analysis ${JSON_PREFIX}_${cfg}.json \
    --dump ${DUMP_BASE}_${cfg}/ \
    --no-trigger \
    --skipbadfiles \
    --nano-version 15 \
    --skipbadfiles

done

echo "All jobs finished!"
