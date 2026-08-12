#!/bin/bash

BASE_DIR="/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/NTuples_WH_2024_HDNA_presel_with_latest_BDT_score"

for MASS in 12 15 20 25 30 35 40 45 50 55 60
do
    echo "========================================"
    echo "Processing mass point M${MASS}"
    echo "========================================"

    prepare_output_file.py \
        --input "${BASE_DIR}" \
        --dataset "WH-2024M${MASS}" \
        --root \
        --batch local \
        --output "${BASE_DIR}" \
        --cats \
        --catDict "${BASE_DIR}/CATS/cat_M${MASS}.json" \
        --syst \
        --varDict "${BASE_DIR}/variation.json" \
        --verbose True

    echo
done

echo "All mass points processed."
