#!/bin/bash

make_json() {

    local PD=$1
    local OUTFILE="samples_${PD}_2024.json"

    declare -A datasets

    if [[ "$PD" == "EGamma" ]]; then
        datasets[C]="/EGamma0/Run2024C-MINIv6NANOv15-v1/NANOAOD /EGamma1/Run2024C-MINIv6NANOv15-v1/NANOAOD"
        datasets[D]="/EGamma0/Run2024D-MINIv6NANOv15-v1/NANOAOD /EGamma1/Run2024D-MINIv6NANOv15-v1/NANOAOD"
        datasets[E]="/EGamma0/Run2024E-MINIv6NANOv15-v1/NANOAOD /EGamma1/Run2024E-MINIv6NANOv15-v1/NANOAOD"
        datasets[F]="/EGamma0/Run2024F-MINIv6NANOv15-v1/NANOAOD /EGamma1/Run2024F-MINIv6NANOv15-v1/NANOAOD"
        datasets[G]="/EGamma0/Run2024G-MINIv6NANOv15-v2/NANOAOD /EGamma1/Run2024G-MINIv6NANOv15-v2/NANOAOD"
        datasets[H]="/EGamma0/Run2024H-MINIv6NANOv15-v2/NANOAOD /EGamma1/Run2024H-MINIv6NANOv15-v1/NANOAOD"
        datasets[I]="/EGamma0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD /EGamma1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD"
    else
        datasets[C]="/Muon0/Run2024C-MINIv6NANOv15-v1/NANOAOD /Muon1/Run2024C-MINIv6NANOv15-v1/NANOAOD"
        datasets[D]="/Muon0/Run2024D-MINIv6NANOv15-v1/NANOAOD /Muon1/Run2024D-MINIv6NANOv15-v1/NANOAOD"
        datasets[E]="/Muon0/Run2024E-MINIv6NANOv15-v1/NANOAOD /Muon1/Run2024E-MINIv6NANOv15-v1/NANOAOD"
        datasets[F]="/Muon0/Run2024F-MINIv6NANOv15-v1/NANOAOD /Muon1/Run2024F-MINIv6NANOv15-v1/NANOAOD"
        datasets[G]="/Muon0/Run2024G-MINIv6NANOv15-v1/NANOAOD /Muon1/Run2024G-MINIv6NANOv15-v2/NANOAOD"
        datasets[H]="/Muon0/Run2024H-MINIv6NANOv15-v2/NANOAOD /Muon1/Run2024H-MINIv6NANOv15-v1/NANOAOD"
        datasets[I]="/Muon0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD /Muon1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD"
    fi

    echo "{" > "$OUTFILE"

    firstEra=true

    for era in C D E F G H I; do

        $firstEra || echo "," >> "$OUTFILE"
        firstEra=false

        echo "  \"Data-2024${era}\": [" >> "$OUTFILE"

        firstFile=true

        for dataset in ${datasets[$era]}; do
            while read -r file; do

                [[ -z "$file" ]] && continue

                file="root://cms-xrd-global.cern.ch/$file"

                $firstFile || echo "," >> "$OUTFILE"
                firstFile=false

                printf '    "%s"' "$file" >> "$OUTFILE"

            done < <(dasgoclient -query="file dataset=${dataset}")
        done

        echo "" >> "$OUTFILE"
        echo -n "  ]" >> "$OUTFILE"

    done

    echo "" >> "$OUTFILE"
    echo "}" >> "$OUTFILE"

    echo "Created $OUTFILE"
}

make_json EGamma
make_json Muon
