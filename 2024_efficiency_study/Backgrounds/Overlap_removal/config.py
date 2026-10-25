import os

# General configuration

BACKGROUND_DIR = "../Backgrounds_merged"

PARQUET_BACKGROUND_DIR = "../merged_compact_parquet"

OUTPUT_DIR = (
    "/eos/user/b/bbapi/www/Analysis_plots/"
    "Overlap_removal/FINAL6/"
)

LUMI_FB = 109.0

# Photon / GenPart selection

PHOTON_PT_MIN = 10.0
PHOTON_ETA_MAX = 3.0

OTHER_PT_MIN = 5.0

DR_CUT = 0.15
DR_MAX = 1.0

# Processes

PROCESSES = {
    "DYGto2LG": {
        "files": [
            "DYGto2LG4.parquet",
            "DYGto2LG50.parquet",
        ],
        "xsec": [88.13, 126.7],
    },

    "DYto2E": {
        "files": [
            "DYto2E10.parquet",
            "DYto2E50.parquet",
        ],
        "xsec": [21140.0, 2124.08],
    },

    "DYto2Mu": {
        "files": [
            "DYto2Mu10.parquet",
            "DYto2Mu50.parquet",
        ],
        "xsec": [21190.0, 2124.08],
    },

    "TTto2L2Nu": {
        "files": [
            "TTto2L2Nu.parquet",
        ],
        "xsec": 98.04,
    },

    "TTtoLNu2Q": {
        "files": [
            "TTtoLNu2Q.parquet",
        ],
        "xsec": 405.87,
    },

    "TTG1Jets": {
        "files": [
            "TTG1Jets.parquet",
        ],
        "xsec": 4.634,
    },

    "WGtoLNuG": {
        "files": [
            "WGtoLNuG.parquet",
        ],
        "xsec": 671.5,
    },

    "WtoENu": {
        "files": [
            "WtoENu0J.parquet",
            "WtoENu1J.parquet",
            "WtoENu2J.parquet",
        ],
        "xsec": [55850, 9177, 3474]
        },

    "WtoMuNu": {
        "files": [
            "WtoMuNu0J.parquet",
            "WtoMuNu1J.parquet",
            "WtoMuNu2J.parquet",
        ],
        "xsec": [55920, 9202, 3490]
        },
}


# Process groups

PROCESS_GROUPS = {

    "DY": [
        "DYGto2LG",
        "DYto2E",
        "DYto2Mu"
    ],

    "TOP": [
        "TTG1Jets",
        "TTto2L2Nu",
        "TTtoLNu2Q"
    ],

    "WJetsG": [
        "WGtoLNuG",
        "WtoENu",
        "WtoMuNu"
    ],

    "ALL_BACKGROUND": [
        "DYGto2LG",
        "DYto2E",
        "DYto2Mu",
        "TTto2L2Nu",
        "TTtoLNu2Q",
        "TTG1Jets",
        "WGtoLNuG",
        "WtoENu",
        "WtoMuNu"
    ],
}
