import os

# General configuration

BACKGROUND_DIR = "../Backgrounds_merged"

OUTPUT_DIR = (
    "/eos/user/b/bbapi/www/Analysis_plots/"
    "Overlap_removal/FINAL2/"
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

# ROOT input
ROOT_BACKGROUND_DIR = "../Skimmer/"

ROOT_PROCESSES = {

    "DYGto2LG": {
        "directories": [
            "DYGto2LG4_24SummerRun3",
            "DYGto2LG50_24SummerRun3",
        ],
        "n_files": [20, 10],
        "xsec": [88.13, 126.7],
    },

    "DYto2E": {
        "directories": [
            "DYto2E10_24SummerRun3",
            "DYto2E50_24SummerRun3",
        ],
        "n_files": [100, 10],
        "xsec": [21140.0, 2124.08],
    },

    "DYto2Mu": {
        "directories": [
            "DYto2Mu10_24SummerRun3",
            "DYto2Mu50_24SummerRun3",
        ],
        "n_files": [100, 10],
        "xsec": [21190.0, 2124.08],
    },

    "TTto2L2Nu": {
        "directories": [
            "TTto2L2Nu_24SummerRun3",
        ],
        "n_files": [5],
        "xsec": [98.04],
    },

    "TTtoLNu2Q": {
        "directories": [
            "TTtoLNu2Q_24SummerRun3",
        ],
        "n_files": [5],
        "xsec": [405.87],
    },

    "TTG1Jets": {
        "directories": [
            "TTG1Jets_24SummerRun3",
        ],
        "n_files": [5],
        "xsec": [4.634],
    },

    "WGtoLNuG": {
        "directories": [
            "WGtoLNuG_24SummerRun3",
        ],
        "n_files": [20],
        "xsec": [671.5],
    },

    "WtoENu": {
        "directories": [
            "WtoENu0J_24SummerRun3",
            "WtoENu1J_24SummerRun3",
            "WtoENu2J_24SummerRun3",
        ],
        "n_files": [200, 200, 200],
        "xsec": [55850, 9177, 3474],
    },

    "WtoMuNu": {
        "directories": [
            "WtoMuNu0J_24SummerRun3",
            "WtoMuNu1J_24SummerRun3",
            "WtoMuNu2J_24SummerRun3",
        ],
        "n_files": [200, 200, 200],
        "xsec": [55920, 9202, 3490],
    },
}

# ROOT_PROCESSES = {

#     "DYGto2LG": {
#         "directories": [
#             "DYGto2LG4_24SummerRun3",
#             "DYGto2LG50_24SummerRun3",
#         ],
#         "n_files": [2, 2],
#         "xsec": [88.13, 126.7],
#     },

#     "DYto2E": {
#         "directories": [
#             "DYto2E10_24SummerRun3",
#             "DYto2E50_24SummerRun3",
#         ],
#         "n_files": [10, 10],
#         "xsec": [21140.0, 2124.08],
#     },

#     "DYto2Mu": {
#         "directories": [
#             "DYto2Mu10_24SummerRun3",
#             "DYto2Mu50_24SummerRun3",
#         ],
#         "n_files": [10, 10],
#         "xsec": [21190.0, 2124.08],
#     },

#     "TTto2L2Nu": {
#         "directories": [
#             "TTto2L2Nu_24SummerRun3",
#         ],
#         "n_files": [2],
#         "xsec": [98.04],
#     },

#     "TTtoLNu2Q": {
#         "directories": [
#             "TTtoLNu2Q_24SummerRun3",
#         ],
#         "n_files": [2],
#         "xsec": [405.87],
#     },

#     "TTG1Jets": {
#         "directories": [
#             "TTG1Jets_24SummerRun3",
#         ],
#         "n_files": [2],
#         "xsec": [4.634],
#     },

#     "WGtoLNuG": {
#         "directories": [
#             "WGtoLNuG_24SummerRun3",
#         ],
#         "n_files": [2],
#         "xsec": [671.5],
#     },

#     "WtoENu": {
#         "directories": [
#             "WtoENu0J_24SummerRun3",
#             "WtoENu1J_24SummerRun3",
#             "WtoENu2J_24SummerRun3",
#         ],
#         "n_files": [10, 10, 10],
#         "xsec": [55850, 9177, 3474],
#     },

#     "WtoMuNu": {
#         "directories": [
#             "WtoMuNu0J_24SummerRun3",
#             "WtoMuNu1J_24SummerRun3",
#             "WtoMuNu2J_24SummerRun3",
#         ],
#         "n_files": [10, 10, 10],
#         "xsec": [55920, 9202, 3490],
#     },
# }

# Process groups

PROCESS_GROUPS = {

    "DY": [
        "DYto2E",
        "DYto2Mu",
        "DYGto2LG"
    ],

    "TOP": [
        "TTto2L2Nu",
        "TTtoLNu2Q",
        "TTG1Jets",
    ],

    "WJetsG": [
        "WtoENu",
        "WtoMuNu",
        "WGtoLNuG"
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
