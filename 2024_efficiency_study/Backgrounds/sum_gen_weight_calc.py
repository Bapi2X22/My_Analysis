import os
import glob
import ROOT


# ======================================================================
# DIRECTORIES
# ======================================================================

sample_dirs = {

    # --------------------------------------------------------------
    # Normal samples
    # --------------------------------------------------------------

    # "DYGto2LG": [
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/DYGto2LG50_24SummerRun3"
    # ],

    # "DYto2E": [
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/DYto2E50_24SummerRun3"
    # ],

    "DYto2Mu": [
        "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/DYto2Mu50_24SummerRun3"
    ]

    # "TTto2L2Nu": [
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/TTto2L2Nu_24SummerRun3"
    # ],

    # "TTtoLNu2Q": [
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/TTtoLNu2Q_24SummerRun3"
    # ],

    # "TTG1Jets": [
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/TTG1Jets_24SummerRun3"
    # ],

    # "WGtoLNuG": [
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WGtoLNuG_24SummerRun3"
    # ],

    # # --------------------------------------------------------------
    # # W + Jets → electron
    # # Combine 0J + 1J + 2J
    # # --------------------------------------------------------------

    # "WtoE": [
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoENu0J_24SummerRun3",
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoENu1J_24SummerRun3",
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoENu2J_24SummerRun3",
    # ],

    # # --------------------------------------------------------------
    # # W + Jets → muon
    # # Combine 0J + 1J + 2J
    # # --------------------------------------------------------------

    # "WtoMu": [
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoMuNu0J_24SummerRun3",
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoMuNu1J_24SummerRun3",
    #     "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/WtoMuNu2J_24SummerRun3",
    # ],
}


# ======================================================================
# GET ALL ROOT FILES
# ======================================================================

def get_root_files(directory):

    return sorted(
        glob.glob(
            os.path.join(directory, "*.root")
        )
    )


# ======================================================================
# READ METADATA FROM ONE FILE
# ======================================================================

def get_metadata(file):

    root_file = ROOT.TFile.Open(file)

    if not root_file or root_file.IsZombie():
        print(f"ERROR opening: {file}")
        return 0, 0.0

    metadata = root_file.Get("Metadata")

    if not metadata:
        print(f"WARNING: Metadata tree not found in {file}")
        root_file.Close()
        return 0, 0.0

    metadata.GetEntry(0)

    n_events = int(metadata.n_events_presel)
    sum_genw = float(metadata.sum_genw_presel)

    root_file.Close()

    return n_events, sum_genw


# ======================================================================
# CALCULATE ONE DIRECTORY
# ======================================================================

def calculate_directory(directory):

    root_files = get_root_files(directory)

    total_events = 0
    total_genw = 0.0

    print("\n" + "=" * 100)
    print("Directory:")
    print(directory)
    print("Number of ROOT files:", len(root_files))
    print("=" * 100)

    for i, file in enumerate(root_files):

        n_events, sum_genw = get_metadata(file)

        print(
            f"[{i + 1}/{len(root_files)}] "
            f"{os.path.basename(file)}"
        )

        print(
            f"    n_events_presel   = {n_events}"
        )

        print(
            f"    sum_genw_presel   = {sum_genw:.10g}"
        )

        total_events += n_events
        total_genw += sum_genw

    print("-" * 80)

    print(
        f"Directory total events     = "
        f"{total_events}"
    )

    print(
        f"Directory total genWeight  = "
        f"{total_genw:.10g}"
    )

    return total_events, total_genw


# ======================================================================
# PROCESS ALL SAMPLES
# ======================================================================

sample_results = {}


for sample, directories in sample_dirs.items():

    print("\n\n")
    print("#" * 100)
    print(f"SAMPLE: {sample}")
    print("#" * 100)

    sample_events = 0
    sample_genw = 0.0

    # Loop over all directories belonging to this sample
    for directory in directories:

        n_events, genw = calculate_directory(
            directory
        )

        sample_events += n_events
        sample_genw += genw

    sample_results[sample] = {
        "events": sample_events,
        "genWeight": sample_genw,
    }

    print("\n" + "*" * 80)

    print(
        f"FINAL {sample}:"
    )

    print(
        f"Total n_events_presel = "
        f"{sample_events}"
    )

    print(
        f"Total sum_genw_presel = "
        f"{sample_genw:.10g}"
    )

    print("*" * 80)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n\n")
print("#" * 100)
print("FINAL SUMMARY")
print("#" * 100)

print(
    f"{'Sample':<20}"
    f"{'Events':>20}"
    f"{'Sum genWeight':>25}"
)

print("-" * 65)

for sample, result in sample_results.items():

    print(
        f"{sample:<20}"
        f"{result['events']:>20}"
        f"{result['genWeight']:>25.10g}"
    )

print("-" * 65)
