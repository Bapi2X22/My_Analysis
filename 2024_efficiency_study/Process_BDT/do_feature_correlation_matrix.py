
import os
import glob
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

TREE_NAME = "DiphotonTree"

OUTPUT_DIR = "Feature_Correlation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BDT input features
FEATURES = [
    "leppt", "lepeta",
    "first_jet_eta", "second_jet_eta",
    "pT1_by_mbb", "pT2_by_mbb",
    "Njets",
    "pholead_eta", "phosublead_eta",
    "first_jet_B", "second_jet_B",
    "pholead_mvaID", "phosublead_mvaID",
    "pT1_by_mgg", "pT2_by_mgg",
    "delphi_gg", "delphi_bb", "delphi_bbgg",
    "delphi_ggMET_Puppi",
    "mass_point",
]

# ROOT files
SIGNAL_FILES = glob.glob("Merged_Inputs/merged_signal.root")

BACKGROUND_FILES = glob.glob("Merged_Inputs/merged_background.root")

# Optional event selection
# Set to None if no additional selection is required.
SELECTION = None

# Example:
# SELECTION = "Diphoton_mass > 100"


# ============================================================
# LOAD FEATURES FROM ROOT FILES
# ============================================================

def load_features(files, features):

    all_data = []

    for filename in files:

        print(f"Reading: {filename}")

        with uproot.open(filename) as root_file:

            tree = root_file[TREE_NAME]

            available_branches = tree.keys()

            missing = [
                f for f in features
                if f not in available_branches
            ]

            if missing:
                raise KeyError(
                    f"Missing branches in {filename}: {missing}"
                )

            arrays = tree.arrays(
                features,
                library="ak"
            )

            # Convert event-level features to NumPy
            data = {}

            for feature in features:

                values = ak.to_numpy(arrays[feature])

                if values.ndim != 1:
                    raise ValueError(
                        f"{feature} is not a scalar event-level feature"
                    )

                data[feature] = values

            df = pd.DataFrame(data)

            all_data.append(df)

    if not all_data:
        raise RuntimeError("No ROOT files found!")

    return pd.concat(all_data, ignore_index=True)


# ============================================================
# LOAD SIGNAL AND BACKGROUND
# ============================================================

df_signal = load_features(SIGNAL_FILES, FEATURES)

df_background = load_features(BACKGROUND_FILES, FEATURES)

print("\nSignal events:", len(df_signal))
print("Background events:", len(df_background))


# ============================================================
# OPTIONAL SELECTION
# ============================================================

if SELECTION is not None:

    df_signal = df_signal.query(SELECTION)
    df_background = df_background.query(SELECTION)


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataframe(df):

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    return df


df_signal = clean_dataframe(df_signal)
df_background = clean_dataframe(df_background)


# Combined dataset
df_combined = pd.concat(
    [df_signal, df_background],
    ignore_index=True
)


# ============================================================
# CORRELATION MATRIX
# ============================================================

def plot_correlation(df, sample_name):

    corr = df[FEATURES].corr(method="pearson")

    # Save numerical matrix
    csv_file = os.path.join(
        OUTPUT_DIR,
        f"correlation_{sample_name}.csv"
    )

    corr.to_csv(csv_file)

    # Plot
    plt.figure(figsize=(13, 11))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={
            "label": "Pearson correlation"
        }
    )

    plt.title(
        f"Feature Correlation Matrix - {sample_name}",
        fontsize=16
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save plots
    png_file = os.path.join(
        OUTPUT_DIR,
        f"correlation_{sample_name}.png"
    )

    pdf_file = os.path.join(
        OUTPUT_DIR,
        f"correlation_{sample_name}.pdf"
    )

    plt.savefig(png_file, dpi=300)
    plt.savefig(pdf_file)

    plt.close()

    print(f"Saved: {png_file}")
    print(f"Saved: {csv_file}")

    return corr


# ============================================================
# GENERATE ALL MATRICES
# ============================================================

corr_signal = plot_correlation(
    df_signal,
    "signal"
)

corr_background = plot_correlation(
    df_background,
    "background"
)

corr_combined = plot_correlation(
    df_combined,
    "combined"
)


# ============================================================
# IDENTIFY HIGHLY CORRELATED FEATURES
# ============================================================

def find_high_correlations(corr, threshold=0.90):

    upper_triangle = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    high_corr = (
        upper_triangle
        .stack()
        .reset_index()
    )

    high_corr.columns = [
        "Feature_1",
        "Feature_2",
        "Correlation"
    ]

    high_corr["Abs_Correlation"] = (
        high_corr["Correlation"].abs()
    )

    high_corr = high_corr[
        high_corr["Abs_Correlation"] >= threshold
    ]

    return high_corr.sort_values(
        "Abs_Correlation",
        ascending=False
    )


for name, corr in [
    ("Signal", corr_signal),
    ("Background", corr_background),
    ("Combined", corr_combined)
]:

    print(f"\n{'='*60}")
    print(f"Highly correlated features: {name}")
    print(f"{'='*60}")

    high_corr = find_high_correlations(
        corr,
        threshold=0.90
    )

    print(high_corr.to_string(index=False))

    high_corr.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"high_correlation_{name.lower()}.csv"
        ),
        index=False
    )

print("\nCorrelation analysis completed!")

