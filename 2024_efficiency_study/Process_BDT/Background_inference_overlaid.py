
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import uproot
import awkward as ak

import os
import matplotlib.pyplot as plt
import ROOT

from sklearn import metrics


# --------------------------------------------------
# Configuration
# --------------------------------------------------

model_file = "Models/BDT_reference_model_061126.pkl"
output_signal = "signal_scored.parquet"
output_background = "background_scored.parquet"
sig_file_name = "Merged_Inputs/merged_signal.root"
bkg_file_name = "Merged_Inputs/merged_background.root"

plot_dir = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check_pBDT/reference_BDT_overlaid/"
os.makedirs(plot_dir, exist_ok=True)

mass_grid = np.array(
    [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
)

lumi = 36.33  # fb^-1

# Exact feature names and order used during training.
# inputVars = [
#     "leppt", "lepeta",
#     "first_jet_eta", "second_jet_eta",
#     "pT1_by_mbb", "pT2_by_mbb",
#     "Njets",
#     "pholead_eta", "phosublead_eta",
#     "first_jet_B", "second_jet_B",
#     "pholead_mvaID", "phosublead_mvaID",
#     "pT1_by_mgg", "pT2_by_mgg",
#     "delphi_gg", "delphi_bb", "delphi_bbgg",
#     "delphi_ggMET_Puppi", "energy_resol_lead", "energy_resol_sublead",
#     "mass_point",
# ]

inputVars = [
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

# -----------------------------------------------
# Input scored Parquet files
# -----------------------------------------------

# Only the resolved category is used.
category = "Resolved"

# BDT threshold: change as needed.
bdt_cut = 0.80
bdt_operator = ">"
setmax_factor = 100

signal_mass_points = [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

xsecs = {
    "WGtoLNuG": 671.5,
    "WtoENu0J": 55850,
    "WtoENu2J": 3474,
    "WtoMuNu2J": 3490,
    "DYto2Mu50": 2124.08,
    "DYto2E50": 2124.08,
    "DYGto2LG4": 88.13,
    "DYGto2LG50": 126.7,
    "TTG1Jets": 4.634,
    "TTto2L2Nu": 98.04,
    "TTtoLNu2Q": 405.87

}

# process_map insertion order determines stack order.
process_map = {
    "WGtoLNuG": 0,
    "WtoENu0J": 1,
    "WtoENu2J": 2,
    "WtoMuNu2J": 3,
    "DYto2Mu50": 4,
    "DYto2E50": 5,
    "DYGto2LG4": 6,
    "DYGto2LG50": 7,
    "TTG1Jets": 8,
    "TTto2L2Nu": 9,
    "TTtoLNu2Q": 10,
}

background_groups = {
    "W#gamma + jets": {
        "processes": [
            "WGtoLNuG",
            "WtoENu0J",
            "WtoENu2J",
            "WtoMuNu2J",
        ],
        "color": ROOT.kGreen + 2,
    },

    "Z#gamma + jets": {
        "processes": [
            "DYto2Mu50",
            "DYto2E50",
            "DYGto2LG4",
            "DYGto2LG50",
        ],
        "color": ROOT.kBlue + 1,
    },

    "t#bar{t}#gamma + jets": {
        "processes": [
            "TTG1Jets",
            "TTto2L2Nu",
            "TTtoLNu2Q",
        ],
        "color": ROOT.kRed + 1,
    },
}

# Distinguishable colors for the individual stack components.

process_colors = {
    "TTG1Jets": ROOT.kRed + 1,
    "TTto2L2Nu": ROOT.kRed + 1,
    "TTtoLNu2Q": ROOT.kRed + 1,
    "DYto2Mu50": ROOT.kBlue + 1,
    "DYto2E50": ROOT.kBlue + 1,
    "DYto2Mu10": ROOT.kBlue + 1,
    "DYto2E10": ROOT.kBlue + 1,
    "DYGto2LG4": ROOT.kBlue + 1,
    "DYGto2LG50": ROOT.kBlue + 1,
    "WGtoLNuG": ROOT.kGreen + 2,
    "WtoENu0J": ROOT.kGreen + 2,
    "WtoENu1J": ROOT.kGreen + 2,
    "WtoENu2J": ROOT.kGreen + 2,
    "WtoMuNu0J": ROOT.kGreen + 2,
    "WtoMuNu1J": ROOT.kGreen + 2,
    "WtoMuNu2J": ROOT.kGreen + 2
}

signal_colors = {
    signal_mass_points[0]: ROOT.kMagenta + 1,
    signal_mass_points[1]: ROOT.kOrange + 7,
    signal_mass_points[2]: ROOT.kCyan + 1,
}

WH_xsec = 1.457       # pb
signal_BR = 0.01      # BR(H -> AA -> bbgg)

signal_xsec = WH_xsec * signal_BR

variables_list = [
    "leppt",
    "lepeta",
    "first_jet_eta",
    "second_jet_eta",
    "pT1_by_mbb",
    "pT2_by_mbb",
    "first_jet_B",
    "second_jet_B",
    "Njets",
    "pholead_eta",
    "phosublead_eta",
    "pT1_by_mgg",
    "pT2_by_mgg",
    "pholead_mvaID",
    "phosublead_mvaID",
    "delphi_gg",
    "delphi_bb",
    "delphi_bbgg",
    "delphi_ggMET_Puppi",
    "pt_ratio",
    # "energy_resol_lead",
    # "energy_resol_sublead",
    "CMS_hgg_mass"
]

plot_config = {

    # Lepton
    "leppt": {
        "range": (0.0, 200.0),
        "nbins": 40,
        "xlabel": "Lepton p_{T} [GeV]",
    },
    "lepeta": {
        "range": (-2.5, 2.5),
        "nbins": 25,
        "xlabel": "Lepton #eta",
    },

    # Jets
    "first_jet_eta": {
        "range": (-4.0, 4.0),
        "nbins": 20,
        "xlabel": "Leading b-Jet #eta",
    },
    "second_jet_eta": {
        "range": (-4.0, 4.0),
        "nbins": 20,
        "xlabel": "Subleading b-Jet #eta",
    },
    "pT1_by_mbb": {
        "range": (0.0, 5.0),
        "nbins": 20,
        "xlabel": "Leading jet p_{T} / m_{bb}",
    },
    "pT2_by_mbb": {
        "range": (0.0, 3.0),
        "nbins": 12,
        "xlabel": "Subleading jet p_{T} / m_{bb}",
    },
    "first_jet_B": {
        "range": (0.0, 1.0),
        "nbins": 40,
        "xlabel": "Leading b-Jet B-tag score",
    },
    "second_jet_B": {
        "range": (0.0, 1.0),
        "nbins": 40,
        "xlabel": "Subleading b-Jet B-tag score",
    },
    "Njets": {
        "range": (0, 10),
        "nbins": 10,
        "xlabel": "Number of Jets per event",
    },

    # Photons
    "pholead_eta": {
        "range": (-2.5, 2.5),
        "nbins": 25,
        "xlabel": "Leading photon #eta",
    },
    "phosublead_eta": {
        "range": (-2.5, 2.5),
        "nbins": 25,
        "xlabel": "Subleading photon #eta",
    },
    "pT1_by_mgg": {
        "range": (0.0, 10.0),
        "nbins": 40,
        "xlabel": "Leading photon p_{T} / m_{#gamma#gamma}",
    },
    "pT2_by_mgg": {
        "range": (0.0, 6.0),
        "nbins": 24,
        "xlabel": "Subleading photon p_{T} / m_{#gamma#gamma}",
    },
    "pholead_mvaID": {
        "range": (-1.0, 1.0),
        "nbins": 40,
        "xlabel": "Leading photon MVA ID",
    },
    "phosublead_mvaID": {
        "range": (-1.0, 1.0),
        "nbins": 40,
        "xlabel": "Subleading photon MVA ID",
    },

    # Angular variables
    "delphi_gg": {
        "range": (0.0, 3.2),
        "nbins": 32,
        "xlabel": "#Delta#phi_{#gamma#gamma}",
    },
    "delphi_bb": {
        "range": (0.0, 3.2),
        "nbins": 32,
        "xlabel": "#Delta#phi_{bb}",
    },
    "delphi_bbgg": {
        "range": (0.0, 3.2),
        "nbins": 32,
        "xlabel": "#Delta#phi_{bb,#gamma#gamma}",
    },
    "delphi_ggMET_Puppi": {
        "range": (0, 3.2),
        "nbins": 16,
        "xlabel": "#Delta#phi between #gamma#gamma and MET",
    },
    "pt_ratio": {
        "range": (0, 1),
        "nbins": 10,
        "xlabel": "Sublead photon p_{T} over lead photon p_{T}",
    },
    "energy_resol_lead": {
        "range": (0, 0.2),
        "nbins": 40,
        "xlabel": "Leading photon relative energy resolution",
    },
    "energy_resol_sublead": {
        "range": (0, 0.2),
        "nbins": 40,
        "xlabel": "Subleading photon relative energy resolution",
    },
    "CMS_hgg_mass":{
        "range": (10, 70),
        "nbins": 60,
        "xlabel": "Diphoton invariant mass (m_{#gamma#gamma}) [GeV]",     
    }
}

def CMS_label(pad,
            #   lumi="39.05 fb^{-1}",
              lumi=109.0,
              year="2024",
              energy="13.6 TeV",
              status="Simulation Preliminary",
              x=0.12,
              y=0.96):

    pad.cd()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(ROOT.kBlack)

    # ---- CMS (bold) ----
    latex.SetTextFont(61)
    latex.SetTextSize(0.06)
    latex.DrawLatex(x, y, "CMS")

    # ---- Status (italic) ----
    if status != "":
        latex.SetTextFont(52)
        latex.SetTextSize(0.045)
        latex.DrawLatex(x + 0.11, y, status)

    # ---- Lumi text (right aligned) ----
    latex.SetTextFont(42)
    latex.SetTextSize(0.045)
    latex.SetTextAlign(31)
    lumi_text = f"{year} ({lumi} fb^{{-1}})"
    latex.DrawLatex(0.88, y, lumi_text)


def draw_statbox_manual(hist, x1, y1, x2, y2, color, label=None):

    stats = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    stats.SetFillColor(0)
    stats.SetBorderSize(1)
    stats.SetTextColor(color)
    stats.SetTextFont(42)
    stats.SetTextSize(0.04)
    stats.SetFillStyle(0)

    nbins = hist.GetNbinsX()

    # underflow = hist.GetBinContent(0)
    # overflow  = hist.GetBinContent(nbins + 1)

    integral = hist.Integral(1, nbins)

    if label is not None:
        stats.AddText(label)

    # stats.AddText(f"Entries = {int(hist.GetEntries())}")
    # stats.AddText(f"Underflow = {underflow:.2f}")
    # stats.AddText(f"Overflow = {overflow:.2f}")
    stats.AddText(f"Integral = {integral:.2f}")

    stats.Draw()

    return stats

def draw_statbox_manual_BA(
    hist_before,
    hist_after,
    x1, y1, x2, y2,
    color,
    label=None,
):

    stats = ROOT.TPaveText(x1, y1, x2, y2, "NDC")

    stats.SetFillColor(0)
    stats.SetFillStyle(0)
    stats.SetBorderSize(1)

    stats.SetLineColor(color)
    stats.SetTextColor(color)

    stats.SetTextFont(42)
    stats.SetTextSize(0.035)
    stats.SetTextAlign(22)

    before_yield = hist_before.Integral(
        1, hist_before.GetNbinsX()
    )

    after_yield = hist_after.Integral(
        1, hist_after.GetNbinsX()
    )

    if label is not None:
        stats.AddText(label)

    stats.AddText(
        f"B: {before_yield:.2f}  A: {after_yield:.2f}"
    )

    stats.Draw()

    return stats

def draw_statbox_manual_small(hist, x1, y1, x2, y2, color, label=None, is_signal = False):

    stats = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    stats.SetFillColor(0)
    stats.SetBorderSize(1)
    stats.SetLineColor(color)
    stats.SetTextColor(color)
    stats.SetTextFont(42)
    stats.SetTextSize(0.04)

    nbins = hist.GetNbinsX()

    # underflow = hist.GetBinContent(0)
    # overflow  = hist.GetBinContent(nbins + 1)

    integral = hist.Integral(1, nbins)

    if label is not None:
        stats.AddText(label)

    # stats.AddText(f"Entries = {int(hist.GetEntries())}")
    if is_signal:
        stats.AddText(f"{integral:.2f}")
    else:
        stats.AddText(f"Integral = {integral:.2f}")

    stats.Draw()

    return stats

def draw_statbox_combined(hist_list, x1, y1, x2, y2, color, label=None):

    stats = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    stats.SetFillColor(0)
    stats.SetBorderSize(1)
    stats.SetLineColor(color)
    stats.SetTextColor(color)
    stats.SetTextFont(42)
    stats.SetTextSize(0.04)
    stats.SetFillStyle(0)

    nbins = hist_list[0].GetNbinsX()

    Integral = 0
    # Entries = 0

    for h in hist_list:
        # underflow += h.GetBinContent(0)
        # overflow += h.GetBinContent(nbins + 1)
        Integral += h.Integral(1, nbins)
        # Entries += h.GetEntries()

    if label is not None:
        stats.AddText(label)

    # stats.AddText(f"Entries = {int(Entries)}")
    stats.AddText(f"Integral = {Integral:.2f}")

    stats.Draw()

    return stats


def draw_all_statboxes(
    h_sig,
    h_components,
    mass,
    signal_color,
    stat_boxes,
):
    # Signal
    stat_boxes.append(
        draw_statbox_manual_small(
            h_sig,
            0.34, 0.78, 0.69, 0.88,
            signal_color,
            label=f"Signal (M_{{A}} = {mass} GeV)",
            is_signal=True,
        )
    )

    # Wgamma + jets
    stat_boxes.append(
        draw_statbox_combined(
            [h_components["W#gamma + jets"]],
            0.18, 0.66, 0.40, 0.76,
            ROOT.kGreen + 2,
            label="W#gamma + jets",
        )
    )

    # Zgamma + jets
    stat_boxes.append(
        draw_statbox_combined(
            [h_components["Z#gamma + jets"]],
            0.40, 0.66, 0.62, 0.76,
            ROOT.kBlue + 1,
            label="Z#gamma + jets",
        )
    )

    # ttbar gamma + jets
    stat_boxes.append(
        draw_statbox_combined(
            [h_components["t#bar{t}#gamma + jets"]],
            0.62, 0.66, 0.84, 0.76,
            ROOT.kRed + 1,
            label="t#bar{t}#gamma + jets",
        )
    )

    return stat_boxes


def combine_hists(hist_dict, process_names, name, color):

    h_combined = None

    for process_name in process_names:

        h = hist_dict[process_name]

        if h_combined is None:
            h_combined = h.Clone(name)
            h_combined.SetDirectory(0)
        else:
            h_combined.Add(h)

    h_combined.SetFillStyle(0)
    h_combined.SetLineColor(color)
    h_combined.SetLineWidth(3)

    return h_combined

# Load the complete scored datasets.
df_sig = pd.read_parquet(output_signal)
df_bkg = pd.read_parquet(output_background)

print("Signal events:", len(df_sig))
print("Background events:", len(df_bkg))
print("Background process IDs:")
print(df_bkg["process_id"].value_counts().sort_index())



def exportTree(final_dataset,treeNameToExport,export_filename): 
    export_file = uproot.recreate(export_filename)
    print("\n\t  file exported  : ",export_filename)
    allBranchesToWrite={ ky : final_dataset[ky].dtype for ky in final_dataset}
    export_file.mktree( treeNameToExport ,allBranchesToWrite)
    export_file[treeNameToExport].extend(final_dataset)
    export_file.close()

#############################################################

#########################################################
def getTH1FromNumpHist(nph):
    hist = ROOT.TH1D("","",len(nph[1])-1,np.array(nph[1],dtype=float))
    for i,freq in enumerate(nph[0]):
        hist.SetBinContent(i+1,freq)
    hist.ResetStats() 	
    return hist


def make_weights_nonnegative(df, weight_col="evt_wgt", label="sample"):
    """
    Apply the same negative-weight treatment as the training script:
    take absolute weights and rescale to preserve the original sum.
    """

    df = df.copy()

    original_sum = df[weight_col].sum()
    absolute_sum = df[weight_col].abs().sum()

    print(f"{label}: original sum = {original_sum:.8g}")
    print(f"{label}: absolute sum = {absolute_sum:.8g}")

    if not np.isfinite(original_sum) or not np.isfinite(absolute_sum):
        raise ValueError(f"{label}: non-finite event-weight sum.")

    if original_sum <= 0 or absolute_sum == 0:
        raise ValueError(
            f"{label}: cannot preserve a non-positive original sum "
            "using non-negative weights. Check the normalization."
        )

    df[weight_col] = df[weight_col].abs()
    df[weight_col] *= original_sum / absolute_sum

    print(f"{label}: rescaled sum = {df[weight_col].sum():.8g}")
    print(f"{label}: remaining negative weights = "
          f"{(df[weight_col] < 0).sum()}")

    return df


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def drawROCOverlay(
    df_sig,
    df_bkg,
    mass_grid,
    name="BDT_plots/ROC_all_mass_points",
    weight_col="evt_wgt",
):
    """
    Overlay ROC curves for all mass hypotheses.

    Signal: matching mass_point, score from BDT_score.
    Background: all events, score from BDT_<mass>.

    X-axis: Signal efficiency.
    Y-axis: Background rejection (1 - FPR).
    """

    fig, ax = plt.subplots(figsize=(9, 8))
    auc_results = {}

    for mass in mass_grid:

        # Signal events for this mass hypothesis
        sig = df_sig.loc[
            np.isclose(df_sig["mass_point"], mass)
        ]

        sig_scores = sig["bdt_score"].to_numpy(dtype=float)
        sig_weights = sig[weight_col].to_numpy(dtype=float)

        # Background evaluated at this mass hypothesis
        bkg_scores = df_bkg[f"bdt_{mass}"].to_numpy(dtype=float)
        bkg_weights = df_bkg[weight_col].to_numpy(dtype=float)

        # Remove invalid scores and weights
        sig_mask = (
            np.isfinite(sig_scores)
            & np.isfinite(sig_weights)
        )
        bkg_mask = (
            np.isfinite(bkg_scores)
            & np.isfinite(bkg_weights)
        )

        sig_scores = sig_scores[sig_mask]
        sig_weights = sig_weights[sig_mask]

        bkg_scores = bkg_scores[bkg_mask]
        bkg_weights = bkg_weights[bkg_mask]

        if len(sig_scores) == 0 or len(bkg_scores) == 0:
            print(f"Skipping mass {mass}: no valid events.")
            continue

        if np.any(sig_weights < 0) or np.any(bkg_weights < 0):
            raise ValueError(
                f"Negative weights found for mass {mass}. "
                "Choose a consistent signed-weight treatment."
            )

        # Combine signal and background
        scores = np.concatenate([sig_scores, bkg_scores])

        labels = np.concatenate([
            np.ones(len(sig_scores), dtype=int),
            np.zeros(len(bkg_scores), dtype=int),
        ])

        weights = np.concatenate([sig_weights, bkg_weights])

        # Weighted ROC
        fpr, tpr, thresholds = metrics.roc_curve(
            labels,
            scores,
            sample_weight=weights,
        )

        auc = metrics.auc(fpr, tpr)
        auc_results[mass] = auc

        # Plot signal efficiency vs background rejection
        ax.plot(
            tpr,
            1.0 - fpr,
            linewidth=1.8,
            label=f"mA = {mass} GeV (AUC = {auc:.3f})",
        )

        print(
            f"mA = {mass:2d} GeV | "
            f"Nsig = {len(sig_scores):7d} | "
            f"Nbkg = {len(bkg_scores):7d} | "
            f"AUC = {auc:.5f}"
        )

    # Reference for a random classifier
    ax.plot(
        [0, 1], [1, 0],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Random classifier",
    )

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection (1 - background efficiency)")
    ax.set_title("BDT ROC curves for different mass hypotheses")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

    ax.legend(
        loc="lower left",
        fontsize=9,
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(plot_dir +  name + ".png", dpi=300)
    fig.savefig(plot_dir + name + ".pdf", dpi=300)
    plt.close(fig)

    print("\nAUC summary:")
    for mass, auc in auc_results.items():
        print(f"mA = {mass:2d} GeV: AUC = {auc:.5f}")

    return auc_results



def loadSignal(filename, inputVars):
    # tree_name = "DiphotonTree/Resolved"
    tree_name = "DiphotonTree"

    # Read all branches from the signal tree.
    with uproot.open(filename) as f:
        tree = f[tree_name]
        dataset_Sig = tree.arrays(library="ak")

    df_sig = ak.to_dataframe(dataset_Sig)

    print("Loaded signal branches:", len(df_sig.columns))
    print("Signal events:", len(df_sig))

    # Check that all required BDT features are available.
    missing = [v for v in inputVars if v not in df_sig.columns]
    if missing:
        raise KeyError(f"Missing signal branches: {missing}")

    return df_sig


def loadMCBackground(filename, inputVars, inputVarsMC):
    # tree_name = "DiphotonTree/Resolved"
    tree_name = "DiphotonTree"

    # Read all branches from the background tree.
    with uproot.open(filename) as f:
        tree = f[tree_name]
        dataset_Bkg = tree.arrays(library="ak")

    df_bkg = ak.to_dataframe(dataset_Bkg)

    print("Loaded background branches:", len(df_bkg.columns))
    print("Background events:", len(df_bkg))

    # Preserve your existing mapping from MC branch names
    # to the common names expected by the analysis.
    renames = dict(zip(inputVarsMC, inputVars))

    # Rename only branches that actually exist in the DataFrame.
    renames = {
        old: new for old, new in renames.items()
        if old in df_bkg.columns
    }

    print("Renames:", renames)
    df_bkg.rename(columns=renames, inplace=True)

    # Check that all required BDT features are available.
    missing = [v for v in inputVars if v not in df_bkg.columns]
    if missing:
        raise KeyError(f"Missing background branches after renaming: {missing}")

    return df_bkg


def make_stack_after_bdt_cut_with_overlay(
    obs_name,
    mass,
    df_sig,
    df_bkg,
    bdt_cut,
    bdt_operator=">",
):
    """
    Overlay before-BDT and after-BDT distributions.

    Before BDT:
        All signal events for the requested mass.
        All background events, drawn as transparent outlines.

    After BDT:
        Signal and background events passing the BDT cut.
        Background drawn as a filled stack.

    Ratio:
        Total background after BDT / total background before BDT.
    """

    score_col = f"bdt_{mass}"

    required_sig = [
        obs_name, "mass_point", "bdt_score", "weight"
    ]
    required_bkg = [
        obs_name, "process_id", score_col, "weight"
    ]

    missing_sig = [
        c for c in required_sig if c not in df_sig.columns
    ]
    missing_bkg = [
        c for c in required_bkg if c not in df_bkg.columns
    ]

    if missing_sig or missing_bkg:
        raise KeyError(
            f"Missing signal columns: {missing_sig}; "
            f"missing background columns: {missing_bkg}"
        )

    # ------------------------------------------------
    # BDT comparison
    # ------------------------------------------------

    operators = {
        ">":  lambda x, cut: x > cut,
        "<":  lambda x, cut: x < cut,
        ">=": lambda x, cut: x >= cut,
        "<=": lambda x, cut: x <= cut,
    }

    if bdt_operator not in operators:
        raise ValueError(
            f"Invalid bdt_operator: {bdt_operator!r}. "
            f"Choose one of {list(operators)}."
        )

    compare = operators[bdt_operator]

    # ------------------------------------------------
    # Event selection
    # ------------------------------------------------

    sig_mass_mask = np.isclose(
        pd.to_numeric(df_sig["mass_point"], errors="coerce"),
        mass,
    )

    # Signal before and after BDT
    sig_before = df_sig.loc[sig_mass_mask].copy()

    sig_after = sig_before.loc[
        compare(sig_before["bdt_score"], bdt_cut)
    ].copy()

    # Background before and after BDT
    bkg_before = df_bkg.copy()

    bkg_after = bkg_before.loc[
        compare(bkg_before[score_col], bdt_cut)
    ].copy()

    print(
        f"\nObservable: {obs_name} | "
        f"mA = {mass} GeV | BDT {bdt_operator} {bdt_cut}"
    )

    print("Signal before BDT:", len(sig_before))
    print("Signal after BDT :", len(sig_after))
    print("Background before BDT:", len(bkg_before))
    print("Background after BDT :", len(bkg_after))

    # ------------------------------------------------
    # Histogram configuration
    # ------------------------------------------------

    config = plot_config[obs_name]

    nbins = config["nbins"]
    xmin, xmax = config["range"]
    xlabel = config.get("xlabel", obs_name)

    # ------------------------------------------------
    # Histogram filling helper
    # ------------------------------------------------

    def make_hist(name, events, process_name=None, is_signal=False):

        hist = ROOT.TH1D(
            name,
            f";{xlabel};Events",
            nbins,
            xmin,
            xmax,
        )

        hist.Sumw2()

        values = pd.to_numeric(
            events[obs_name], errors="coerce"
        ).to_numpy(dtype=float)

        weights = pd.to_numeric(
            events["weight"], errors="coerce"
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(values)
            & np.isfinite(weights)
            & (values != -999)
        )

        values = values[valid]
        weights = weights[valid]

        if is_signal:
            scale = signal_xsec * lumi * 1000.0
        else:
            scale = xsecs[process_name] * lumi * 1000.0

        for value, weight in zip(values, weights):
            hist.Fill(float(value), float(weight * scale))

        return hist

    # ------------------------------------------------
    # Create the two stacks
    # ------------------------------------------------

    stack_before = ROOT.THStack(
        f"stack_before_{obs_name}_m{mass}",
        f";{xlabel};Events",
    )

    stack_after = ROOT.THStack(
        f"stack_after_{obs_name}_m{mass}",
        f";{xlabel};Events",
    )

    background_before = {}
    background_after = {}

    output_hists = []

    # ------------------------------------------------
    # Create individual process histograms
    # ------------------------------------------------

    for process_name, process_id in process_map.items():

        events_before = bkg_before.loc[
            bkg_before["process_id"] == process_id
        ]

        events_after = bkg_after.loc[
            bkg_after["process_id"] == process_id
        ]

        h_before = make_hist(
            f"h_before_{process_name}_{obs_name}_m{mass}",
            events_before,
            process_name=process_name,
        )

        h_after = make_hist(
            f"h_after_{process_name}_{obs_name}_m{mass}",
            events_after,
            process_name=process_name,
        )

        background_before[process_name] = h_before
        background_after[process_name] = h_after

        output_hists.extend([h_before, h_after])

    # ------------------------------------------------
    # Combine individual processes into 3 groups
    # ------------------------------------------------

    group_before = {}
    group_after = {}

    for group_name, group_info in background_groups.items():

        processes = group_info["processes"]
        color = group_info["color"]

        h_before_group = None
        h_after_group = None

        for process_name in processes:

            # Before BDT
            if h_before_group is None:

                h_before_group = background_before[
                    process_name
                ].Clone(
                    f"h_before_{group_name}_{obs_name}_m{mass}"
                )

                h_before_group.SetDirectory(0)

            else:
                h_before_group.Add(
                    background_before[process_name]
                )

            # After BDT
            if h_after_group is None:

                h_after_group = background_after[
                    process_name
                ].Clone(
                    f"h_after_{group_name}_{obs_name}_m{mass}"
                )

                h_after_group.SetDirectory(0)

            else:
                h_after_group.Add(
                    background_after[process_name]
                )

        # Before BDT: transparent group boundaries
        h_before_group.SetFillStyle(0)
        h_before_group.SetLineColor(color)
        h_before_group.SetLineWidth(3)

        # After BDT: filled group stack
        h_after_group.SetFillStyle(1001)
        h_after_group.SetFillColor(color)
        h_after_group.SetLineColor(color)
        h_after_group.SetLineWidth(1)

        group_before[group_name] = h_before_group
        group_after[group_name] = h_after_group

        # Add exactly 3 histograms to each stack
        stack_before.Add(h_before_group)
        stack_after.Add(h_after_group)

        output_hists.extend([
            h_before_group,
            h_after_group,
        ])

        print(
            f"{group_name:20s}: "
            f"before = {h_before_group.Integral():.6g}, "
            f"after = {h_after_group.Integral():.6g}"
        )

    # ------------------------------------------------
    # Calculate total background explicitly
    # ------------------------------------------------

    h_bkg_before = None
    h_bkg_after = None

    for group_name in background_groups:

        if h_bkg_before is None:

            h_bkg_before = group_before[group_name].Clone(
                f"total_background_before_{obs_name}_m{mass}"
            )
            h_bkg_before.SetDirectory(0)

            h_bkg_after = group_after[group_name].Clone(
                f"total_background_after_{obs_name}_m{mass}"
            )
            h_bkg_after.SetDirectory(0)

        else:

            h_bkg_before.Add(group_before[group_name])
            h_bkg_after.Add(group_after[group_name])
    # ------------------------------------------------
    # Signal histograms: before and after BDT
    # ------------------------------------------------

    h_sig_before = make_hist(
        f"h_signal_before_{obs_name}_m{mass}",
        sig_before,
        is_signal=True,
    )

    h_sig_after = make_hist(
        f"h_signal_after_{obs_name}_m{mass}",
        sig_after,
        is_signal=True,
    )

    signal_color = ROOT.kOrange + 7

    # Before BDT: dashed
    h_sig_before.SetLineColor(signal_color)
    h_sig_before.SetLineWidth(3)
    h_sig_before.SetLineStyle(2)
    h_sig_before.SetFillStyle(0)

    # After BDT: solid
    h_sig_after.SetLineColor(signal_color)
    h_sig_after.SetLineWidth(3)
    h_sig_after.SetLineStyle(1)
    h_sig_after.SetFillStyle(0)

    h_before_legend = ROOT.TH1D("h_before_legend", "", 1, 0, 1)
    h_after_legend = ROOT.TH1D("h_after_legend", "", 1, 0, 1)

    # Before BDT: filled box
    h_before_legend.SetFillColor(ROOT.kGray + 1)
    h_before_legend.SetFillStyle(1001)
    h_before_legend.SetLineColor(ROOT.kGray + 1)

    # After BDT: line
    h_after_legend.SetFillStyle(0)
    h_after_legend.SetLineColor(ROOT.kBlack)
    h_after_legend.SetLineWidth(2)

    output_hists.extend([h_sig_before, h_sig_after, h_before_legend, h_after_legend])

    # ------------------------------------------------
    # Total background histograms
    # ------------------------------------------------
    h_bkg_before.SetDirectory(0)
    h_bkg_after.SetDirectory(0)

    # ------------------------------------------------
    # Canvas and pads
    # ------------------------------------------------

    canvas = ROOT.TCanvas(
        f"c_{obs_name}_m{mass}",
        "",
        1000,
        850,
    )

    # Upper pad
    pad_top = ROOT.TPad(
        f"pad_top_{obs_name}_m{mass}",
        "",
        0.0, 0.30,
        1.0, 1.0,
    )

    # Lower ratio pad
    pad_ratio = ROOT.TPad(
        f"pad_ratio_{obs_name}_m{mass}",
        "",
        0.0, 0.0,
        1.0, 0.30,
    )

    pad_top.SetLeftMargin(0.12)
    pad_top.SetRightMargin(0.05)
    pad_top.SetTopMargin(0.08)
    pad_top.SetBottomMargin(0.02)
    pad_top.SetLogy()

    pad_ratio.SetLeftMargin(0.12)
    pad_ratio.SetRightMargin(0.05)
    pad_ratio.SetTopMargin(0.03)
    pad_ratio.SetBottomMargin(0.35)

    pad_top.Draw()
    pad_ratio.Draw()

    # ------------------------------------------------
    # Draw upper pad
    # ------------------------------------------------

    # ------------------------------------------------
    # Draw upper pad
    # ------------------------------------------------

    # ------------------------------------------------
    # Draw upper pad
    # ------------------------------------------------

    pad_top.cd()

    ymax = max(
        h_bkg_before.GetMaximum(),
        h_bkg_after.GetMaximum(),
        h_sig_before.GetMaximum(),
        h_sig_after.GetMaximum(),
    )

    ymin = 0.03
    ymax *= setmax_factor

    stack_after.SetMinimum(ymin)
    stack_after.SetMaximum(ymax)

    # 1. Draw filled after-BDT background stack FIRST
    stack_after.Draw("HIST")

    # 2. Draw cumulative before-BDT outlines
    cumulative_before = {}

    previous = None

    for i, group_name in enumerate(background_groups):

        h = group_before[group_name].Clone(
            f"cumulative_before_{i}_{obs_name}_m{mass}"
        )
        h.SetDirectory(0)

        if previous is not None:
            h.Add(previous)

        h.SetFillStyle(0)
        h.SetLineColor(background_groups[group_name]["color"])
        h.SetLineWidth(2)

        h.Draw("HIST SAME")

        cumulative_before[group_name] = h
        previous = h

    # 3. Signal after BDT
    h_sig_after.Draw("HIST SAME")

    # 4. Signal before BDT
    h_sig_before.Draw("HIST SAME")

    # ------------------------------------------------
    # Combined background stat boxes (upper pad)
    # ------------------------------------------------

    stat_boxes = []

    bottom_y2 = 0.76
    bottom_y1 = 0.66

    stat_boxes.append(
        draw_statbox_manual_BA(
            group_before["W#gamma + jets"],
            group_after["W#gamma + jets"],
            0.24, 0.66, 0.46, 0.76,
            ROOT.kGreen + 2,
            label="W#gamma + jets",
        )
    )

    stat_boxes.append(
        draw_statbox_manual_BA(
            group_before["Z#gamma + jets"],
            group_after["Z#gamma + jets"],
            0.46, 0.66, 0.68, 0.76,
            ROOT.kBlue + 1,
            label="Z#gamma + jets",
        )
    )

    stat_boxes.append(
        draw_statbox_manual_BA(
            group_before["t#bar{t}#gamma + jets"],
            group_after["t#bar{t}#gamma + jets"],
            0.68, 0.66, 0.90, 0.76,
            ROOT.kRed + 1,
            label="t#bar{t}#gamma + jets",
        )
    )


    top_y2 = 0.88
    top_y1 = 0.76

    stat_boxes.append(
        draw_statbox_manual_BA(
            h_bkg_before,
            h_bkg_after,
            0.39, 0.76, 0.64, 0.88,
            ROOT.kBlack,
            label="Total Background",
        )
    )

    # ------------------------------------------------
    # Signal stat box (upper pad)
    # ------------------------------------------------

    signal_stat_boxes = []


    signal_stat_boxes.append(
        draw_statbox_manual_BA(
            h_sig_before,
            h_sig_after,
            0.15, 0.76, 0.38, 0.88,
            signal_color,
            label=f"Signal (M_{{A}} = {mass} GeV)",
        )
    )

    # Axis formatting
    stack_after.GetYaxis().SetTitle("Events")
    stack_after.GetXaxis().SetLabelSize(0)

    # Explicitly construct cumulative histograms
    cumulative_before = {}

    previous = None

    for i, group_name in enumerate(background_groups):

        h = group_before[group_name].Clone(
            f"cumulative_before_{i}_{obs_name}_m{mass}"
        )
        h.SetDirectory(0)

        if previous is not None:
            h.Add(previous)

        h.SetFillStyle(0)
        h.SetLineColor(background_groups[group_name]["color"])
        h.SetLineWidth(2)
        h.SetLineStyle(1)

        cumulative_before[group_name] = h

        h.Draw("HIST SAME")

        previous = h

    stack_after.GetYaxis().SetTitle("Events")
    stack_after.GetXaxis().SetLabelSize(0)

    # ------------------------------------------------
    # Legend
    # ------------------------------------------------

    legend = ROOT.TLegend(0.58, 0.55, 0.93, 0.89)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.025)

    legend.AddEntry(
        h_sig_after,
        f"Signal (after BDT), m_{{A}} = {mass} GeV",
        "l",
    )

    legend.AddEntry(
        h_sig_before,
        "Signal (before BDT)",
        "l",
    )

    # ------------------------------------------------
    # Before / After BDT style legend
    # -----------------------------------------------

    bdt_legend = ROOT.TLegend(0.74, 0.77, 0.94, 0.84)
    bdt_legend.SetBorderSize(0)
    bdt_legend.SetFillStyle(0)
    bdt_legend.SetTextFont(42)
    bdt_legend.SetTextSize(0.025)

    bdt_legend.AddEntry(
        h_before_legend,
        "B: Before BDT",
        "f",
    )

    bdt_legend.AddEntry(
        h_after_legend,
        "A: After BDT",
        "l",
    )

    bdt_legend.Draw()

    # ------------------------------------------------
    # Annotation
    # ------------------------------------------------

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.030)

    latex.DrawLatex(
        0.80, 0.89,
        f"Category {category}",
    )

    latex.DrawLatex(
        0.80, 0.85,
        f"BDT {bdt_operator} {bdt_cut:.2f}",
    )

    bin_width = (xmax - xmin) / nbins
    latex.DrawLatex(
        0.66, 0.89,
        f"Bin width = {bin_width:.2f}",
    )


    CMS_label(canvas, lumi=lumi)

    # ------------------------------------------------
    # Ratio: after BDT / before BDT
    # ------------------------------------------------

    pad_ratio.cd()

    h_ratio = h_bkg_after.Clone(
        f"ratio_after_over_before_{obs_name}_m{mass}"
    )
    h_ratio.SetDirectory(0)
    h_ratio.Reset("ICES")

    # Since after-BDT events are a subset of before-BDT events,
    # Cov(after, before) = SumW2(after).
    #
    # Include this covariance in the ratio uncertainty.
    for ibin in range(1, nbins + 1):

        after = h_bkg_after.GetBinContent(ibin)
        before = h_bkg_before.GetBinContent(ibin)

        var_after = h_bkg_after.GetBinError(ibin) ** 2
        var_before = h_bkg_before.GetBinError(ibin) ** 2

        if before == 0:
            continue

        ratio = after / before

        covariance = var_after

        variance = (
            var_after / (before ** 2)
            + (after ** 2) * var_before / (before ** 4)
            - 2.0 * after * covariance / (before ** 3)
        )

        error = np.sqrt(max(variance, 0.0))

        h_ratio.SetBinContent(ibin, ratio)
        h_ratio.SetBinError(ibin, error)

    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.8)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetLineColor(ROOT.kBlack)

    h_ratio.SetTitle("")
    h_ratio.SetStats(0)

    h_ratio.GetYaxis().SetTitle("(After/Before) BDT")
    h_ratio.GetXaxis().SetTitle(xlabel)

    h_ratio.GetYaxis().SetNdivisions(505)
    h_ratio.GetYaxis().SetTitleSize(0.08)
    h_ratio.GetYaxis().SetTitleOffset(0.48)
    h_ratio.GetYaxis().SetLabelSize(0.09)

    h_ratio.GetXaxis().SetTitleSize(0.13)
    h_ratio.GetXaxis().SetTitleOffset(1.05)
    h_ratio.GetXaxis().SetLabelSize(0.10)

    # --------------------------------------------------------
    # Dynamic Y-axis range for ratio pad
    # --------------------------------------------------------
    ratio_min = 0.0
    ratio_max = 0.0

    for ibin in range(1, nbins + 1):
        value = h_ratio.GetBinContent(ibin)
        error = h_ratio.GetBinError(ibin)

        if value == 0:
            continue

        ratio_min = min(ratio_min, value - error)
        ratio_max = max(ratio_max, value + error)

    # Add some padding
    if ratio_max > ratio_min:
        padding = 0.15 * (ratio_max - ratio_min)
    else:
        padding = 0.05

    h_ratio.SetMinimum(max(0.0, ratio_min - padding))
    h_ratio.SetMaximum(ratio_max + padding)

    h_ratio.Draw("E1")

    # Reference line at ratio = 1
    unity = ROOT.TLine(
        xmin, 1.0,
        xmax, 1.0,
    )

    unity.SetLineColor(ROOT.kRed + 1)
    unity.SetLineStyle(2)
    unity.SetLineWidth(2)
    unity.Draw("SAME")

    # ------------------------------------------------
    # Save outputs
    # ------------------------------------------------

    stem = (
        f"{plot_dir}/overlay_{obs_name}"
        f"_m{mass}_BDT{bdt_cut:.2f}"
    )

    canvas.cd()
    canvas.Modified()
    canvas.Update()

    canvas.SaveAs(stem + ".png")
    canvas.SaveAs(stem + ".pdf")

    fout = ROOT.TFile(stem + ".root", "RECREATE")

    for hist in output_hists:
        hist.Write()

    h_bkg_before.Write("total_background_before")
    h_bkg_after.Write("total_background_after")
    h_sig_before.Write("signal_before")
    h_sig_after.Write("signal_after")
    h_ratio.Write("background_ratio_after_over_before")

    canvas.Write("canvas")

    fout.Close()

    print("Total background before BDT:", h_bkg_before.Integral())
    print("Total background after BDT :", h_bkg_after.Integral())
    print("Signal before BDT:", h_sig_before.Integral())
    print("Signal after BDT :", h_sig_after.Integral())
    print("Saved:", stem)

    # Keep ROOT objects alive until the canvas is saved.
    canvas._keepalive = [
        pad_top,
        pad_ratio,
        stack_before,
        stack_after,
        h_sig_before,
        h_sig_after,
        h_bkg_before,
        h_bkg_after,
        h_ratio,
        unity,
        legend,
        latex,
        *cumulative_before.values(),
        *stat_boxes,
        *signal_stat_boxes,
    ]

    canvas.Close()

def drawScoresInference(
    df_sig,
    df_bkg,
    mass,
    name,
    nBins=25,
    c_min=0.0,
    c_max=1.0,
    weight_col="evt_wgt",
):
    """
    Plot BDT score distributions for one mass hypothesis.

    Signal:
        Use events whose existing mass_point equals mass.

    Background:
        Use BDT_<mass>, the background score evaluated at mass.

    Outputs:
        name.png
        name.pdf
        name_hists.root
        name_trees_sig.root
        name_trees_bkg.root
    """

    sig = df_sig.loc[
        np.isclose(df_sig["mass_point"], mass)
    ].copy()

    bkg = df_bkg.copy()

    sig_scores = sig["bdt_score"].to_numpy(dtype=float)
    bkg_scores = bkg[f"bdt_{mass}"].to_numpy(dtype=float)

    sig_weights = sig[weight_col].to_numpy(dtype=float)
    bkg_weights = bkg[weight_col].to_numpy(dtype=float)

    # Remove non-finite values and invalid weights.
    sig_mask = (
        np.isfinite(sig_scores)
        & np.isfinite(sig_weights)
    )
    bkg_mask = (
        np.isfinite(bkg_scores)
        & np.isfinite(bkg_weights)
    )

    sig_scores, sig_weights = (
        sig_scores[sig_mask], sig_weights[sig_mask]
    )
    bkg_scores, bkg_weights = (
        bkg_scores[bkg_mask], bkg_weights[bkg_mask]
    )

    if np.any(sig_weights < 0) or np.any(bkg_weights < 0):
        raise ValueError(
            "Negative event weights found. Define a consistent "
            "negative-weight treatment before plotting."
        )

    bins = np.linspace(c_min, c_max, nBins)

    # Weighted distributions.
    H_sig = np.histogram(
        sig_scores, bins=bins, weights=sig_weights
    )
    H_bkg = np.histogram(
        bkg_scores, bins=bins, weights=bkg_weights
    )

    # Sum of squared weights gives the correct MC statistical
    # variance for a weighted histogram.
    err_sig = np.sqrt(np.histogram(
        sig_scores, bins=bins, weights=sig_weights**2
    )[0])

    err_bkg = np.sqrt(np.histogram(
        bkg_scores, bins=bins, weights=bkg_weights**2
    )[0])

    edges = H_sig[1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    h_sig = H_sig[0]
    h_bkg = H_bkg[0]

    ymax = max(
        np.max(h_sig + err_sig) if len(h_sig) else 0,
        np.max(h_bkg + err_bkg) if len(h_bkg) else 0,
    )
    ymax = ymax * 1.25 if ymax > 0 else 1.0

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        centers, h_sig, width=widths,
        color="blue", alpha=0.45,
        label=f"Signal, mA={mass} GeV",
    )

    ax.bar(
        centers, h_bkg, width=widths,
        color="red", alpha=0.40,
        label="Background",
    )

    ax.errorbar(
        centers, h_sig, yerr=err_sig,
        fmt="o", color="blue", markersize=3,
    )

    ax.errorbar(
        centers, h_bkg, yerr=err_bkg,
        fmt="o", color="red", markersize=3,
    )

    ax.set_xlim(c_min, c_max)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("BDT output")
    ax.set_ylabel("Weighted events")
    ax.set_title(f"BDT signal-background separation ({mass} GeV)")
    ax.legend()
    fig.tight_layout()

    fig.savefig(plot_dir + name + ".png", dpi=300)
    fig.savefig(plot_dir + name + ".pdf", dpi=300)
    plt.close(fig)

    # --------------------------------------------------
    # Save ROOT histograms with sum-of-weights-squared errors
    # --------------------------------------------------
    fout = ROOT.TFile( plot_dir + name + "_hists.root", "RECREATE")

    for hist_name, values, errors in [
        ("signal", h_sig, err_sig),
        ("background", h_bkg, err_bkg),
    ]:
        hist = ROOT.TH1D(
            f"{name}_{hist_name}_mvaHist",
            f"{hist_name};BDT output;Weighted events",
            nBins, c_min, c_max,
        )
        hist.Sumw2()

        for i, (value, error) in enumerate(
            zip(values, errors), start=1
        ):
            hist.SetBinContent(i, float(value))
            hist.SetBinError(i, float(error))

        hist.Write()

    fout.Close()

    # --------------------------------------------------
    # Export scores to ROOT trees using your existing helper
    # --------------------------------------------------
    exportTree(
        {"BDT": sig_scores},
        "Tree",
        export_filename=plot_dir + name + "_trees_sig.root",
    )

    exportTree(
        {"BDT": bkg_scores},
        "Tree",
        export_filename=plot_dir + name + "_trees_bkg.root",
    )

    print(f"Mass hypothesis: {mass} GeV")
    print(f"Signal events: {len(sig_scores)}")
    print(f"Background events: {len(bkg_scores)}")
    print(f"Signal weighted yield: {sig_weights.sum():.6g}")
    print(f"Background weighted yield: {bkg_weights.sum():.6g}")

def drawScoresInferenceROOT(
    df_sig,
    df_bkg,
    mass,
    name,
    nBins=25,
    c_min=0.0,
    c_max=1.0,
    weight_col="weight",
    process_col="process_id",
    process_map=None,
):

    ROOT.gStyle.SetOptStat(0)
    """
    Produce:
      1. Normalized signal/background overlay
      2. Absolute-yield stacked background with signal overlay
      3. Normalized stacked background with signal overlay

    Save:
      name_normalized.root
      name_stack.root
      name_normalized.png/pdf
      name_stack.png/pdf
      name_normalized_stack.png/pdf
    """

    if process_map is None:
        process_map = {}

    # --------------------------------------------------
    # Select events
    # --------------------------------------------------

    sig = df_sig.loc[
        np.isclose(df_sig["mass_point"], mass)
    ].copy()

    bkg = df_bkg.copy()

    sig_scores = sig["bdt_score"].to_numpy(dtype=float)
    sig_weights = sig[weight_col].to_numpy(dtype=float)

    bkg_scores = bkg[f"bdt_{mass}"].to_numpy(dtype=float)
    bkg_weights = bkg[weight_col].to_numpy(dtype=float)

    sig_mask = (
        np.isfinite(sig_scores) &
        np.isfinite(sig_weights)
    )

    bkg_mask = (
        np.isfinite(bkg_scores) &
        np.isfinite(bkg_weights)
    )

    sig_scores = sig_scores[sig_mask]
    sig_weights = sig_weights[sig_mask]

    bkg = bkg.loc[bkg_mask].copy()

    bkg_scores = bkg_scores[bkg_mask]
    bkg_weights = bkg_weights[bkg_mask]

    sig_weights*= signal_xsec * lumi * 1000.0

    process_id_to_name = {
        process_id: process_name
        for process_name, process_id in process_map.items()
    }

    # Map process IDs to process names
    bkg["Process"] = bkg[process_col].map(process_id_to_name)

    # Map process names to cross sections
    bkg_xsecs = bkg["Process"].map(xsecs).to_numpy(dtype=float)

    # Check missing cross sections
    if not np.all(np.isfinite(bkg_xsecs)):
        missing_processes = bkg.loc[
            ~np.isfinite(bkg_xsecs), "Process"
        ].unique()

        raise ValueError(
            f"Missing cross sections for: {missing_processes}"
        )

    # Apply cross-section and luminosity scaling
    bkg_weights *= bkg_xsecs * lumi * 1000.0

    # if np.any(sig_weights < 0) or np.any(bkg_weights < 0):
    #     raise ValueError("Negative event weights found.")

    # --------------------------------------------------
    # Histogram creation
    # --------------------------------------------------

    def make_hist(hist_name, title):
        hist = ROOT.TH1D(
            hist_name,
            title,
            nBins,
            c_min,
            c_max
        )
        hist.Sumw2()
        hist.SetDirectory(0)
        return hist

    h_sig = make_hist(
        f"h_signal_m{mass}",
        ";BDT output;Weighted events"
    )

    h_bkg = make_hist(
        f"h_background_m{mass}",
        "Background;BDT output;Weighted events"
    )

    # Fill signal
    for score, weight in zip(sig_scores, sig_weights):
        h_sig.Fill(float(score), float(weight))

    # Fill total background
    for score, weight in zip(bkg_scores, bkg_weights):
        h_bkg.Fill(float(score), float(weight))

    # --------------------------------------------------
    # Grouped background components
    # --------------------------------------------------

    h_components = {}

    for group_name, group_info in background_groups.items():

        hist = make_hist(
            f"h_group_{len(h_components)}_m{mass}",
            f"{group_name};BDT output;Weighted events"
        )

        # Convert process names to integer IDs
        process_ids = [
            process_map[p]
            for p in group_info["processes"]
        ]

        mask = bkg[process_col].isin(process_ids).to_numpy()

        for score, weight in zip(
            bkg_scores[mask],
            bkg_weights[mask]
        ):
            hist.Fill(float(score), float(weight))

        hist.SetFillColor(group_info["color"])
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)

        h_components[group_name] = hist

    # --------------------------------------------------
    # Process labels
    # --------------------------------------------------

    def process_label(process):
        return process_map.get(process, str(process))

    # --------------------------------------------------
    # 1. Normalized overlay
    # --------------------------------------------------

    h_sig_norm = h_sig.Clone(f"h_signal_norm_m{mass}")
    h_bkg_norm = h_bkg.Clone(f"h_background_norm_m{mass}")

    h_sig_norm.SetDirectory(0)
    h_bkg_norm.SetDirectory(0)

    if h_sig_norm.Integral() > 0:
        h_sig_norm.Scale(1.0 / h_sig_norm.Integral())

    if h_bkg_norm.Integral() > 0:
        h_bkg_norm.Scale(1.0 / h_bkg_norm.Integral())

    h_sig_norm.SetLineColor(ROOT.kBlue)
    h_sig_norm.SetLineWidth(2)
    h_sig_norm.SetFillColorAlpha(ROOT.kBlue, 0.25)

    h_bkg_norm.SetLineColor(ROOT.kRed)
    h_bkg_norm.SetLineWidth(2)
    h_bkg_norm.SetFillColorAlpha(ROOT.kRed, 0.25)

    c1 = ROOT.TCanvas(
        f"c_normalized_{mass}", "", 1000, 700
    )

    h_sig_norm.GetXaxis().SetTitle("BDT output")
    h_sig_norm.GetYaxis().SetTitle("Normalized weighted events")

    h_sig_norm.SetMaximum(
        1.25 * max(
            h_sig_norm.GetMaximum(),
            h_bkg_norm.GetMaximum()
        )
    )

    h_sig_norm.Draw("HIST")
    h_bkg_norm.Draw("HIST SAME")

    # 1. Normalized overlay
    stat_boxes_overlay = draw_all_statboxes(
        h_sig_norm, h_components,
        mass, ROOT.kBlack, []
    )

    CMS_label(c1, lumi=lumi, y=0.92)

    c1.SaveAs(plot_dir + name + "_normalized.png")
    c1.SaveAs(plot_dir + name + "_normalized.pdf")

    # --------------------------------------------------
    # 2. Absolute-yield stacked background
    # --------------------------------------------------

    stack = ROOT.THStack(
        f"stack_{mass}",
        f";BDT output;Weighted events"
    )

    for group_name, hist in h_components.items():

        hist.SetFillColor(
            background_groups[group_name]["color"]
        )

        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)

        stack.Add(hist)

    h_sig.SetLineColor(ROOT.kBlack)
    h_sig.SetLineWidth(3)
    h_sig.SetFillStyle(0)

    c2 = ROOT.TCanvas(
        f"c_stack_{mass}", "", 1000, 700
    )

    c2.SetLogy()

    stack.Draw("HIST")
    stack.GetXaxis().SetTitle("BDT output")
    stack.GetYaxis().SetTitle("Weighted events")

    stack.SetMinimum(0.001)

    stack.SetMaximum(
        100 * max(
            stack.GetMaximum(),
            h_sig.GetMaximum()
        )
    )

    h_sig.Draw("HIST SAME")

    # 2. Absolute-yield stack
    stat_boxes_stack = draw_all_statboxes(
        h_sig, h_components,
        mass, ROOT.kBlack, []
    )

    CMS_label(c2, lumi=lumi, y=0.92)

    c2.SaveAs(plot_dir + name + "_stack.png")
    c2.SaveAs(plot_dir + name + "_stack.pdf")

    # --------------------------------------------------
    # 3. Normalized stacked background
    # --------------------------------------------------

    h_components_norm = {}

    total_bkg = sum(
        hist.Integral()
        for hist in h_components.values()
    )

    for group_name, hist in h_components.items():

        h_norm = hist.Clone(
            f"h_group_{len(h_components_norm)}_norm_m{mass}"
        )

        h_norm.SetDirectory(0)

        if total_bkg > 0:
            h_norm.Scale(1.0 / total_bkg)

        h_components_norm[group_name] = h_norm

    h_sig_unit = h_sig.Clone(
        f"h_signal_unit_m{mass}"
    )

    h_sig_unit.SetDirectory(0)

    if h_sig_unit.Integral() > 0:
        h_sig_unit.Scale(1.0 / h_sig_unit.Integral())

    stack_norm = ROOT.THStack(
        f"stack_norm_{mass}",
        f";BDT output;Normalized weighted events"
    )

    for group_name, hist in h_components_norm.items():

        hist.SetFillColor(
            background_groups[group_name]["color"]
        )

        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)

        stack_norm.Add(hist)

    h_sig_unit.SetLineColor(ROOT.kBlack)
    h_sig_unit.SetLineWidth(3)
    h_sig_unit.SetFillStyle(0)

    c3 = ROOT.TCanvas(
        f"c_normalized_stack_{mass}", "", 1000, 700
    )

    stack_norm.Draw("HIST")
    stack_norm.GetXaxis().SetTitle("BDT output")
    stack_norm.GetYaxis().SetTitle("Normalized weighted events")

    # 3. Normalized stack
    stat_boxes_norm_stack = draw_all_statboxes(
        h_sig_unit, h_components_norm,
        mass, ROOT.kBlack, []
    )

    stack_norm.SetMaximum(
        1.25 * max(
            stack_norm.GetMaximum(),
            h_sig_unit.GetMaximum()
        )
    )

    h_sig_unit.Draw("HIST SAME")

    CMS_label(c3, lumi=lumi, y=0.92)

    c3.SaveAs(plot_dir + name + "_normalized_stack.png")
    c3.SaveAs(plot_dir + name + "_normalized_stack.pdf")

    # --------------------------------------------------
    # Save ROOT histograms
    # --------------------------------------------------

    fout = ROOT.TFile(
        plot_dir + name + "_normalized.root",
        "RECREATE"
    )

    h_sig_norm.Write()
    h_bkg_norm.Write()

    fout.Close()

    fout = ROOT.TFile(
        plot_dir + name + "_stack.root",
        "RECREATE"
    )

    h_sig.Write()
    h_bkg.Write()

    for process, hist in h_components.items():
        hist.Write()

    for process, hist in h_components_norm.items():
        hist.Write()

    h_sig_unit.Write()

    fout.Close()

    # --------------------------------------------------
    # Print yields
    # --------------------------------------------------

    print(f"\nMass hypothesis: {mass} GeV")

    print(f"Signal events: {len(sig_scores)}")
    print(f"Background events: {len(bkg_scores)}")

    print(f"Signal weighted yield: {h_sig.Integral():.6g}")
    print(f"Background weighted yield: {h_bkg.Integral():.6g}")

    print("\nBackground components:")

    for process, hist in h_components.items():
        print(
            f"{process_label(process):25s} "
            f"{hist.Integral():.6g}"
        )

    print("\nAll plots and ROOT histograms saved.")


def drawROCInference(
    df_sig,
    df_bkg,
    mass,
    name,
    weight_col="evt_wgt",
):
    """
    Weighted ROC for one mass hypothesis.

    Signal events are selected at their corresponding mass_point.
    Background events use their BDT_<mass> score.
    """

    sig = df_sig.loc[
        np.isclose(df_sig["mass_point"], mass)
    ].copy()

    sig_scores = sig["bdt_score"].to_numpy(dtype=float)
    bkg_scores = df_bkg[f"bdt_{mass}"].to_numpy(dtype=float)

    sig_weights = sig[weight_col].to_numpy(dtype=float)
    bkg_weights = df_bkg[weight_col].to_numpy(dtype=float)

    sig_mask = (
        np.isfinite(sig_scores) & np.isfinite(sig_weights)
    )
    bkg_mask = (
        np.isfinite(bkg_scores) & np.isfinite(bkg_weights)
    )

    sig_scores, sig_weights = (
        sig_scores[sig_mask], sig_weights[sig_mask]
    )
    bkg_scores, bkg_weights = (
        bkg_scores[bkg_mask], bkg_weights[bkg_mask]
    )

    if (
        np.any(sig_weights < 0)
        or np.any(bkg_weights < 0)
    ):
        raise ValueError(
            "Negative event weights found. Define a consistent "
            "negative-weight treatment before calculating ROC."
        )

    scores = np.concatenate([sig_scores, bkg_scores])
    labels = np.concatenate([
        np.ones(len(sig_scores), dtype=int),
        np.zeros(len(bkg_scores), dtype=int),
    ])
    weights = np.concatenate([sig_weights, bkg_weights])

    fpr, tpr, thresholds = metrics.roc_curve(
        labels,
        scores,
        sample_weight=weights,
    )

    auc = metrics.auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    # x = signal efficiency; y = background rejection.
    ax.plot(tpr, 1.0 - fpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [1, 0], "--", color="gray")

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection (1 - background efficiency)")
    ax.set_title(f"Weighted ROC ({mass} GeV)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    fig.tight_layout()

    fig.savefig(plot_dir + name + ".png", dpi=300)
    fig.savefig(plot_dir + name + ".pdf", dpi=300)
    plt.close(fig)

    print(f"Mass hypothesis: {mass} GeV")
    print(f"Weighted AUC: {auc:.5f}")

    return auc




with open(model_file, "rb") as f:
    modelDict = pickle.load(f)

model = modelDict["xgbModel"]

print("Loaded model:", modelDict.get("modelName", model_file))

# --------------------------------------------------
# Load signal and background events
# --------------------------------------------------

# These are your existing functions from the training script.
# They should return pandas DataFrames.

print("Loading signal...")
df_sig = loadSignal(sig_file_name, ["evt_wgt"] + inputVars)

print("Loading background...")
df_bkg = loadMCBackground(bkg_file_name,
    ["evt_wgt"] + inputVars,
    ["evt_wgt"] + inputVars
)

# Work on copies so the original DataFrames are unchanged.
df_sig = df_sig.copy()
df_bkg = df_bkg.copy()

# --------------------------------------------------
# Common inference function
# --------------------------------------------------

def predict_bdt(df, mass_point=None):
    """
    Predict BDT scores for all rows in df.

    If mass_point is provided, override the mass_point feature
    for every event with the requested mass hypothesis.
    Otherwise, use each event's existing mass_point.

    Returns a NumPy array of BDT scores.
    """

    missing = [v for v in inputVars if v not in df.columns]
    if missing:
        raise KeyError(f"Missing BDT input features: {missing}")

    # Select features in the exact training order.
    X = df[inputVars].copy()

    # Optionally evaluate at a particular mass hypothesis.
    if mass_point is not None:
        X.loc[:, "mass_point"] = mass_point

    # XGBoost can handle NaN values, but not infinite values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Convert features to numeric arrays.
    X_array = X.to_numpy(dtype=np.float32)

    dmatrix = xgb.DMatrix(
        X_array,
        feature_names=inputVars,
        missing=np.nan
    )

    return model.predict(dmatrix)


# --------------------------------------------------
# 1. Signal inference
# --------------------------------------------------

print("Scoring signal events...")

# Each signal event retains its existing mass_point.
# One score per event.
df_sig["bdt_score"] = predict_bdt(df_sig)

# Save all original columns plus BDT_score.
df_sig.to_parquet(output_signal, index=False)

print("Signal events:", len(df_sig))
print("Signal output:", output_signal)


# --------------------------------------------------
# 2. Background inference
# --------------------------------------------------

print("Scoring background events...")

# Evaluate every background event at every mass hypothesis.
# Each mass hypothesis gets its own score column.

for mass in mass_grid:
    column = f"bdt_{mass}"

    print(f"Evaluating background at mass = {mass} GeV")

    df_bkg[column] = predict_bdt(
        df_bkg,
        mass_point=mass
    )

# Optional: store the score for one reference mass in a
# generic BDT_score column. Here, the reference is 50 GeV.

df_bkg.to_parquet(output_background, index=False)



# Load the inference outputs
df_sig = pd.read_parquet("signal_scored.parquet")
df_bkg = pd.read_parquet("background_scored.parquet")

# Apply the training-script convention once to each full sample.
df_sig = make_weights_nonnegative(
    df_sig, label="Signal"
)

df_bkg = make_weights_nonnegative(
    df_bkg, label="Background"
)


os.makedirs("BDT_plots", exist_ok=True)

auc_results = {}

for mass in mass_grid:
    print(f"\nProcessing mass hypothesis {mass} GeV")

    name = f"BDT_plots/BDT_{mass}"

    drawScoresInference(
        df_sig=df_sig,
        df_bkg=df_bkg,
        mass=mass,
        name=name,
        nBins=25,
        c_min=0.0,
        c_max=1.0,
    )

    id_to_process = {
        v: k for k, v in process_map.items()
    }

    drawScoresInferenceROOT(
        df_sig,
        df_bkg,
        mass=mass,
        name=name,
        process_col="process_id",
        process_map=process_map
    )

    auc_results[mass] = drawROCInference(
        df_sig=df_sig,
        df_bkg=df_bkg,
        mass=mass,
        name=f"BDT_plots/ROC_{mass}",
    )

auc_results = drawROCOverlay(
    df_sig=df_sig,
    df_bkg=df_bkg,
    mass_grid=mass_grid,
    name="BDT_plots/ROC_all_mass_points",
)

print("\nSummary of weighted AUC values:")
for mass, auc in auc_results.items():
    print(f"mA = {mass:2d} GeV: AUC = {auc:.5f}")


for mass in signal_mass_points:
    for obs_name in variables_list:

        if obs_name not in df_sig.columns:
            print(f"Skipping {obs_name}: not in signal Parquet")
            continue

        if obs_name not in df_bkg.columns:
            print(f"Skipping {obs_name}: not in background Parquet")
            continue

        make_stack_after_bdt_cut_with_overlay(
            obs_name=obs_name,
            mass=mass,
            df_sig=df_sig,
            df_bkg=df_bkg,
            bdt_cut=bdt_cut, bdt_operator=bdt_operator
        )


print("Background events:", len(df_bkg))
print("Background output:", output_background)

print("Inference completed!")
