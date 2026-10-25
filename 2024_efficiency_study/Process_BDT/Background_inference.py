
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

model_file = "HtoAATo2b2g_Oct23_26_wo_pt_over_mgg_uni_mass_weight.pkl"
output_signal = "signal_scored.parquet"
output_background = "background_scored.parquet"
sig_file_name = "merged_signal_with_allCorr_uniform_mass_weight.root"
bkg_file_name = "merged_bkg_with_allCorr_uniform_mass_weight.root"

mass_grid = np.array(
    [12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
)

lumi = 109.0  # fb^-1

# Exact feature names and order used during training.
inputVars = [
    "leppt", "lepeta",
    "first_jet_eta", "second_jet_eta",
    "pT1_by_mbb", "pT2_by_mbb",
    "n_bJets",
    "pholead_eta", "phosublead_eta",
    "first_jet_B", "second_jet_B",
    "pholead_mvaID", "phosublead_mvaID",
    # "pT1_by_mgg", "pT2_by_mgg",
    "delphi_gg", "delphi_bb", "delphi_bbgg",
    "delphi_ggMET_Puppi", "pt_ratio",
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
              y=0.92):

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


plot_dir = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check_pBDT/After_BDT_0p8_pt_ratio_wo_pt_over_mgg_uni_mass_weight/"
os.makedirs(plot_dir, exist_ok=True)

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

        sig_scores = sig["BDT_score"].to_numpy(dtype=float)
        sig_weights = sig[weight_col].to_numpy(dtype=float)

        # Background evaluated at this mass hypothesis
        bkg_scores = df_bkg[f"BDT_{mass}"].to_numpy(dtype=float)
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
    fig.savefig(name + ".png", dpi=300)
    fig.savefig(name + ".pdf", dpi=300)
    plt.close(fig)

    print("\nAUC summary:")
    for mass, auc in auc_results.items():
        print(f"mA = {mass:2d} GeV: AUC = {auc:.5f}")

    return auc_results



def loadSignal(filename, inputVars):
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


def make_stack_after_bdt_cut(
    obs_name,
    mass,
    df_sig,
    df_bkg,
    bdt_cut,
    bdt_operator=">"
):
    """
    Make a resolved-category MC stack after a BDT cut.

    Signal:
        mass_point == mass and BDT_score > bdt_cut

    Background:
        BDT_<mass> > bdt_cut, split by process_id.

    Background stack order follows process_map exactly.
    """

    score_col = f"bdt_{mass}"

    required_sig = [
        obs_name, "mass_point", "BDT_score", "weight"
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

    # Supported comparison operators
    operators = {
        ">":  lambda x, cut: x > cut,
        "<":  lambda x, cut: x < cut,
        ">=": lambda x, cut: x >= cut,
        "<=": lambda x, cut: x <= cut
    }

    if bdt_operator not in operators:
        raise ValueError(
            f"Invalid bdt_operator: {bdt_operator!r}. "
            f"Choose one of {list(operators)}."
        )

    compare = operators[bdt_operator]

    # Signal selection
    sig_mass = np.isclose(
        pd.to_numeric(df_sig["mass_point"], errors="coerce"),
        mass,
    )
    sig_mask = sig_mass & compare(df_sig["BDT_score"], bdt_cut)
    sig = df_sig.loc[sig_mask].copy()

    # Background selection
    bkg_mask = compare(df_bkg[score_col], bdt_cut)
    bkg = df_bkg.loc[bkg_mask].copy()

    print(
        f"\nObservable: {obs_name} | "
        f"mA = {mass} GeV | BDT {bdt_operator} {bdt_cut}"
    )
    print("Selected signal events:", len(sig))
    print("Selected background events:", len(bkg))

    # ------------------------------------------------
    # Histogram configuration
    # ------------------------------------------------
    config = plot_config[obs_name]
    nbins = config["nbins"]
    xmin, xmax = config["range"]

    stack = ROOT.THStack(
        f"stack_{obs_name}_m{mass}",
        f";{config.get('xlabel', obs_name)};Events",
    )

    background_hists = {}
    output_hists = []

    # The original script uses weights = normalized event
    # weights * cross section * luminosity * 1000.
    # Keep that normalization if evt_wgt has the same meaning.
    for process_name, process_id in process_map.items():

        process_events = bkg.loc[
            bkg["process_id"] == process_id
        ]

        hist = ROOT.TH1D(
            f"h_{process_name}_{obs_name}_m{mass}",
            f";{config.get('xlabel', obs_name)};Events",
            nbins,
            xmin,
            xmax,
        )
        hist.Sumw2()

        values = pd.to_numeric(
            process_events[obs_name], errors="coerce"
        ).to_numpy(dtype=float)

        weights = pd.to_numeric(
            process_events["weight"], errors="coerce"
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(values)
            & np.isfinite(weights)
            & (values != -999)
        )

        values = values[valid]
        weights = weights[valid]

        # Use the same xsec/luminosity scaling as the old script.
        scale = xsecs[process_name] * lumi * 1000.0

        for value, weight in zip(values, weights):
            hist.Fill(float(value), float(weight * scale))

        hist.SetFillColor(process_colors[process_name])
        hist.SetLineWidth(0)

        # Add in process_map order.
        stack.Add(hist)

        background_hists[process_name] = hist
        output_hists.append(hist)

        print(
            f"  {process_name:12s}: "
            f"{len(process_events):8d} events, "
            f"yield = {hist.Integral():.6g}"
        )

    # ------------------------------------------------
    # Signal histogram
    # ------------------------------------------------
    h_sig = ROOT.TH1D(
        f"h_signal_m{mass}_{obs_name}",
        f";{config.get('xlabel', obs_name)};Events",
        nbins,
        xmin,
        xmax,
    )
    h_sig.Sumw2()

    values = pd.to_numeric(
        sig[obs_name], errors="coerce"
    ).to_numpy(dtype=float)

    weights = pd.to_numeric(
        sig["weight"], errors="coerce"
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (values != -999)
    )

    values = values[valid]
    weights = weights[valid]

    signal_scale = signal_xsec * lumi * 1000.0

    for value, weight in zip(values, weights):
        h_sig.Fill(float(value), float(weight * signal_scale))

    h_sig.SetLineColor(ROOT.kBlack)
    h_sig.SetLineWidth(3)
    h_sig.SetFillStyle(0)

    output_hists.append(h_sig)

    # ------------------------------------------------
    # Draw stack and signal overlay
    # ------------------------------------------------
    canvas = ROOT.TCanvas(
        f"c_{obs_name}_m{mass}", "", 1000, 800
    )
    canvas.SetLogy()
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.12)

    stack.Draw("HIST")
    stack.GetXaxis().SetTitle(
        config.get("xlabel", obs_name)
    )
    stack.GetYaxis().SetTitle("Events")

    h_bkg_total = stack.GetStack().Last()

    ymax = max(
        h_bkg_total.GetMaximum(),
        h_sig.GetMaximum(),
    )

    stack.SetMinimum(0.1)
    # stack.SetMaximum(ymax * 1.4 if ymax > 0 else 1.0)
    stack.SetMaximum(stack.GetMaximum() * setmax_factor)

    h_sig.Draw("HIST SAME")


    # ------------------------------------------------
    # Combined stat boxes instead of a legend
    # ------------------------------------------------
    stat_boxes = []

    # Total background
    h_bkg = stack.GetStack().Last().Clone(
        f"hBkg_{obs_name}_m{mass}"
    )
    h_bkg.SetDirectory(0)

    top_y2 = 0.88
    top_y1 = 0.76

    stat_boxes.append(
        draw_statbox_manual(
            h_bkg,
            0.39, top_y1, 0.64, top_y2,
            ROOT.kBlack,
            label="Background",
        )
    )

    # ------------------------------------------------
    # Group the individual stack histograms
    # ------------------------------------------------
    h_WGJets = []
    h_DYG = []
    h_TTG = []

    for process_name in process_map:

        h = background_hists[process_name]

        if process_name in [
            "WGtoLNuG",
            "WtoENu0J",
            "WtoENu2J",
            "WtoMuNu2J",
        ]:
            h_WGJets.append(h)

        elif process_name in [
            "DYto2Mu50",
            "DYto2E50",
            "DYGto2LG4",
            "DYGto2LG50",
        ]:
            h_DYG.append(h)

        elif process_name in [
            "TTG1Jets",
            "TTto2L2Nu",
            "TTtoLNu2Q",
        ]:
            h_TTG.append(h)

    # Three combined background stat boxes
    bottom_y2 = 0.76
    bottom_y1 = 0.66

    stat_boxes.append(
        draw_statbox_combined(
            h_WGJets,
            0.24, bottom_y1, 0.46, bottom_y2,
            ROOT.kGreen + 2,
            label="W#gamma + jets",
        )
    )

    stat_boxes.append(
        draw_statbox_combined(
            h_DYG,
            0.46, bottom_y1, 0.68, bottom_y2,
            ROOT.kBlue + 1,
            label="Z#gamma + jets",
        )
    )

    stat_boxes.append(
        draw_statbox_combined(
            h_TTG,
            0.68, bottom_y1, 0.90, bottom_y2,
            ROOT.kRed + 1,
            label="t#bar{t}#gamma + jets",
        )
    )

    # ------------------------------------------------
    # Signal stat boxes
    # ------------------------------------------------
    signal_stat_boxes = []

    x1, y1, x2, y2 = 0.15, 0.76, 0.38, 0.88

    signal_stat_boxes.append(
        draw_statbox_manual(
            h_sig,
            x1, y1, x2, y2,
            ROOT.kBlack,
            label=f"M_{{A}} = {mass} GeV"
        )
    )

    latex = ROOT.TLatex()

    latex.SetNDC()
    latex.SetTextSize(0.030)
    latex.SetTextFont(42)

    xmin, xmax = plot_config[obs_name]["range"]
    nbins = plot_config[obs_name]["nbins"]

    bin_width = (xmax - xmin) / nbins

    latex.DrawLatex(0.68, 0.87, f"Category {category}")
    latex.DrawLatex(0.68, 0.82, f"binWidth = {bin_width:.2f}")
    latex.DrawLatex(0.68, 0.77, f"BDT_cut {bdt_operator} {bdt_cut:.2f}")


    CMS_label(canvas,lumi=lumi)

    # Keep ROOT objects alive until the canvas is saved.
    canvas._stat_boxes = stat_boxes
    canvas._signal_stat_boxes = signal_stat_boxes

    canvas.Modified()
    canvas.Update()


    # ------------------------------------------------
    # Save plots and histograms
    # ------------------------------------------------
    stem = (
        f"{plot_dir}/stacked_{obs_name}"
        f"_m{mass}_BDT{bdt_cut:.2f}"
    )

    canvas.SaveAs(stem + ".png")
    canvas.SaveAs(stem + ".pdf")

    fout = ROOT.TFile(stem + ".root", "RECREATE")

    for hist in output_hists:
        hist.Write()

    h_bkg_total.Write("total_background")
    canvas.Write("canvas")
    fout.Close()

    print("Total background yield:", h_bkg_total.Integral())
    print("Signal yield:", h_sig.Integral())
    print("Saved:", stem)

    canvas.Close()

    # Keep ROOT objects alive until after saving.
    return


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

    sig_scores = sig["BDT_score"].to_numpy(dtype=float)
    bkg_scores = bkg[f"BDT_{mass}"].to_numpy(dtype=float)

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

    fig.savefig(name + ".png", dpi=300)
    fig.savefig(name + ".pdf", dpi=300)
    plt.close(fig)

    # --------------------------------------------------
    # Save ROOT histograms with sum-of-weights-squared errors
    # --------------------------------------------------
    fout = ROOT.TFile(name + "_hists.root", "RECREATE")

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
        export_filename=name + "_trees_sig.root",
    )

    exportTree(
        {"BDT": bkg_scores},
        "Tree",
        export_filename=name + "_trees_bkg.root",
    )

    print(f"Mass hypothesis: {mass} GeV")
    print(f"Signal events: {len(sig_scores)}")
    print(f"Background events: {len(bkg_scores)}")
    print(f"Signal weighted yield: {sig_weights.sum():.6g}")
    print(f"Background weighted yield: {bkg_weights.sum():.6g}")


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

    sig_scores = sig["BDT_score"].to_numpy(dtype=float)
    bkg_scores = df_bkg[f"BDT_{mass}"].to_numpy(dtype=float)

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

    fig.savefig(name + ".png", dpi=300)
    fig.savefig(name + ".pdf", dpi=300)
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

        make_stack_after_bdt_cut(
            obs_name=obs_name,
            mass=mass,
            df_sig=df_sig,
            df_bkg=df_bkg,
            bdt_cut=bdt_cut, bdt_operator=bdt_operator
        )


print("Background events:", len(df_bkg))
print("Background output:", output_background)

print("Inference completed!")
