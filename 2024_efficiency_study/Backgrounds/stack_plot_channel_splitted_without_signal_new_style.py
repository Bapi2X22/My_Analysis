import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import ROOT
from glob import glob
import pyarrow.parquet as pq

ROOT.gROOT.SetBatch(True)   # (you are saving files → good practice)
ROOT.gStyle.SetOptStat(1110)
ROOT.gROOT.ForceStyle()

# Plot_dir = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check/Before_BDT/"
# Plot_dir_BDT = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check/After_BDT/"

Plot_dir_ele = "/eos/user/b/bbapi/www/Analysis_plots/DATA_MC_validation/Cutflow/Overlap_removed_btag/TTbar_treatment_dR_with_sig/For_presentation/ele/"
Plot_dir_mu = "/eos/user/b/bbapi/www/Analysis_plots/DATA_MC_validation/Cutflow/Overlap_removed_btag/TTbar_treatment_dR_with_sig/For_presentation/mu/"

# Plot_dir = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check_without_DY/Before_BDT/"
# Plot_dir_BDT = "/eos/user/b/bbapi/www/Analysis_plots/BDT/variables_check_without_DY/After_BDT/"

def CMS_label(pad,
            #   lumi="39.05 fb^{-1}",
              lumi=109.0,
              year="2024",
              energy="13.6 TeV",
              status="Simulation Preliminary",
              x=0.12,
              y=0.95):

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


def get_global_range(arrays, padding=0.05, invalid_value=-999.0):

    mins = []
    maxs = []

    for arr in arrays:

        arr = ak.to_numpy(ak.flatten(arr, axis=None))

        # Remove invalid entries
        arr = arr[arr != invalid_value]

        if len(arr) == 0:
            continue

        mins.append(np.min(arr))
        maxs.append(np.max(arr))

    if len(mins) == 0:
        return 0., 1.

    xmin = min(mins)
    xmax = max(maxs)

    width = xmax - xmin

    xmin -= padding * width
    xmax += padding * width

    return xmin, xmax


def get_mgg(one_photon, category_mask):

    cat_ph = one_photon[category_mask]

    sorted_ph = cat_ph[ak.argsort(cat_ph.pt, ascending=False)]
    padded_ph = ak.pad_none(sorted_ph, 2)

    lead = padded_ph[:, 0]
    sub  = padded_ph[:, 1]

    valid_mask = (
        ~ak.is_none(lead.pt) &
        ~ak.is_none(sub.pt)
    )

    lead = lead[valid_mask]
    sub  = sub[valid_mask]

    lead_p4 = ak.zip(
        {
            "pt": lead.pt,
            "eta": lead.eta,
            "phi": lead.phi,
            "mass": ak.zeros_like(lead.pt),
        },
        with_name="Momentum4D",
    )

    sub_p4 = ak.zip(
        {
            "pt": sub.pt,
            "eta": sub.eta,
            "phi": sub.phi,
            "mass": ak.zeros_like(sub.pt),
        },
        with_name="Momentum4D",
    )

    diphoton = lead_p4 + sub_p4

    return ak.to_numpy(diphoton.mass)


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

    nbins = hist.GetNbinsX()

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

# ==============================
# INPUTS
# ==============================

lumi = 39.05  # fb^-1
# lumi = 32.05  # fb^-1


# Cross sections in pb
xsecs = {
    "WGtoLNuG": 671.5,
    "WtoENu0J": 55850,
    "WtoENu1J": 9177,
    "WtoENu2J": 3474,
    "WtoMuNu0J": 55920,
    "WtoMuNu1J": 9202,
    "WtoMuNu2J": 3490,
    "DYto2Mu10": 21190.0,
    "DYto2Mu50": 2124.08,
    "DYto2E10": 21140.0,
    "DYto2E50": 2124.08,
    "DYGto2LG4": 88.13,
    "DYGto2LG50": 126.7,
    "TTG1Jets": 4.634,
    "TTto2L2Nu": 98.04,
    "TTtoLNu2Q": 405.87
}

signal_mass_points = [20, 35, 55]

signal_colors = {
    signal_mass_points[0]: ROOT.kMagenta + 1,
    signal_mass_points[1]: ROOT.kOrange + 7,
    signal_mass_points[2]: ROOT.kCyan + 1,
}

WH_xsec = 1.457       # pb
signal_BR = 0.01      # BR(H -> AA -> bbgg)

signal_xsec = WH_xsec * signal_BR

variables_list = [
    # Leptons
    "electron_eta",
    "electron_phi",
    "electron_pt",
    "muon_eta",
    "muon_phi",
    "muon_pt",

    # Photons
    "pholead_eta",
    "pholead_phi",
    "pholead_pt",
    "phosublead_eta",
    "phosublead_phi",
    "phosublead_pt",

    # Jets
    "first_jet_eta",
    "first_jet_phi",
    "first_jet_pt",
    "second_jet_eta",
    "second_jet_phi",
    "second_jet_pt",

    #multiplicity
    "n_bJets",
    "Njets",

    #MET
    "PuppiMET_sumEt",
    "PuppiMET_pt",
    "PuppiMET_phi",
    "delphi_ggMET_Puppi"
]

plot_config = {
    # Leptons
    "electron_pt": {
        "range": (0.0, 200.0),
        "nbins": 40,
        "xlabel": "Electron p_{T} [GeV]",
    },
    "muon_pt": {
        "range": (0.0, 200.0),
        "nbins": 40,
        "xlabel": "Muon p_{T} [GeV]",
    },
    "electron_eta": {
        "range": (-2.5, 2.5),
        "nbins": 25,
        "xlabel": "Electron #eta",
    },
    "muon_eta": {
        "range": (-2.5, 2.5),
        "nbins": 25,
        "xlabel": "Muon #eta",
    },
    "electron_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,
        "xlabel": "Electron #phi",
    },
    "muon_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,
        "xlabel": "Muon #phi",
    },

    # Photons
    "pholead_pt": {
        "range": (0.0, 200.0),
        "nbins": 40,
        "xlabel": "Leading photon p_{T} [GeV]",
    },
    "phosublead_pt": {
        "range": (0.0, 200.0),
        "nbins": 40,
        "xlabel": "Subleading photon p_{T} [GeV]",
    },
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
    "pholead_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,
        "xlabel": "Leading photon #phi",
    },
    "phosublead_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,
        "xlabel": "Subleading photon #phi",
    },

    # Jets
    "first_jet_pt": {
        "range": (0.0, 200.0),
        "nbins": 40,
        "xlabel": "Leading b-Jet p_{T} [GeV]",
    },
    "second_jet_pt": {
        "range": (0.0, 200.0),
        "nbins": 40,
        "xlabel": "Subleading b-Jet p_{T} [GeV]",
    },
    "first_jet_eta": {
        "range": (-5.0, 5.0),
        "nbins": 25,
        "xlabel": "Leading b-Jet #eta",
    },
    "second_jet_eta": {
        "range": (-5.0, 5.0),
        "nbins": 25,
        "xlabel": "Subleading b-Jet #eta",
    },
    "first_jet_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,
        "xlabel": "Leading b-Jet #phi",
    },
    "second_jet_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,
        "xlabel": "Subleading b-Jet #phi",
    },
    "Njets": {
        "range": (0, 10),
        "nbins": 10,
        "xlabel": "Number of Jets per event",
    },
    "n_bJets": {
        "range": (0, 5),
        "nbins": 5,
        "xlabel": "Number of b-Jets per event",
    },
    "PuppiMET_sumEt": {
        "range": (0, 800),
        "nbins": 40,
        "xlabel": "Total missing transverse energy [GeV]",
    },
    "PuppiMET_pt": {
        "range": (0, 200),
        "nbins": 40,
        "xlabel": "Total missing transverse momentum [GeV]",
    },
    "PuppiMET_phi": {
        "range": (-3.2, 3.2),
        "nbins": 32,
        "xlabel": "Missing transverse momentum #phi",
    },
    "delphi_ggMET_Puppi": {
        "range": (0, 3.2),
        "nbins": 16,
        "xlabel": "#Delta#phi between #gamma#gamma and MET",
    }
}

# ==============================
# PATHS
# ==============================

bkg_base = (
    "/eos/user/b/bbapi/"
    "My_Analysis/2024_efficiency_study/"
    "Backgrounds/NTuples_2024_BKG_overlap_removal_final/merged_all_cat"
)

data_base = (
    "/eos/user/b/bbapi/"
    "My_Analysis/2024_efficiency_study/"
    "Data/NTuples_2024_Data_skim_lepCorr/merged"
)


# ==============================
# CATEGORIES
# ==============================

# categories = ["CAT1", "Inclusive", "twopho"]
categories = ["Resolved", "1-bJet", "Inclusive"]
channels = ["electron", "muon"]


# ==============================
# BACKGROUND CONFIGURATION
# ==============================

background_colors = {
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


processes = {
    process: {
        "file": {
            category: (
                f"{bkg_base}/{process}-24SummerRun3/"
                f"{category}_merged.parquet"
            )
            for category in categories
        },
        "xsec": xsecs[process],
        "color": background_colors[process],
    }
    for process in xsecs
}

signal_base_dir = (
    "/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/"
    "NTuples_WH_2024_all_corrections/merged"
)

signal_mass_points = [20, 35, 55]

signal_processes = {
    mass: {
        "file": {
            category: (
                f"{signal_base_dir}/WH-2024M{mass}/"
                f"{category}_merged.parquet"
            )
            for category in categories
        }
    }
    for mass in signal_mass_points
}

# ==============================
# DATA FILES
# ==============================

data_periods = {
    "electron": [
        "Data_EGamma-Data-2024E",
        "Data_EGamma-Data-2024F",
    ],
    "muon": [
        "Data_Muon-Data-2024E",
        "Data_Muon-Data-2024F",
    ],
}


def get_data_files(channel, category):
    """Return a flat list of data Parquet files."""

    channel_name = (
        "EGamma"
        if channel == "electron"
        else "Muon"
    )

    files = []

    for period in data_periods[channel]:

        pattern = (
            f"{data_base}/{period}/"
            f"{channel_name}-Data-2024*"
            f"_{category}_merged.parquet"
        )

        files.extend(glob(pattern))

    return sorted(files)


data_files = {
    channel: {
        category: get_data_files(channel, category)
        for category in categories
    }
    for channel in channels
}


# ==============================
# HELPER FUNCTIONS
# ==============================

def make_hist(name, obs_name):
    """Create an empty histogram."""

    config = plot_config[obs_name]
    xmin, xmax = config["range"]

    hist = ROOT.TH1F(
        name,
        f";{obs_name};Events",
        config["nbins"],
        xmin,
        xmax,
    )

    hist.Sumw2()
    hist.SetStats(1)

    return hist


def fill_hist(hist, values, weights, scale=1.0, overflow=False):
    """Fill a ROOT histogram."""

    xmax = hist.GetXaxis().GetXmax()
    eps = 1e-6

    for value, weight in zip(values, weights):

        if overflow and value >= xmax:
            value = xmax - eps

        hist.Fill(
            float(value),
            float(scale * weight),
        )


def get_channel_mask(channel, electron_pt, muon_pt):
    """Return the electron/muon event selection."""

    if channel == "electron":
        return (
            (electron_pt != -999.)
            & (muon_pt == -999.)
        )

    return (
        (electron_pt == -999.)
        & (muon_pt != -999.)
    )


def fill_parquet_hist(
    hist,
    filenames,
    obs_name,
    columns,
    scale=1.0,
    channel=None,
    overflow=False,
):
    """
    Read one or more Parquet files in batches and fill a histogram.

    filenames may be either a single filename or a list of filenames.
    """

    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:

        if not isinstance(filename, str):
            raise TypeError(
                "Each Parquet input must be a string path. "
                f"Received {type(filename).__name__}: {filename}"
            )

        pf = pq.ParquetFile(filename)

        for batch in pf.iter_batches(
            batch_size=200000,
            columns=columns,
        ):

            events = ak.Array(batch.to_pydict())

            values = np.asarray(events[obs_name])
            weights = np.asarray(events.weight)

            mask = values != -999.

            if channel is not None:

                electron_pt = np.asarray(
                    events.electron_pt
                )

                muon_pt = np.asarray(
                    events.muon_pt
                )

                mask &= get_channel_mask(
                    channel,
                    electron_pt,
                    muon_pt,
                )

            fill_hist(
                hist,
                values[mask],
                weights[mask],
                scale=scale,
                overflow=overflow,
            )


def make_ratio(data_hist, bkg_hist, name):
    """Create Data/MC ratio histogram."""

    ratio = data_hist.Clone(name)
    ratio.SetStats(0)

    for i in range(1, ratio.GetNbinsX() + 1):

        data = data_hist.GetBinContent(i)
        mc = bkg_hist.GetBinContent(i)

        if mc > 0:
            ratio.SetBinContent(i, data / mc)
            ratio.SetBinError(
                i,
                data_hist.GetBinError(i) / mc,
            )
        else:
            ratio.SetBinContent(i, 0)
            ratio.SetBinError(i, 0)

    return ratio


def draw_ratio(ratio, obs_name):
    """Configure and draw the Data/MC ratio."""

    ratio.SetTitle("")

    ratio.GetYaxis().SetTitle("Data/MC")
    ratio.GetYaxis().CenterTitle()

    ratio.GetYaxis().SetNdivisions(505)
    ratio.GetYaxis().SetTitleSize(0.09)
    ratio.GetYaxis().SetTitleOffset(0.45)
    ratio.GetYaxis().SetLabelSize(0.08)

    # ratio.GetXaxis().SetTitle(obs_name)
    ratio.GetXaxis().SetTitle(plot_config[obs_name]["xlabel"])
    ratio.GetXaxis().SetTitleSize(0.12)
    ratio.GetXaxis().SetLabelSize(0.10)

    ratio.SetMinimum(0.0)
    ratio.SetMaximum(2.0)

    ratio.SetMarkerStyle(20)
    ratio.SetLineColor(ROOT.kBlack)

    ratio.Draw("PE")

    line = ROOT.TLine(
        ratio.GetXaxis().GetXmin(),
        1.0,
        ratio.GetXaxis().GetXmax(),
        1.0,
    )

    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw("same")

    return line


def draw_plot(
    stack,
    h_data,
    signal_hists,
    channel,
    obs_name,
    category,
):
    """Draw and save the stacked MC plus data plot."""

    use_logy = True

    canvas = ROOT.TCanvas(
        f"c_{channel}_{obs_name}_{category}",
        "",
        1000,
        800,
    )

    pad1 = ROOT.TPad(
        f"pad1_{channel}_{obs_name}_{category}",
        "pad1",
        0.0,
        0.30,
        1.0,
        1.0,
    )

    pad2 = ROOT.TPad(
        f"pad2_{channel}_{obs_name}_{category}",
        "pad2",
        0.0,
        0.00,
        1.0,
        0.30,
    )

    pad1.SetBottomMargin(0.02)
    pad1.SetLeftMargin(0.12)
    # pad1.SetRightMargin(0.15)

    pad2.SetTopMargin(0.05)
    pad2.SetBottomMargin(0.35)
    pad2.SetLeftMargin(0.12)
    # pad2.SetRightMargin(0.15)
    pad2.SetGridy()

    if use_logy:
        pad1.SetLogy()

    pad1.Draw()
    pad2.Draw()

    pad1.cd()

    stack.Draw("hist")

    stack.GetXaxis().SetLabelSize(0)
    stack.GetXaxis().SetTitleSize(0)

    if use_logy:
        stack.SetMinimum(0.1)
        if "n_bjets" in obs_name.lower():
            stack.SetMaximum(stack.GetMaximum() * 20.0)
        else:
            stack.SetMaximum(stack.GetMaximum() * 10.0)

    elif "mass" in obs_name.lower():
        stack.SetMaximum(stack.GetMaximum() * 1.65)
    elif "pt" in obs_name.lower():
        stack.SetMaximum(stack.GetMaximum() * 1.6)
    else:
        stack.SetMaximum(stack.GetMaximum() * 1.4)

    h_bkg = stack.GetStack().Last().Clone(
        f"hBkg_{channel}_{obs_name}_{category}"
    )

    h_data.Draw("PE SAME")

    for mass, h_sig in signal_hists.items():

        h_sig.SetLineColor(signal_colors[mass])
        h_sig.SetLineWidth(3)
        h_sig.SetLineStyle(1)
        h_sig.SetFillStyle(0)

        h_sig.Draw("HIST SAME")

    ratio = make_ratio(
        h_data,
        h_bkg,
        f"ratio_{channel}_{obs_name}_{category}",
    )

    stat_boxes = []

    # Upper row: Background + Data
    top_y2 = 0.88
    top_y1 = 0.76

    stat_boxes.append(
        draw_statbox_manual(
            h_bkg,
            0.39, top_y1, 0.64, top_y2,
            ROOT.kBlack,
            label="Background"
        )
    )

    stat_boxes.append(
        draw_statbox_manual(
            h_data,
            0.64, top_y1, 0.90, top_y2,
            ROOT.kBlack,
            label="Data"
        )
    )


    # ---------------------------------------------------------
    # Individual background components
    # ---------------------------------------------------------
    hists = stack.GetHists()

    # box_top = y_top - 2 * height - gap

    h_WGJets = []
    h_DYG = []
    h_TTG = []

    for i in range(hists.GetSize()):

        h = hists.At(i)

        # Extract only the sample name
        suffix = f"_{channel}_{obs_name}_{category}"
        label = h.GetName().replace(suffix, "")

        if label in ["WtoENu0J", "WtoENu1J", "WtoENu2J", "WtoMuNu0J", "WtoMuNu1J", "WtoMuNu2J", "WGtoLNuG"]:
            h_WGJets.append(h)
        if label in ["DYto2E10", "DYto2E50", "DYto2Mu10", "DYto2Mu50", "DYGto2LG4", "DYGto2LG50"]:
            h_DYG.append(h)
        if label in ["TTG1Jets", "TTto2L2Nu", "TTtoLNu2Q"]:
            h_TTG.append(h)

    # Bottom row: 3 combined processes
    bottom_y2 = 0.76
    bottom_y1 = 0.66

    stat_boxes.append(draw_statbox_combined(h_WGJets,0.39, bottom_y1, 0.56, bottom_y2,ROOT.kGreen + 2,label="W#gamma + jets"))
    stat_boxes.append(draw_statbox_combined(h_DYG,0.56, bottom_y1, 0.73, bottom_y2,ROOT.kBlue + 1,label="Z#gamma + jets"))
    stat_boxes.append(draw_statbox_combined(h_TTG,0.73, bottom_y1, 0.90, bottom_y2,ROOT.kRed + 1,label="t#bar{t}#gamma + jets"))

    signal_stat_boxes = []

    signal_box_positions = {
        signal_mass_points[0]: (0.15, 0.69, 0.23, 0.80),
        signal_mass_points[1]: (0.23, 0.69, 0.31, 0.80),
        signal_mass_points[2]: (0.31, 0.69, 0.38, 0.80),
    }

    for mass, h_sig in signal_hists.items():

        x1, y1, x2, y2 = signal_box_positions[mass]

        signal_stat_boxes.append(
            draw_statbox_manual_small(
                h_sig,
                x1, y1, x2, y2,
                signal_colors[mass],
                label=f"M_{{A}}{mass}",
                is_signal=True
            )
        )

    latex = ROOT.TLatex()

    latex.SetNDC()
    latex.SetTextSize(0.050)
    latex.SetTextFont(42)

    xmin, xmax = plot_config[obs_name]["range"]
    nbins = plot_config[obs_name]["nbins"]

    bin_width = (xmax - xmin) / nbins

    latex.DrawLatex(0.15, 0.86, f"Category {category}")
    latex.DrawLatex(0.15, 0.81, f"binWidth = {bin_width:.2f}")

    CMS_label(
        canvas,
        lumi=lumi,
    )

    pad2.cd()

    line = draw_ratio(
        ratio,
        obs_name,
    )

    canvas._stat_boxes = stat_boxes
    canvas._line = line
    # canvas._legend = legend

    canvas.Modified()
    canvas.Update()

    plot_dir = (
        Plot_dir_ele
        if channel == "electron"
        else Plot_dir_mu
    )

    print(
        f"Saving {channel}: "
        f"{obs_name}, {category}"
    )

    canvas.SaveAs(
        f"{plot_dir}/"
        f"stacked_{obs_name}_{category}_linear.png"
    )

    canvas.SaveAs(
        f"{plot_dir}/"
        f"stacked_{obs_name}_{category}_linear.pdf"
    )


# ==============================
# MAIN LOOP
# ==============================

for channel in channels:

    for obs_name in variables_list:

        overflow = ("pt" in obs_name.lower()) or ("sumet" in obs_name.lower())

        for category in categories:

            stack = ROOT.THStack(
                f"stack_{channel}_{obs_name}_{category}",
                f";{obs_name};Events",
            )

            mc_columns = [
                obs_name,
                "weight",
                "electron_pt",
                "muon_pt",
            ]

            for process_name, process in processes.items():

                hist = make_hist(
                    f"{process_name}_{channel}_{obs_name}_{category}",
                    obs_name,
                )

                scale = (
                    process["xsec"]
                    * lumi
                    * 1000.0
                )

                fill_parquet_hist(
                    hist=hist,
                    filenames=process["file"][category],
                    obs_name=obs_name,
                    columns=mc_columns,
                    scale=scale,
                    channel=channel,
                    overflow=overflow,
                )

                hist.SetFillColor(
                    process["color"]
                )
                hist.SetLineColor(0)
                hist.SetLineWidth(0)

                stack.Add(hist)

            h_data = make_hist(
                f"data_{channel}_{obs_name}_{category}",
                obs_name,
            )

            fill_parquet_hist(
                hist=h_data,
                filenames=data_files[channel][category],
                obs_name=obs_name,
                columns=[
                    obs_name,
                    "weight",
                ],
                scale=1.0,
                channel=None,
                overflow=overflow,
            )

            h_data.SetMarkerStyle(20)
            h_data.SetMarkerColor(ROOT.kBlack)
            h_data.SetLineColor(ROOT.kBlack)

            signal_hists = {}

            for mass in signal_mass_points:

                h_sig = make_hist(
                    f"signal_mA{mass}_{channel}_{obs_name}_{category}",
                    obs_name,
                )

                signal_process = signal_processes[mass]

                signal_scale = (
                    signal_xsec
                    * lumi
                    * 1000.0
                )

                fill_parquet_hist(
                    hist=h_sig,
                    filenames=signal_process["file"][category],
                    obs_name=obs_name,
                    columns=mc_columns,
                    scale=signal_scale,
                    channel=channel,
                    overflow=overflow,
                )

                signal_hists[mass] = h_sig

            draw_plot(
                stack=stack,
                h_data=h_data,
                signal_hists=signal_hists,
                channel=channel,
                obs_name=obs_name,
                category=category,
            )
