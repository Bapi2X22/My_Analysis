from dataclasses import dataclass
import ROOT

@dataclass
class PlotSpec:
    name: str
    data: str
    weight: str
    bins: int
    range: tuple
    xlabel: str
    ylabel: str = "Events"
    normalize: bool = False
    overflow: bool = False

PLOTS = {
    "min_dr": PlotSpec(
        name="min_dr",
        data="min_dr",
        weight="min_dr_weight",
        bins=30,
        range=(0.0, 2.0),
        xlabel=r"Minimum #Delta R(#gamma, GenPart)",
    ),

    "n_photons": PlotSpec(
        name="n_photons",
        data="n_photons",
        weight="n_photons_weight",
        bins=4,
        range=(0.0, 4.0),
        xlabel="Number of gen prompt photons",
    ),

    "n_photons_high": PlotSpec(
        name="n_photons_high",
        data="n_photons_high",
        weight="n_photons_high_weight",
        bins=4,
        range=(0.0, 4.0),
        xlabel=r"Number of gen prompt photons with #Delta R > 0.15",
    ),

    "n_photons_low": PlotSpec(
        name="n_photons_low",
        data="n_photons_low",
        weight="n_photons_low_weight",
        bins=4,
        range=(0.0, 4.0),
        xlabel=r"Number of gen prompt photons with #Delta R < 0.15",
    ),

    "gen_photon_pt": PlotSpec(
        name="gen_photon_pt",
        data="gen_photon_pt",
        weight="gen_photon_pt_weight",
        bins=50,
        range=(0.0, 200.0),
        xlabel=r"Gen photon p_{T} [GeV]",
    ),

    "gen_photon_pt_after_dr_high": PlotSpec(
        name="gen_photon_pt_after_dr_high",
        data="gen_photon_pt_after_dr_high",
        weight="gen_photon_pt_after_dr_high_weight",
        bins=50,
        range=(0.0, 200.0),
        xlabel=r"Gen photon p_{T} after #Delta R > 0.15 [GeV]",
    ),

    "gen_photon_pt_after_dr_low": PlotSpec(
        name="gen_photon_pt_after_dr_low",
        data="gen_photon_pt_after_dr_low",
        weight="gen_photon_pt_after_dr_low_weight",
        bins=50,
        range=(0.0, 200.0),
        xlabel=r"Gen photon p_{T} after #Delta R < 0.15 [GeV]",
    )
}

PROCESS_COLORS = {
    "DYGto2LG": ROOT.kRed + 1,
    "DYto2E": ROOT.kBlue + 1,
    "DYto2Mu": ROOT.kGreen + 2,
    "TTto2L2Nu": ROOT.kOrange + 1,
    "TTtoLNu2Q": ROOT.kMagenta + 1,
    "TTG1Jets": ROOT.kCyan + 1,
    "WGtoLNuG": ROOT.kViolet + 1,
    "WtoENu": ROOT.kAzure + 1,
    "WtoMuNu": ROOT.kTeal + 1,
}



SINGLE_PLOTS = [
    "min_dr",
    "n_photons",
    "n_photons_high",
    "n_photons_low",
    "gen_photon_pt",
    "gen_photon_pt_after_dr_high",
    "gen_photon_pt_after_dr_low"
]


OVERLAY_PLOTS = [
    {
        "name": "min_dr_DY",
        "plot": "min_dr",
        "group": "DY",
    },
    {
        "name": "min_dr_TOP",
        "plot": "min_dr",
        "group": "TOP",
    },
    {
        "name": "min_dr_WJETS",
        "plot": "min_dr",
        "group": "WJetsG",
    },
]

STACK_PLOTS = [
    {
        "name": "min_dr_DY_stack",
        "plot": "min_dr",
        "group": "DY",
    },
    {
        "name": "min_dr_TOP_stack",
        "plot": "min_dr",
        "group": "TOP",
    },
    {
        "name": "min_dr_WJETS_stack",
        "plot": "min_dr",
        "group": "WJetsG",
    },
    {
        "name": "nPho_DY_stack",
        "plot": "n_photons",
        "group": "DY",
    },
    {
        "name": "nPho_TOP_stack",
        "plot": "n_photons",
        "group": "TOP",
    },
    {
        "name": "nPho_WJETS_stack",
        "plot": "n_photons",
        "group": "WJetsG",
    },
    {
        "name": "genPhoPt_DY_stack",
        "plot": "gen_photon_pt",
        "group": "DY",
    },
    {
        "name": "genPhoPt_TOP_stack",
        "plot": "gen_photon_pt",
        "group": "TOP",
    },
    {
        "name": "genPhoPt_WJETS_stack",
        "plot": "gen_photon_pt",
        "group": "WJetsG",
    }
]

STACK_COMPARISONS = [

    {
        "name": "nPho_DY_stack",
        "plot": "n_photons",
        "high": "n_photons_high",
        "low": "n_photons_low",
        "group": "DY",
    },

    {
        "name": "nPho_TOP_stack",
        "plot": "n_photons",
        "high": "n_photons_high",
        "low": "n_photons_low",
        "group": "TOP",
    },
    {
        "name": "nPho_WJETS_stack",
        "plot": "n_photons",
        "high": "n_photons_high",
        "low": "n_photons_low",
        "group": "WJetsG",
    },
    {
        "name": "genPhoPtAfterDR_DY_stack",
        "plot": "gen_photon_pt",
        "high": "gen_photon_pt_after_dr_high",
        "low": "gen_photon_pt_after_dr_low",
        "group": "DY",
    },

    {
        "name": "genPhoPtAfterDR_TOP_stack",
        "plot": "gen_photon_pt",
        "high": "gen_photon_pt_after_dr_high",
        "low": "gen_photon_pt_after_dr_low",
        "group": "TOP",
    },

    {
        "name": "genPhoPtAfterDR_WJETS_stack",
        "plot": "gen_photon_pt",
        "high": "gen_photon_pt_after_dr_high",
        "low": "gen_photon_pt_after_dr_low",
        "group": "WJetsG",
    }

]