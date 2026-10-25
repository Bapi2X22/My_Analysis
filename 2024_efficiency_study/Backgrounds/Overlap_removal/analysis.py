from dataclasses import dataclass
import numpy as np
import os
import awkward as ak
import ROOT
import time
import pyarrow.parquet as pq
import pyarrow as pa

from plot_config import PLOTS

from data_loader import load_process,  get_root_files, get_sum_genweight, load_root_file
from physics import photon_min_dr
from config import DR_CUT 
from higgs_dna.tools.SC_eta import add_photon_SC_eta

# def delta_r_mask(first, second, threshold):
#     mval = first.metric_table(second)
#     return ak.all(mval > threshold, axis=-1)

def delta_r_mask(first, second, threshold):

    pairs = ak.cartesian(
        [first, second],
        axis=1,
        nested=True,
    )

    deta = pairs["0"].eta - pairs["1"].eta
    dphi = np.arctan2(
        np.sin(pairs["0"].phi - pairs["1"].phi),
        np.cos(pairs["0"].phi - pairs["1"].phi),
    )

    dr = np.sqrt(deta**2 + dphi**2)

    return ak.all(dr > threshold, axis=-1)

def photon_preselection_bbgg(photons, events, electrons, muons):
    dr_cut = delta_r_mask(photons, electrons, 0.2)
    dr_cut_muon = delta_r_mask(photons, muons, 0.2)
    return photons[(~photons.pixelSeed) & (photons.pt > 15.0) & (photons.isScEtaEB | photons.isScEtaEE) & dr_cut & dr_cut_muon & ak.where(photons.isScEtaEB, photons.mvaID > 0.0439603, photons.mvaID > -0.249526)]

def select_electrons_bbgg(electrons):
    return (electrons.pt > 30) & (abs(electrons.eta) < 2.5) & electrons.mvaIso_WP80

def select_muons_bbgg(muons):
    return (muons.pt > 26.0) & (abs(muons.eta) < 2.4) & muons.mediumId & (muons.pfIsoId >= 3) & muons.isGlobal


@dataclass
class AnalysisResult:
    min_dr: np.ndarray
    min_dr_weight: np.ndarray
    n_photons: np.ndarray
    n_photons_weight: np.ndarray
    n_photons_high: np.ndarray
    n_photons_high_weight: np.ndarray
    n_photons_low: np.ndarray
    n_photons_low_weight: np.ndarray
    gen_photon_pt: np.ndarray
    gen_photon_pt_weight: np.ndarray
    gen_photon_pt_after_dr_high: np.ndarray
    gen_photon_pt_after_dr_high_weight: np.ndarray
    gen_photon_pt_after_dr_low: np.ndarray
    gen_photon_pt_after_dr_low_weight: np.ndarray
    n_reco_photons: np.ndarray
    n_reco_photons_weight: np.ndarray
    n_reco_photons_high: np.ndarray
    n_reco_photons_high_weight: np.ndarray
    n_reco_photons_low: np.ndarray
    n_reco_photons_low_weight: np.ndarray
    reco_photon_pt: np.ndarray
    reco_photon_pt_weight: np.ndarray
    reco_photon_pt_after_dr_high: np.ndarray
    reco_photon_pt_after_dr_high_weight: np.ndarray
    reco_photon_pt_after_dr_low: np.ndarray
    reco_photon_pt_after_dr_low_weight: np.ndarray

    @classmethod
    def merge(cls, results):
        fields = cls.__dataclass_fields__

        return cls(
            **{
                field: np.concatenate(
                    [getattr(result, field) for result in results]
                )
                for field in fields
            }
        )



@dataclass
class AnalysisResult_skim:

    # ------------------------------------------------------------
    # Gen photon quantities
    # ------------------------------------------------------------

    min_dr: object
    min_dr_weight: object

    n_photons: object
    n_photons_weight: object

    n_photons_high: object
    n_photons_high_weight: object

    n_photons_low: object
    n_photons_low_weight: object

    gen_photon_pt: object
    gen_photon_pt_weight: object

    gen_photon_pt_after_dr_high: object
    gen_photon_pt_after_dr_high_weight: object

    gen_photon_pt_after_dr_low: object
    gen_photon_pt_after_dr_low_weight: object

    # ------------------------------------------------------------
    # Event selections
    # ------------------------------------------------------------

    keep_high: object
    keep_low: object

    # ------------------------------------------------------------
    # Reco photon quantities
    # ------------------------------------------------------------

    n_reco_photons: object
    n_reco_photons_weight: object

    n_reco_photons_high: object
    n_reco_photons_high_weight: object

    n_reco_photons_low: object
    n_reco_photons_low_weight: object

    reco_photon_pt: object
    reco_photon_pt_weight: object

    reco_photon_pt_after_dr_high: object
    reco_photon_pt_after_dr_high_weight: object

    reco_photon_pt_after_dr_low: object
    reco_photon_pt_after_dr_low_weight: object

    @classmethod
    def merge(cls, results):

        if not results:
            raise ValueError("No results to merge")

        fields = cls.__dataclass_fields__

        return cls(
            **{
                field: ak.concatenate(
                    [getattr(result, field) for result in results],
                    axis=0
                )
                for field in fields
            }
        )

def load_parquet_file(filename):
    data = ak.from_parquet(filename)
    table = pq.read_table(filename)
    metadata = table.schema.metadata
    if metadata is None:
        raise RuntimeError(
            f"No metadata found in Parquet file:\n{filename}"
        )

    if b"sum_genw_presel" not in metadata:
        raise RuntimeError(
            f"sum_genw_presel not found in Parquet file:\n{filename}"
        )

    sum_genw_presel = float(
        metadata[b"sum_genw_presel"].decode("utf-8")
    )
    return data, sum_genw_presel

def per_photon_weights(weights, pho_pt):
    counts = ak.to_numpy(ak.num(pho_pt, axis=1))
    w = np.divide(weights, counts, out=np.zeros_like(weights, dtype=float), where=counts > 0)
    return np.repeat(w, counts)


def analyze_events(events,weights,photon_pt_min,photon_eta_max,other_pt_min,parentage_func):
    min_dr, gen_pho_pt = photon_min_dr(events, photon_pt_min=photon_pt_min, photon_eta_max=photon_eta_max, other_pt_min=other_pt_min, parentage_func=parentage_func)
    min_dr_tmp = ak.fill_none(min_dr, np.inf)
    pass_dr_high = min_dr_tmp > DR_CUT
    pass_dr_low = min_dr_tmp < DR_CUT
    n_pho = ak.num(gen_pho_pt, axis=1)
    n_pho_after_dr_high = ak.sum(pass_dr_high, axis=1)
    n_pho_after_dr_low = ak.sum(pass_dr_low, axis=1)
    # keep_high = (n_pho == 0) | (n_pho_after_dr_high > 0)
    keep_high = (n_pho_after_dr_high > 0)
    keep_low = (n_pho == 0) | (n_pho_after_dr_low > 0)
    gen_pho_pt_after_dr_high = gen_pho_pt[pass_dr_high]
    gen_pho_pt_after_dr_low = gen_pho_pt[pass_dr_low]
    n_pho_high = n_pho_after_dr_high[keep_high]
    n_pho_low = n_pho_after_dr_low[keep_low]
    weights = np.asarray(weights)
    photon_counts = ak.to_numpy(ak.num(gen_pho_pt, axis=1))
    photon_counts_after_dr_high = ak.to_numpy(ak.num(gen_pho_pt_after_dr_high, axis=1))
    photon_counts_after_dr_low = ak.to_numpy(ak.num(gen_pho_pt_after_dr_low, axis=1))
    photon_weight = np.divide(weights, photon_counts, out=np.zeros_like(weights, dtype=float), where=photon_counts > 0)
    min_dr_flat = ak.to_numpy(ak.flatten(min_dr_tmp))
    gen_photon_pt_flat = ak.to_numpy(ak.flatten(gen_pho_pt))
    gen_photon_pt_after_dr_high_flat = ak.to_numpy(ak.flatten(gen_pho_pt_after_dr_high))
    gen_photon_pt_after_dr_low_flat = ak.to_numpy(ak.flatten(gen_pho_pt_after_dr_low))
    min_dr_weight = np.repeat(photon_weight, photon_counts)
    gen_photon_pt_weight = np.repeat(photon_weight, photon_counts)
    gen_photon_pt_after_dr_high_weight = np.repeat(photon_weight,photon_counts_after_dr_high)
    gen_photon_pt_after_dr_low_weight = np.repeat(photon_weight,photon_counts_after_dr_low)

    # Reco photon quantities
    events["Photon"] = add_photon_SC_eta(events.Photon, events.PV)
    reco_pho = events.Photon
    reco_pho["mass"] = ak.zeros_like(reco_pho.pt)
    reco_pho["charge"] = ak.zeros_like(reco_pho.pt)
    ele = events.Electron
    mu = events.Muon
    ele = ele[select_electrons_bbgg(ele)]
    mu = mu[select_muons_bbgg(mu)]
    reco_pho = photon_preselection_bbgg(reco_pho, events, ele, mu)

    reco_pho_pt = reco_pho.pt
    reco_pho_counts = ak.to_numpy(ak.num(reco_pho, axis=1))
    reco_pho_pt_flat = ak.to_numpy(ak.flatten(reco_pho_pt))
    # reco_pho_weights = np.divide(weights,reco_pho_counts,out=np.zeros_like(weights, dtype=float),where=reco_pho_counts > 0)
    # reco_pho_pt_weight = np.repeat(reco_pho_weights,reco_pho_counts)
    reco_pho_pt_after_dr_high = reco_pho_pt[keep_high]
    reco_pho_counts_after_dr_high = ak.to_numpy(ak.num(reco_pho_pt_after_dr_high, axis=1))
    reco_pho_pt_after_dr_high_flat = ak.to_numpy(ak.flatten(reco_pho_pt_after_dr_high))
    # weights_high = weights[keep_high]
    # reco_pho_weights_high = np.repeat(weights_high,reco_pho_counts_after_dr_high)
    reco_pho_pt_after_dr_low = reco_pho_pt[keep_low]
    reco_pho_counts_after_dr_low = ak.to_numpy(ak.num(reco_pho_pt_after_dr_low, axis=1))
    reco_pho_pt_after_dr_low_flat = ak.to_numpy(ak.flatten(reco_pho_pt_after_dr_low))
    # weights_low = weights[keep_low]
    # reco_pho_weights_low = np.repeat(weights_low,reco_pho_counts_after_dr_low)
    reco_pho_pt_weight    = per_photon_weights(weights, reco_pho_pt)
    reco_pho_weights_high = per_photon_weights(weights[keep_high], reco_pho_pt[keep_high])
    reco_pho_weights_low  = per_photon_weights(weights[keep_low], reco_pho_pt[keep_low])
    reco_pho_counts = ak.to_numpy(ak.num(reco_pho, axis=1))
    reco_pho_counts_after_dr_high = ak.to_numpy(ak.num(reco_pho_pt_after_dr_high, axis=1))
    reco_pho_counts_after_dr_low = ak.to_numpy(ak.num(reco_pho_pt_after_dr_low, axis=1))

    return AnalysisResult(
        min_dr=min_dr_flat,
        min_dr_weight=min_dr_weight,
        n_photons=np.asarray(n_pho),
        n_photons_weight=weights,
        n_photons_high=np.asarray(n_pho_high),
        n_photons_high_weight=weights[ak.to_numpy(keep_high)],
        n_photons_low=np.asarray(n_pho_low),
        n_photons_low_weight=weights[ak.to_numpy(keep_low)],
        gen_photon_pt=gen_photon_pt_flat,
        gen_photon_pt_weight=gen_photon_pt_weight,
        gen_photon_pt_after_dr_high=gen_photon_pt_after_dr_high_flat,
        gen_photon_pt_after_dr_high_weight=gen_photon_pt_after_dr_high_weight,
        gen_photon_pt_after_dr_low=gen_photon_pt_after_dr_low_flat,
        gen_photon_pt_after_dr_low_weight=gen_photon_pt_after_dr_low_weight,   
        n_reco_photons=reco_pho_counts,
        n_reco_photons_weight=weights,
        n_reco_photons_high=reco_pho_counts_after_dr_high,
        n_reco_photons_high_weight=weights[ak.to_numpy(keep_high)],
        n_reco_photons_low=reco_pho_counts_after_dr_low,
        n_reco_photons_low_weight=weights[ak.to_numpy(keep_low)],
        reco_photon_pt=reco_pho_pt_flat,
        reco_photon_pt_weight=reco_pho_pt_weight,
        reco_photon_pt_after_dr_high=reco_pho_pt_after_dr_high_flat,
        reco_photon_pt_after_dr_high_weight=reco_pho_weights_high,
        reco_photon_pt_after_dr_low=reco_pho_pt_after_dr_low_flat,
        reco_photon_pt_after_dr_low_weight=reco_pho_weights_low
    )


def analyze_events_skim(
    events,
    weights,
    photon_pt_min,
    photon_eta_max,
    other_pt_min,
    parentage_func
):

    # ============================================================
    # Gen photon quantities
    # ============================================================

    min_dr, gen_pho_pt = photon_min_dr(
        events,
        photon_pt_min=photon_pt_min,
        photon_eta_max=photon_eta_max,
        other_pt_min=other_pt_min,
        parentage_func=parentage_func
    )

    # Replace None by infinity.
    # Therefore a photon with no valid matching particle has
    # min_dr = infinity and automatically passes dR > DR_CUT.
    min_dr_tmp = ak.fill_none(min_dr, np.inf)

    # ------------------------------------------------------------
    # Photon-level dR masks
    # ------------------------------------------------------------

    pass_dr_high = min_dr_tmp > DR_CUT
    pass_dr_low = min_dr_tmp < DR_CUT

    # ------------------------------------------------------------
    # Number of gen photons per event
    # ------------------------------------------------------------

    n_pho = ak.num(
        gen_pho_pt,
        axis=1
    )

    # ------------------------------------------------------------
    # Number of photons passing dR requirements per event
    # ------------------------------------------------------------

    n_pho_after_dr_high = ak.sum(
        pass_dr_high,
        axis=1
    )

    n_pho_after_dr_low = ak.sum(
        pass_dr_low,
        axis=1
    )

    # ------------------------------------------------------------
    # Event selections
    # ------------------------------------------------------------

    keep_high = n_pho_after_dr_high > 0

    keep_low = (
        (n_pho == 0)
        | (n_pho_after_dr_low > 0)
    )

    # ------------------------------------------------------------
    # Gen photon arrays after dR selection
    #
    # These remain event-aligned jagged arrays.
    # ------------------------------------------------------------

    gen_pho_pt_after_dr_high = gen_pho_pt[
        pass_dr_high
    ]

    gen_pho_pt_after_dr_low = gen_pho_pt[
        pass_dr_low
    ]

    # ============================================================
    # Event weights
    # ============================================================

    weights = np.asarray(weights)

    # ------------------------------------------------------------
    # Photon counts
    # ------------------------------------------------------------

    photon_counts = ak.to_numpy(
        n_pho
    )

    # ------------------------------------------------------------
    # Per-photon weight
    #
    # Each event weight is divided equally among its gen photons.
    # Events with zero photons receive zero photon weight.
    # ------------------------------------------------------------

    photon_weight = np.divide(
        weights,
        photon_counts,
        out=np.zeros_like(
            weights,
            dtype=float
        ),
        where=photon_counts > 0
    )

    # ------------------------------------------------------------
    # Broadcast event-level photon weight to photon-level arrays
    # ------------------------------------------------------------

    min_dr_weight = ak.broadcast_arrays(
        min_dr_tmp,
        photon_weight
    )[1]

    gen_photon_pt_weight = ak.broadcast_arrays(
        gen_pho_pt,
        photon_weight
    )[1]

    gen_photon_pt_after_dr_high_weight = ak.broadcast_arrays(
        gen_pho_pt_after_dr_high,
        photon_weight
    )[1]

    gen_photon_pt_after_dr_low_weight = ak.broadcast_arrays(
        gen_pho_pt_after_dr_low,
        photon_weight
    )[1]

    # ============================================================
    # Reco photons
    # ============================================================

    reco_pho = events.Photon

    reco_pho_pt = reco_pho.pt

    # ------------------------------------------------------------
    # Number of reco photons per event
    # ------------------------------------------------------------

    reco_pho_counts = ak.num(
        reco_pho_pt,
        axis=1
    )

    # ------------------------------------------------------------
    # Preserve original selection semantics
    #
    # IMPORTANT:
    #
    # keep_high / keep_low are EVENT-level selections.
    #
    # Therefore:
    #
    #   reco_pho_pt_after_dr_high
    #
    # means all reco photons in events passing keep_high.
    #
    # This preserves the behavior of your original code.
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Preserve event alignment.
    #
    # Events passing keep_high/keep_low keep their reco photons.
    # Events failing the selection get an empty photon list.
    # ------------------------------------------------------------

    empty_reco_photons = ak.Array(
        [[] for _ in range(len(reco_pho_pt))]
    )

    reco_pho_pt_after_dr_high = ak.where(
        keep_high,
        reco_pho_pt,
        empty_reco_photons
    )

    reco_pho_pt_after_dr_low = ak.where(
        keep_low,
        reco_pho_pt,
        empty_reco_photons
    )

    # ============================================================
    # Reco photon counts
    # ============================================================

    reco_pho_counts_after_dr_high = ak.num(
        reco_pho_pt_after_dr_high,
        axis=1
    )

    reco_pho_counts_after_dr_low = ak.num(
        reco_pho_pt_after_dr_low,
        axis=1
    )

    # ============================================================
    # Reco photon weights
    # ============================================================

    # ------------------------------------------------------------
    # Reco photon weights
    # ------------------------------------------------------------

    reco_pho_counts_np = ak.to_numpy(
        reco_pho_counts
    )

    reco_pho_weight = np.divide(
        weights,
        reco_pho_counts_np,
        out=np.zeros_like(
            weights,
            dtype=float
        ),
        where=reco_pho_counts_np > 0
    )

    # One weight per reco photon
    reco_pho_pt_weight = ak.broadcast_arrays(
        reco_pho_pt,
        reco_pho_weight
    )[1]

    # For high/low selections, keep the same event-level
    # selection as the reco photon pT arrays.

    reco_pho_weights_high = ak.broadcast_arrays(
        reco_pho_pt_after_dr_high,
        reco_pho_weight
    )[1]

    reco_pho_weights_low = ak.broadcast_arrays(
        reco_pho_pt_after_dr_low,
        reco_pho_weight
    )[1]

    # ============================================================
    # Return
    #
    # Every field is now event-aligned.
    #
    # Scalar event quantities:
    #     length = number of events
    #
    # Photon quantities:
    #     length = number of events
    #     each event contains a variable-length list
    # ============================================================

    return AnalysisResult_skim(

        # --------------------------------------------------------
        # Gen photons
        # --------------------------------------------------------

        min_dr=min_dr_tmp,

        min_dr_weight=min_dr_weight,

        n_photons=np.asarray(
            n_pho
        ),

        n_photons_weight=weights,

        n_photons_high=np.asarray(
            n_pho_after_dr_high
        ),

        n_photons_high_weight=weights,

        n_photons_low=np.asarray(
            n_pho_after_dr_low
        ),

        n_photons_low_weight=weights,

        gen_photon_pt=gen_pho_pt,

        gen_photon_pt_weight=gen_photon_pt_weight,

        gen_photon_pt_after_dr_high=
            gen_pho_pt_after_dr_high,

        gen_photon_pt_after_dr_high_weight=
            gen_photon_pt_after_dr_high_weight,

        gen_photon_pt_after_dr_low=
            gen_pho_pt_after_dr_low,

        gen_photon_pt_after_dr_low_weight=
            gen_photon_pt_after_dr_low_weight,

        # --------------------------------------------------------
        # Event selections
        # --------------------------------------------------------

        keep_high=np.asarray(
            keep_high
        ),

        keep_low=np.asarray(
            keep_low
        ),

        # --------------------------------------------------------
        # Reco photons
        # --------------------------------------------------------

        n_reco_photons=np.asarray(
            reco_pho_counts
        ),

        n_reco_photons_weight=weights,

        n_reco_photons_high=np.asarray(
            reco_pho_counts_after_dr_high
        ),

        n_reco_photons_high_weight=weights,

        n_reco_photons_low=np.asarray(
            reco_pho_counts_after_dr_low
        ),

        n_reco_photons_low_weight=weights,

        reco_photon_pt=reco_pho_pt,

        reco_photon_pt_weight=reco_pho_pt_weight,

        reco_photon_pt_after_dr_high=
            reco_pho_pt_after_dr_high,

        reco_photon_pt_after_dr_high_weight=
            reco_pho_weights_high,

        reco_photon_pt_after_dr_low=
            reco_pho_pt_after_dr_low,

        reco_photon_pt_after_dr_low_weight=
            reco_pho_weights_low
    )

def run_process(filename,xsec_pb,lumi_fb,photon_pt_min,photon_eta_max,other_pt_min,parentage_func):
    events, weights = load_process(filename, xsec_pb, lumi_fb)
    return analyze_events(events,weights,photon_pt_min,photon_eta_max,other_pt_min,parentage_func)

def run_process_group(
    files,
    xsecs,
    background_dir,
    lumi_fb,
    photon_pt_min,
    photon_eta_max,
    other_pt_min,
    parentage_func
):
    if isinstance(files, str):
        files = [files]

    if isinstance(xsecs, (int, float)):
        xsecs = [xsecs]

    results = []

    for filename, xsec_pb in zip(files, xsecs):
        print(f"  File: {filename}")
        print(f"  Xsec: {xsec_pb} pb")

        results.append(
            run_process(
                filename=os.path.join(background_dir, filename),
                xsec_pb=xsec_pb,
                lumi_fb=lumi_fb,
                photon_pt_min=photon_pt_min,
                photon_eta_max=photon_eta_max,
                other_pt_min=other_pt_min,
                parentage_func=parentage_func
            )
        )

    return AnalysisResult.merge(results)

def run_process_skim(filename, xsec_pb, lumi_fb):
    table = pq.read_table(filename)
    metadata = table.schema.metadata
    if metadata is None or b"sum_genw_presel" not in metadata:
        raise RuntimeError(
            f"Missing sum_genw_presel metadata in {filename}"
        )
    sum_genw_presel = float(
        metadata[b"sum_genw_presel"].decode("utf-8")
    )
    data = {}
    for field in AnalysisResult_skim.__dataclass_fields__:
        if field not in table.column_names:
            raise RuntimeError(
                f"Missing column '{field}' in {filename}"
            )
        column = table[field]
        # Preserve jagged columns as Awkward
        if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            data[field] = ak.from_arrow(column)
        else:
            data[field] = column.to_numpy()

    result = AnalysisResult_skim(**data)
    normalization = (
        xsec_pb
        * lumi_fb
        * 1000.0
        / sum_genw_presel
    )
    for field in AnalysisResult_skim.__dataclass_fields__:
        if field.endswith("_weight"):
            value = getattr(result, field)
            setattr(result, field, value * normalization)
    return result


def run_process_group_skim(
    files,
    xsecs,
    background_dir,
    lumi_fb,
):
    if isinstance(files, str):
        files = [files]
    if isinstance(xsecs, (int, float)):
        xsecs = [xsecs]
    results = []
    for filename, xsec_pb in zip(files, xsecs):
        full_path = os.path.join(
            background_dir,
            filename
        )

        print(f"  File: {full_path}")
        print(f"  Xsec: {xsec_pb} pb")

        results.append(
            run_process_skim(
                filename=full_path,
                xsec_pb=xsec_pb,
                lumi_fb=lumi_fb,
            )
        )

    return AnalysisResult_skim.merge(results)