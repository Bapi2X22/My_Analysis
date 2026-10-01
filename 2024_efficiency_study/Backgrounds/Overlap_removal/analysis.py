from dataclasses import dataclass
import numpy as np
import os
import awkward as ak

from data_loader import load_process, load_root_process
from physics import photon_min_dr
from config import DR_CUT 

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

def analyze_events(events,weights,photon_pt_min,photon_eta_max,other_pt_min,parentage_func):

    min_dr, gen_pho_pt = photon_min_dr(events, photon_pt_min=photon_pt_min, photon_eta_max=photon_eta_max, other_pt_min=other_pt_min, parentage_func=parentage_func)

    min_dr_tmp = ak.fill_none(min_dr, np.inf)

    pass_dr_high = min_dr_tmp > DR_CUT
    pass_dr_low = min_dr_tmp < DR_CUT

    n_pho = ak.num(gen_pho_pt, axis=1)

    n_pho_after_dr_high = ak.sum(pass_dr_high, axis=1)
    n_pho_after_dr_low = ak.sum(pass_dr_low, axis=1)

    keep_high = (n_pho == 0) | (n_pho_after_dr_high > 0)
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
        gen_photon_pt_after_dr_low_weight=gen_photon_pt_after_dr_low_weight
    )

def run_process(filename,xsec_pb,lumi_fb,photon_pt_min,photon_eta_max,other_pt_min,parentage_func):
    events, weights = load_process(filename, xsec_pb, lumi_fb)
    return analyze_events(events,weights,photon_pt_min,photon_eta_max,other_pt_min,parentage_func)

# def run_process(
#     filename,
#     xsec_pb,
#     lumi_fb,
#     photon_pt_min,
#     photon_eta_max,
#     other_pt_min,
#     parentage_func
# ):
#     events, weights = load_process(filename, xsec_pb, lumi_fb)

#     min_dr, gen_pho_pt = photon_min_dr(events, photon_pt_min=photon_pt_min, photon_eta_max=photon_eta_max, other_pt_min=other_pt_min, parentage_func=parentage_func)

#     min_dr_tmp = ak.fill_none(min_dr, np.inf)

#     pass_dr_high = min_dr_tmp > DR_CUT
#     pass_dr_low = min_dr_tmp < DR_CUT

#     n_pho = ak.num(gen_pho_pt, axis=1)

#     n_pho_after_dr_high = ak.sum(pass_dr_high, axis=1)
#     n_pho_after_dr_low = ak.sum(pass_dr_low, axis=1)

#     keep_high = (n_pho == 0) | (n_pho_after_dr_high > 0)
#     keep_low = (n_pho == 0) | (n_pho_after_dr_low > 0)

#     gen_pho_pt_after_dr_high = gen_pho_pt[pass_dr_high]
#     gen_pho_pt_after_dr_low = gen_pho_pt[pass_dr_low]

#     n_pho_high = n_pho_after_dr_high[keep_high]
#     n_pho_low = n_pho_after_dr_low[keep_low]

#     weights = np.asarray(weights)

#     photon_counts = ak.to_numpy(ak.num(gen_pho_pt, axis=1))
#     photon_counts_after_dr_high = ak.to_numpy(ak.num(gen_pho_pt_after_dr_high, axis=1))
#     photon_counts_after_dr_low = ak.to_numpy(ak.num(gen_pho_pt_after_dr_low, axis=1))
#     photon_weight = np.divide(weights, photon_counts, out=np.zeros_like(weights, dtype=float), where=photon_counts > 0)

#     min_dr_flat = ak.to_numpy(ak.flatten(min_dr_tmp))
#     gen_photon_pt_flat = ak.to_numpy(ak.flatten(gen_pho_pt))
#     gen_photon_pt_after_dr_high_flat = ak.to_numpy(ak.flatten(gen_pho_pt_after_dr_high))
#     gen_photon_pt_after_dr_low_flat = ak.to_numpy(ak.flatten(gen_pho_pt_after_dr_low))
#     min_dr_weight = np.repeat(photon_weight, photon_counts)
#     gen_photon_pt_weight = np.repeat(photon_weight, photon_counts)
#     gen_photon_pt_after_dr_high_weight = np.repeat(photon_weight,photon_counts_after_dr_high)
#     gen_photon_pt_after_dr_low_weight = np.repeat(photon_weight,photon_counts_after_dr_low)

#     return AnalysisResult(
#         min_dr=min_dr_flat,
#         min_dr_weight=min_dr_weight,
#         n_photons=np.asarray(n_pho),
#         n_photons_weight=weights,
#         n_photons_high=np.asarray(n_pho_high),
#         n_photons_high_weight=weights[ak.to_numpy(keep_high)],
#         n_photons_low=np.asarray(n_pho_low),
#         n_photons_low_weight=weights[ak.to_numpy(keep_low)],
#         gen_photon_pt=gen_photon_pt_flat,
#         gen_photon_pt_weight=gen_photon_pt_weight,
#         gen_photon_pt_after_dr_high=gen_photon_pt_after_dr_high_flat,
#         gen_photon_pt_after_dr_high_weight=gen_photon_pt_after_dr_high_weight,
#         gen_photon_pt_after_dr_low=gen_photon_pt_after_dr_low_flat,
#         gen_photon_pt_after_dr_low_weight=gen_photon_pt_after_dr_low_weight
#     )

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

def run_root_process_group(directories,n_files,xsecs,background_dir,lumi_fb,photon_pt_min,photon_eta_max,other_pt_min,parentage_func):
    if isinstance(directories, str):
        directories = [directories]
    if isinstance(n_files, int):
        n_files = [n_files]
    if isinstance(xsecs, (int, float)):
        xsecs = [xsecs]
    results = []
    for directory, n_file, xsec_pb in zip(directories, n_files, xsecs):
        print(f"  Directory: {directory}")
        print(f"  Xsec: {xsec_pb} pb")
        root_results = load_root_process(directories=[directory],n_files=[n_file],xsecs=[xsec_pb],background_dir=background_dir,lumi_fb=lumi_fb)
        for events, weights in root_results:
            results.append(analyze_events(events,weights,photon_pt_min,photon_eta_max,other_pt_min,parentage_func))
    return AnalysisResult.merge(results)