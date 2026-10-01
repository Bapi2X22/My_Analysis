import awkward as ak
import numpy as np
from config import DR_CUT, OTHER_PT_MIN, PHOTON_ETA_MAX, PHOTON_PT_MIN

def is_allowed_parent(pdgId):
    apdg = abs(pdgId)

    # Quarks: d, u, s, c, b, t
    is_quark = (apdg >= 1) & (apdg <= 6)

    # Gluon
    is_gluon = apdg == 21

    # Leptons: e, nu_e, mu, nu_mu, tau, nu_tau
    is_lepton = (
        (apdg == 11) |
        (apdg == 13) |
        (apdg == 15))

    # Bosons: gamma, Z, W, Higgs
    is_boson = (
        (apdg == 22) |
        (apdg == 23) |
        (apdg == 24) |
        (apdg == 25)
    )

    return is_quark | is_gluon | is_lepton | is_boson

def is_allowed_parent_TT(pdgId):
    apdg = abs(pdgId)
    is_quark = (apdg >= 1) & (apdg <= 6)
    is_gluon = apdg == 21
    return is_quark | is_gluon 

def photon_parentage_ok(gen_pho, gen, parentage_func=is_allowed_parent, max_depth=30):

    mother_idx = gen_pho.genPartIdxMother

    # Start with all photons considered valid
    history_ok = ak.ones_like(mother_idx, dtype=bool)

    # Photons with a valid mother need to be followed
    active = mother_idx >= 0

    current_idx = mother_idx

    for _ in range(max_depth):

        if not ak.any(active):
            break

        # Prevent -1 from being used as an index
        safe_idx = ak.where(active, current_idx, 0)

        ancestor = gen[safe_idx]

        ancestor_pdgId = ancestor.pdgId

        # Is this ancestor an allowed particle?
        allowed = parentage_func(ancestor_pdgId)

        # If an active ancestor is NOT allowed,
        # reject this photon.
        history_ok = history_ok & ak.where(active, allowed, True)

        # Move to the next mother only if current ancestor
        # was allowed and has a valid mother.
        next_idx = ancestor.genPartIdxMother

        active = (active & allowed & (next_idx >= 0))

        current_idx = next_idx

    return history_ok

def photon_min_dr(events, photon_pt_min=PHOTON_PT_MIN, photon_eta_max=PHOTON_ETA_MAX, other_pt_min=OTHER_PT_MIN, parentage_func=is_allowed_parent):
    gen = events.GenPart
    apdg = abs(gen.pdgId)

    is_photon = apdg == 22
    photon_kin = (gen.pt > photon_pt_min) & (abs(gen.eta) < photon_eta_max)

    gen_idx = ak.local_index(gen, axis=1)

    photon_daughter_matrix = gen.genPartIdxMother[:, :, None] == gen_idx[:, None, :]

    daughter_is_photon = apdg[:, None, :] == 22

    has_photon_daughter = ak.any(photon_daughter_matrix & daughter_is_photon, axis=2)

    is_last_photon = is_photon & ~has_photon_daughter

    gen_pho = gen[is_last_photon & photon_kin]

    pho_history = photon_parentage_ok(gen_pho, gen, parentage_func=parentage_func)
    gen_pho = gen_pho[pho_history]

    gen_pho_pt = gen_pho.pt

    n_pho = ak.num(gen_pho)

    is_quark = (apdg >= 1) & (apdg <= 6)

    is_gluon = apdg == 21

    is_lepton = ((apdg == 11) | (apdg == 13) | (apdg == 15))

    is_boson = ((apdg == 23) | (apdg == 24) | (apdg == 25))

    is_other = is_quark | is_gluon | is_lepton | is_boson

    other_kin = gen.pt > other_pt_min
    other = gen[is_other & other_kin]

    pairs = ak.cartesian({"pho": gen_pho, "part": other}, axis=1, nested=True)

    pho = pairs["pho"]
    part = pairs["part"]

    deta = pho.eta - part.eta

    dphi = np.arctan2(np.sin(pho.phi - part.phi), np.cos(pho.phi - part.phi))

    dr = np.sqrt(deta**2 + dphi**2)

    min_dr = ak.min(dr, axis=2)

    return min_dr, gen_pho_pt


def fake_only_photons_event_count(events, photon_pt_min=10.0, photon_eta_max=3.0, other_pt_min=5.0):
    gen = events.GenPart
    apdg = abs(gen.pdgId)

    is_photon = apdg == 22
    photon_kin = (gen.pt > photon_pt_min) & (abs(gen.eta) < photon_eta_max)

    gen_idx = ak.local_index(gen, axis=1)

    photon_daughter_matrix = gen.genPartIdxMother[:, :, None] == gen_idx[:, None, :]

    daughter_is_photon = apdg[:, None, :] == 22

    has_photon_daughter = ak.any(photon_daughter_matrix & daughter_is_photon, axis=2)

    is_last_photon = is_photon & ~has_photon_daughter

    gen_pho = gen[is_last_photon & photon_kin]

    pho_history = photon_parentage_ok(gen_pho, gen)
    gen_pho = gen_pho[pho_history]

    n_pho = ak.num(gen_pho)

    at_least_one_real_pho = n_pho > 0

    n_remaining_events = len(events[~at_least_one_real_pho])

    return n_remaining_events