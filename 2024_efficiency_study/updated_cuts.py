import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector


def delta_r_mask(
    first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float
) -> ak.highlevel.Array:
    """
    Select objects from first which are at least threshold away from all objects in second.
    The result is a mask (i.e., a boolean array) of the same shape as first.

    :param first: objects which are required to be at least threshold away from all objects in second
    :type first: coffea.nanoevents.methods.candidate.PtEtaPhiMCandidate
    :param second: objects which are all objects in first must be at leats threshold away from
    :type second: coffea.nanoevents.methods.candidate.PtEtaPhiMCandidate
    :param threshold: minimum delta R between objects
    :type threshold: float
    :return: boolean array of objects in objects1 which pass delta_R requirement
    :rtype: coffea.nanoevents.methods.candidate.PtEtaPhiMCandidate
    """

    mval = first.metric_table(second)
    return ak.all(mval > threshold, axis=-1)


def build_diphoton_candidates(photons, min_pt_lead_photon):
    """
    Build diphoton candidates from a collection of photons.

    Parameters:
        photons (awkward.Array): The input photon collection.
        min_pt_lead_photon (float): The minimum pT required for the leading photon.

    Returns:
        awkward.Array: The diphoton candidate collection with calculated kinematic properties.
    """
    # Sort photons in descending order of pT
    sorted_photons = photons[ak.argsort(photons.pt, ascending=False)]
    # Ensure a 'charge' field exists; default to zero if not provided
    sorted_photons["charge"] = ak.zeros_like(sorted_photons.pt)

    # Create all possible pairs of photons (combinations) with fields "pho_lead" and "pho_sublead"
    diphotons = ak.combinations(sorted_photons, 2, fields=["pho_lead", "pho_sublead"])

    # Apply the cut on the leading photon's pT
    diphotons = diphotons[diphotons["pho_lead"].pt > min_pt_lead_photon]

    # Combine four-momenta of the two photons
    diphoton_4mom = diphotons["pho_lead"] + diphotons["pho_sublead"]
    diphotons["pt"] = diphoton_4mom.pt
    diphotons["eta"] = diphoton_4mom.eta
    diphotons["phi"] = diphoton_4mom.phi
    diphotons["mass"] = diphoton_4mom.mass
    diphotons["charge"] = diphoton_4mom.charge

    # Calculate rapidity
    diphoton_pz = diphoton_4mom.z
    diphoton_e = diphoton_4mom.energy
    diphotons["rapidity"] = 0.5 * numpy.log((diphoton_e + diphoton_pz) / (diphoton_e - diphoton_pz))

    # Sort diphoton candidates by pT in descending order
    diphotons = diphotons[ak.argsort(diphotons.pt, ascending=False)]

    return diphotons

photons = events.Photon
electrons = events.Electron
muons = events.Muon
jets = events.Jet

def select_electrons(
    electrons: ak.highlevel.Array
    # diphotons
) -> ak.highlevel.Array:
    pt_cut = electrons.pt > 30

    eta_cut = abs(electrons.eta) < 2.5

    id_cut = electrons.mvaIso_WP80

    return pt_cut & eta_cut & id_cut 


def select_muons(
    muons: ak.highlevel.Array
    # diphotons: ak.highlevel.Array
) -> ak.highlevel.Array:
    pt_cut = muons.pt > 24

    eta_cut = abs(muons.eta) < 2.4

    id_cut = muons.mediumId

    iso_cut = muons.pfIsoId >= 3

    global_cut = muons.isGlobal

    return pt_cut & eta_cut & id_cut & iso_cut & global_cut

diphotons = build_diphoton_candidates(photons, min_pt_lead_photon=15)

def select_jets(
    jets: ak.highlevel.Array,
    diphotons, 
    electrons,
    muons
) -> ak.highlevel.Array:

    jetId_cut = jets.jetId >= 2

    pt_cut = jets.pt > 20
    eta_cut = abs(jets.eta) < 2.4

    dr_cut_lead_pho_jet = delta_r_mask(jets, diphotons.pho_lead, 0.4)
    dr_cut_sublead_pho_jet = delta_r_mask(jets, diphotons.pho_sublead, 0.4)
    dr_cut_ele = delta_r_mask(jets, electrons, 0.4)
    dr_cut_mu = delta_r_mask(jets, muons, 0.4)

    return (
        (jetId_cut)
        & (pt_cut)
        & (eta_cut)
        & (dr_cut_ele)
        & (dr_cut_mu)
        & (dr_cut_lead_pho_jet)
        & (dr_cut_sublead_pho_jet)
    )


def photon_preselection(
    photons: ak.Array,
    electrons, 
    muons
) -> ak.Array:
    """
    Apply preselection cuts to photons.
    Note that these selections are applied on each photon, it is not based on the diphoton pair.
    """
    abs_eta = np.abs(photons.eta)
    valid_eta = (abs_eta <= 2.5) & ~((abs_eta >= 1.442) & (abs_eta <= 1.566))

    dr_cut = delta_r_mask(photons, electrons, 0.2)
    dr_cut_muon = delta_r_mask(photons, muons, 0.2)

        # quadratic EA corrections in Run3 : https://indico.cern.ch/event/1204277/contributions/5064356/attachments/2538496/4369369/CutBasedPhotonID_20221031.pdf
    return photons[
        (photons.pt > 15)
        & valid_eta
        & dr_cut
        & dr_cut_muon
        &(ak.where(photons.isScEtaEB, photons.mvaID > 0.0439603, photons.mvaID > -0.249526))
    ]

    # Select b-Jets

    bJets = jets[jets.btagUParTAK4B > 0.0246]
