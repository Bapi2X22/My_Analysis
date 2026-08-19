import awkward as ak

# def event_mask(collections):
#     """
#     collections is a dictionary containing the selected objects.
#     """

#     jets = collections["Jet"]
#     electrons = collections["Electron"]
#     muons = collections["Muon"]
#     photons = collections["Photon"]

#     nJet = ak.num(jets)
#     nEle = ak.num(electrons)
#     nMuon = ak.num(muons)
#     nPho = ak.num(photons)

#     mask = (
#         (nJet >= 1)
#         &
#         (nPho >= 2)
#         &
#         ((nEle + nMuon) >= 1)
#     )

#     return mask

def event_mask(collections):

    jets = collections["Jet"]
    electrons = collections["Electron"]
    muons = collections["Muon"]
    photons = collections["Photon"]

    nJet = ak.num(jets)
    nEle = ak.num(electrons)
    nMuon = ak.num(muons)
    nPho = ak.num(photons)

    lepton_cut = (nEle + nMuon) >= 1
    photon_cut = nPho >= 2
    jet_cut = nJet >= 1

    mask_lepton = lepton_cut
    mask_lepton_photon = lepton_cut & photon_cut
    mask_final = mask_lepton_photon & jet_cut

    cut_masks = {
        "lepton": mask_lepton,
        "lepton_photon": mask_lepton_photon,
        "final": mask_final,
    }

    return mask_final, cut_masks