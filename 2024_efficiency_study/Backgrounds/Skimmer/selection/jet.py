import awkward as ak
def jet_mask(jets, apply_kinematic_cuts_jet=False):

    if apply_kinematic_cuts_jet:
        return ((jets.pt > 15) & (abs(jets.eta) < 2.4))
    return ak.ones_like(jets.pt, dtype=bool)