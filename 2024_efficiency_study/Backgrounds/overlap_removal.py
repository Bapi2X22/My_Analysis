def photon_from_DYG(photons, genpart):

    order = ak.argsort(photons.pt, axis=1, ascending=False)
    photons = photons[order]
    photons = photons[:, :2]

    idx = photons.genPartIdx
    valid = idx >= 0

    photon = ak.mask(genpart[idx], valid)

    mother_idx = photon.genPartIdxMother
    mother = ak.mask(genpart[mother_idx], mother_idx >= 0)

    photon_pdgId = photon.pdgId
    mother_pdgId = mother.pdgId

    from_lepton = (abs(photon_pdgId) == 22) & ((abs(mother_pdgId) == 11) | (abs(mother_pdgId) == 13))
    from_Z = abs(mother_pdgId) == 23
    from_gamma = abs(mother_pdgId) == 22

    is_DYG_photon = from_lepton | from_Z | from_gamma

    event_is_DYGto2LG = ak.any(is_DYG_photon, axis=1)

    return event_is_DYGto2LG