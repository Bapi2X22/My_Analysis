def photon_mask(photons, apply_pixelSeed=True):
    mask = ((photons.pt > 12) & (photons.isScEtaEB | photons.isScEtaEE))
    if apply_pixelSeed:
        mask = mask & (~photons.pixelSeed)
    return mask