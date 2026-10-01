import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import pandas as pd
from particle import particle


def particle_name(pdgid):
    try:
        return Particle.from_pdgid(int(pdgid)).name
    except Exception:
        return str(int(pdgid))

def print_genpart_tree(genPart, event_idx):
    particles = genPart[event_idx]
    daughters = {}
    for i in range(len(particles)):
        mother = int(particles[i].genPartIdxMother)
        if mother >= 0 and mother != i:
            daughters.setdefault(mother, []).append(i)

    def print_node(idx, prefix="", is_last=True):
        p = particles[idx]
        name = particle_name(p.pdgId)

        pt = float(p["pt"])
        eta = float(p["eta"])
        phi = float(p["phi"])
        mass = float(p["mass"])

        connector = "`-- " if is_last else "|-- "

        print(f"{prefix}{connector}{name}  pt={pt:.2f}  eta={eta:.2f}  phi={phi:.2f}  mass={mass:.2f}")

        children = daughters.get(idx, [])

        for j, child in enumerate(children):
            child_prefix = prefix + ("    " if is_last else "|   ")
            print_node(child, child_prefix, j == len(children) - 1)

    print("\n" + "=" * 100)
    print(f"EVENT {event_idx}")
    print("=" * 100)

    roots = [i for i in range(len(particles)) if int(particles[i].genPartIdxMother) < 0]
    for j, root in enumerate(roots):
        print_node(root, "", j == len(roots) - 1)

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


def print_debug_table(
    mother_idx,
    active,
    safe_idx,
    ancestor_pdgId,
    allowed,
    next_idx,
    history_ok,
):
    event_idx = ak.broadcast_arrays(
        ak.local_index(mother_idx, axis=0),
        mother_idx
    )[0]

    photon_idx = ak.local_index(mother_idx, axis=1)

    df = pd.DataFrame({
        "event": ak.to_numpy(ak.flatten(event_idx)),
        "photon": ak.to_numpy(ak.flatten(photon_idx)),
        "mother": ak.to_numpy(ak.flatten(mother_idx)),
        "active": ak.to_numpy(ak.flatten(active)),
        "safe_idx": ak.to_numpy(ak.flatten(safe_idx)),
        "ancestor_pdgId": ak.to_numpy(ak.flatten(ancestor_pdgId)),
        "allowed": ak.to_numpy(ak.flatten(allowed)),
        "next_idx": ak.to_numpy(ak.flatten(next_idx)),
        "history_ok": ak.to_numpy(ak.flatten(history_ok)),
    })

    print(df.to_string(index=False))

def photon_parentage_ok(gen_pho, gen, max_depth=30):

    mother_idx = gen_pho.genPartIdxMother

    print("mother_idx: ", mother_idx)

    # Start with all photons considered valid
    history_ok = ak.ones_like(mother_idx, dtype=bool)

    print("history ok: ", history_ok)

    # Photons with a valid mother need to be followed
    active = mother_idx >= 0

    print("active: ", active)

    current_idx = mother_idx

    for _ in range(max_depth):

        if not ak.any(active):
            break

        # Prevent -1 from being used as an index
        safe_idx = ak.where(active, current_idx, 0)

        print("safe_idx: ", safe_idx)

        ancestor = gen[safe_idx]

        print("ancestor: ", ancestor)

        ancestor_pdgId = ancestor.pdgId

        print("ancestor pdgId: ", ancestor_pdgId)

        # Is this ancestor an allowed particle?
        allowed = is_allowed_parent(ancestor_pdgId)

        print("allowed: ", allowed)

        # If an active ancestor is NOT allowed,
        # reject this photon.
        history_ok = history_ok & ak.where(active, allowed, True)

        print("history_ok: ", history_ok)

        # Move to the next mother only if current ancestor
        # was allowed and has a valid mother.
        next_idx = ancestor.genPartIdxMother

        print("next_idx: ", next_idx)

        print(f"\n{'='*100}")
        print(f"DEPTH = {max_depth}")
        print(f"{'='*100}")

        # print_debug_table(mother_idx, active, safe_idx, ancestor_pdgId, allowed, next_idx, history_ok)

        active = (active & allowed & (next_idx >= 0))

        print("new_active: ", active)

        current_idx = next_idx

        print("new current_idx: ", current_idx)

    return history_ok


def photon_isolation_ok(events, photon_pt_min=10.0, photon_eta_max = 3.0,  dr_min=0.2, other_pt_min = 5.0):

    gen = events.GenPart
    apdg = abs(gen.pdgId)

    is_photon = apdg == 22
    photon_kin = (gen.pt > photon_pt_min) & (abs(gen.eta) < photon_eta_max)

    gen_idx = ak.local_index(gen, axis=1)

    print("gen_idx: ", gen_idx)

    print("gen_idx[:, None, :]: ", gen_idx[:, None, :])

    print("gen.genPartIdxMother[:, :, None]: ", gen.genPartIdxMother[:, :, None])

    photon_daughter_matrix = gen.genPartIdxMother[:, :, None] == gen_idx[:, None, :]

    print("photon_daughter_matrix: ", photon_daughter_matrix)

    print("where true: ", ak.any(photon_daughter_matrix, axis=2))

    daughter_is_photon = apdg[:, None, :] == 22

    print("daughter is photon: ", daughter_is_photon)

    has_photon_daughter = ak.any(photon_daughter_matrix & daughter_is_photon, axis=2)

    print("has_photon_daughter: ", has_photon_daughter)

    print("Any photon daughter: ", ak.any(has_photon_daughter == True, axis =1))

    is_last_photon = is_photon & ~has_photon_daughter

    print("Any last photon: ", ak.any(is_last_photon, axis =1))

    print("is_last_photon: ", is_last_photon)

    gen_pho = gen[is_last_photon & photon_kin]

    pho_history = photon_parentage_ok(gen_pho, gen)
    gen_pho = gen_pho[pho_history]

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

    print("min_dr: ", min_dr)

    is_isolated, is_nonisolated = ak.any(min_dr > dr_min, axis=1), ak.all(min_dr < dr_min, axis=1)

    idx_iso = ak.where(is_isolated)[0]
    print("idx_iso: ", idx_iso)
    print("min_dr_true: ", min_dr[idx_iso])

    idx_non_iso = ak.where(is_nonisolated)[0]
    print("idx_non_iso: ", idx_non_iso)
    print("min_dr_true_non_iso: ", min_dr[idx_non_iso])

    print("isolated, nonisolated: ", is_isolated, is_nonisolated)

    print("min_dr: ", min_dr)

    return is_isolated, is_nonisolated


# file = 'root://xrootd-cms.infn.it//store/mc/RunIII2024Summer24NanoAODv15/DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v6/2520000/3a3ff919-a8b6-4054-9518-2b203bb702b1.root'
# file = 'root://xrootd-cms.infn.it///store/mc/RunIII2024Summer24NanoAODv15/DYGto2LG-1Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/2820000/291f3f90-494d-4c77-8484-2837a28372c5.root'
# file = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/Skimmer/TTG1Jets_24SummerRun3/6a92fd94-7618-4f42-a02c-6b2ebdaec35c_skim.root'
# file = 'root://xrootd-cms.infn.it///store/mc/RunIII2024Summer24NanoAODv15/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/2810000/f60b4a6c-2801-43b0-b542-6d933a71a396.root'
file = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/TTto2L2Nu_24SummerRun3_selected/part_00000.parquet'


if ".root" in file:
    factory = NanoEventsFactory.from_root(f"{file}:Events", schemaclass=NanoAODSchema)
    events = factory.events()
else:
    events = ak.from_parquet(file)

# factory = NanoEventsFactory.from_root(
#     f"{file}:Events",
#     schemaclass=NanoAODSchema,
# )
# events = factory.events()

# for i in range(0, 10):
#     print(f"\n{'-' * 80}")
#     print(f"EVENT {i}")
#     print(f"{'-' * 80}")
#     print_genpart_tree(events.GenPart, i)

is_iso, is_non_iso = photon_isolation_ok(events)

print("is_iso: ", is_iso)
print("is_non_iso: ", is_non_iso)


