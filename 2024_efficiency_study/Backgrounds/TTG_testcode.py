from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.jetID import add_jetId
from coffea.analysis_tools import Weights

def is_allowed_parent(pdgId):
    apdg = abs(pdgId)

    # Quarks: d, u, s, c, b, t
    is_quark = (apdg >= 1) & (apdg <= 6)

    # Gluon
    is_gluon = apdg == 21

    return is_quark | is_gluon 

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

    return min_dr, is_isolated, is_nonisolated


# file = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/TTto2L2Nu_24SummerRun3_selected/part_00000.parquet'
file = '/eos/user/b/bbapi/My_Analysis/2024_efficiency_study/Backgrounds/TTG1Jets_24SummerRun3_selected/part_00000.parquet'
# file = 'root://xrootd-cms.infn.it///store/mc/RunIII2024Summer24NanoAODv15/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/2810000/f60b4a6c-2801-43b0-b542-6d933a71a396.root'

if ".root" in file:
    factory = NanoEventsFactory.from_root(f"{file}:Events", schemaclass=NanoAODSchema)
    events = factory.events()
else:
    events = ak.from_parquet(file)

print("I am here 1")

events["Photon"] = add_photon_SC_eta(events.Photon, events.PV)
events["Electron"] = ak.with_field(events.Electron, events.Electron.eta + events.Electron.deltaEtaSC, "ScEta")
Jets = events.Jet
Jet_id = add_jetId(Jets, 15, "2024", flattenUnflatten=True)
events["Jet"] = ak.with_field(Jets, Jet_id, "jetId")

print("I am here 2")

corrections_list = ["Smearing", "Material", "FNUF", "Electron_Smearing_EGM", "jec_jet_syst", "MuonScaRe"]

for correction in corrections_list:

    varying_function = available_object_corrections[correction]
    print(f"Applying {correction}")
    events = varying_function(
        events=events, year="2024"
    )

print("I am here 3")

event_weights = Weights(size=len(events))

electrons = events.Electron
muons = events.Muon
jets = events.Jet

weight_corrections_list = ["TriggerSF_singleLep"]

# corrections to event weights:
# for correction_name in weight_corrections_list:
#     varying_function = available_weight_corrections[correction_name]
#     # event_weights = varying_function(
#     #     events=events[selection_mask],
#     #     photons=events[f"diphotons_{do_variation}"][
#     #         selection_mask
#     #     ],
#     #     muons=muons[selection_mask],
#     #     electrons=electrons[selection_mask],
#     #     jets=jets[selection_mask],
#     #     weights=event_weights,
#     #     dataset_name=dataset_name,
#     #     year=self.year[dataset_name][0],
#     # )

#     event_weights = varying_function(
#         events=events,
#         muons=muons,
#         electrons=electrons,
#         jets=jets,
#         weights=event_weights,
#         # dataset_name=dataset_name,
#         year="2024",
#     )

print(events.Photon.fields)

print("I am here 4")

gen = events.GenPart

gen_pho = gen[gen.pdgId == 22]

gen_pho_mother = gen[gen_pho.genPartIdxMother].pdgId

allowed_pdg = [1, 2 ,3 , 4, 5, 6, 21]

allowed_mask = ak.any([abs(gen_pho_mother) == pdg for pdg in allowed_pdg], axis=0)

event_mask = ak.any(allowed_mask, axis=1)

mdr, is_iso, is_noniso = photon_isolation_ok(events)

print("event_mask: ", event_mask)
print("mdr: ", mdr)
print("is_iso: ", is_iso)
print("is_noniso: ", is_noniso)


