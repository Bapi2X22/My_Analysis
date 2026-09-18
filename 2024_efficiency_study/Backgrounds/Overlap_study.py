import awkward as ak
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import matplotlib.colors as colors
import correctionlib
import matplotlib.pyplot as plt
from particle import Particle
import os
import glob
import argparse



parser = argparse.ArgumentParser(
    description="Plot mother PDG IDs of lead and sublead photons"
)

parser.add_argument(
    "--input-dir",
    required=True,
    help="Directory containing ROOT files"
)

parser.add_argument(
    "--max-files",
    type=int,
    default=-1,
    help="Maximum number of ROOT files to process. -1 = all files"
)

parser.add_argument(
    "--label",
    default="",
    help="Label to add to the plots"
)

parser.add_argument(
    "--plot-name",
    default="photon_mother_pdgId.png",
    help="Output plot name"
)

args = parser.parse_args()

def add_jetId(jets, flattenUnflatten=False):
    """
    Add (or recompute) jet ID to the jets object based on the NanoAOD version.
    """
    abs_eta = abs(jets.eta)

    jerc_json = "/eos/user/b/bbapi/HiggsDNA_220526/HiggsDNA/higgs_dna/systematics/JSONs/POG/JME/2024_Summer24/jetid.json.gz"

    cset = correctionlib.CorrectionSet.from_file(jerc_json)

    if flattenUnflatten:
        counts = ak.num(jets)
        jets = ak.flatten(jets, axis=1)

    eval_dict = {
        "eta": jets.eta,
        "chHEF": jets.chHEF,
        "neHEF": jets.neHEF,
        "chEmEF": jets.chEmEF,
        "neEmEF": jets.neEmEF,
        "muEF": jets.muEF,
        "chMultiplicity": jets.chMultiplicity,
        "neMultiplicity": jets.neMultiplicity,
        "multiplicity": jets.chMultiplicity + jets.neMultiplicity
    }

    ## Default tight for NanoAOD version 13 and above
    idTight = cset["AK4PUPPI_Tight"]
    inputsTight = [eval_dict[input.name] for input in idTight.inputs]
    idTight_value = idTight.evaluate(*inputsTight) * 2  # equivalent to bit2

    # Default tight lepton veto
    idTightLepVeto = cset["AK4PUPPI_TightLeptonVeto"]
    inputsTightLepVeto = [eval_dict[input.name] for input in idTightLepVeto.inputs]
    idTightLepVeto_value = idTightLepVeto.evaluate(*inputsTightLepVeto) * 4  # equivalent to bit3

    # Default jet ID
    id_value = idTight_value + idTightLepVeto_value

    if flattenUnflatten:
        return ak.unflatten(id_value, counts)
    else:
        return id_value


def delta_r_mask(
    first: ak.highlevel.Array, second: ak.highlevel.Array, threshold: float
) -> ak.highlevel.Array:

    mval = first.metric_table(second)
    return ak.all(mval > threshold, axis=-1)

def build_diphoton_candidates(photons, min_pt_lead_photon):
    # Sort photons in descending order of pT
    sorted_photons = photons[ak.argsort(photons.pt, ascending=False)]

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
    diphotons["rapidity"] = 0.5 * np.log((diphoton_e + diphoton_pz) / (diphoton_e - diphoton_pz))

    # Sort diphoton candidates by pT in descending order
    diphotons = diphotons[ak.argsort(diphotons.pt, ascending=False)]
    diphotons = ak.with_name(diphotons, "PtEtaPhiMCandidate")

    return diphotons

def photon_preselection_bbgg(
    photons: ak.Array,
    events: ak.Array,
    electrons: ak.Array,
    muons: ak.Array
) -> ak.Array:

    dr_cut = delta_r_mask(photons, electrons, 0.2)
    dr_cut_muon = delta_r_mask(photons, muons, 0.2)

    return photons[
        (~photons.pixelSeed)
        & (photons.pt > 15.0)
        & (photons.isScEtaEB | photons.isScEtaEE)
        & dr_cut
        & dr_cut_muon
        &(ak.where(photons.isScEtaEB, photons.mvaID > 0.0439603, photons.mvaID > -0.249526))
    ]


def select_electrons_bbgg(
    electrons: ak.highlevel.Array
) -> ak.highlevel.Array:
    pt_cut = electrons.pt > 30

    eta_cut = abs(electrons.eta) < 2.5

    id_cut = electrons.mvaIso_WP80

    return pt_cut & eta_cut & id_cut

def select_muons_bbgg(
    muons: ak.highlevel.Array
) -> ak.highlevel.Array:
    pt_cut = muons.pt > 26.0

    eta_cut = abs(muons.eta) < 2.4

    id_cut = muons.mediumId
        
    iso_cut = muons.pfIsoId >= 3
   
    global_cut = muons.isGlobal

    return pt_cut & eta_cut & id_cut & iso_cut & global_cut 


def select_jets_bbgg(
    jets: ak.highlevel.Array,
    diphotons: ak.highlevel.Array,
    muons: ak.highlevel.Array,
    electrons: ak.highlevel.Array,
    clean_jet_pho: bool = True,
    clean_jet_ele: bool = True,
    clean_jet_muo: bool = True
) -> ak.highlevel.Array:
    

    jetId_cut = jets.jetId >= 2

    pt_cut = jets.pt > 20
    eta_cut = abs(jets.eta) < 2.4

    if (clean_jet_pho) & (ak.num(diphotons.pt, axis=0) > 0):
        lead = ak.zip(
            {
                "pt": diphotons.pho_lead.pt,
                "eta": diphotons.pho_lead.eta,
                "phi": diphotons.pho_lead.phi,
                "mass": diphotons.pho_lead.mass,
                "charge": diphotons.pho_lead.charge,
            }
        )
        lead = ak.with_name(lead, "PtEtaPhiMCandidate")
        sublead = ak.zip(
            {
                "pt": diphotons.pho_sublead.pt,
                "eta": diphotons.pho_sublead.eta,
                "phi": diphotons.pho_sublead.phi,
                "mass": diphotons.pho_sublead.mass,
                "charge": diphotons.pho_sublead.charge,
            }
        )
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        sublead = ak.with_name(sublead, "PtEtaPhiMCandidate")
        dr_pho_lead_cut = delta_r_mask(jets, lead, 0.4)
        dr_pho_sublead_cut = delta_r_mask(jets, sublead, 0.4)
    else:
        dr_pho_lead_cut = jets.pt > -1
        dr_pho_sublead_cut = jets.pt > -1

    if (clean_jet_ele) & (ak.num(electrons.pt, axis=0) > 0):
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        dr_electrons_cut = delta_r_mask(jets, electrons, 0.4)
    else:
        dr_electrons_cut = jets.pt > -1

    if (clean_jet_muo) & (ak.num(muons.pt, axis=0) > 0):
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        dr_muons_cut = delta_r_mask(jets, muons, 0.4)
    else:
        dr_muons_cut = jets.pt > -1

    return (
        (jetId_cut)
        & (pt_cut)
        & (eta_cut)
        & (dr_pho_lead_cut)
        & (dr_pho_sublead_cut)
        & (dr_electrons_cut)
        & (dr_muons_cut)
    )

# =========================================================
# Input directory
# =========================================================

files = sorted(
    glob.glob(
        os.path.join(args.input_dir, "*.root")
    )
)

if args.max_files > 0:
    files = files[:args.max_files]

print(f"Found {len(files)} ROOT files")
print(f"Processing {len(files)} ROOT files")


# =========================================================
# Dictionaries to accumulate PDG-ID counts
# =========================================================

lead_counts = {}
sublead_counts = {}


# =========================================================
# PDG ID -> particle name
# =========================================================

def pdgid_to_name(pdgid):
    try:
        p = Particle.from_pdgid(int(pdgid))
        return p.name if p is not None else str(int(pdgid))
    except Exception:
        return str(int(pdgid))


# =========================================================
# Process one ROOT file
# =========================================================

# =========================================================
# Counters
# =========================================================

lead_counts, sublead_counts, pair_counts = {}, {}, {}
genmother_pair_counts = {}

same_genpart_count = 0
mask41_count = 0
same_genpart_mask41_count = 0
total_selected_events = 0


# =========================================================
# Process one file
# =========================================================

genmother_pair_counts = {}

def process_file(file):
    global same_genpart_count, mask41_count, same_genpart_mask41_count, total_selected_events
    try:
        print(f"Processing: {file}")
        events = NanoEventsFactory.from_root(f"{file}:Events", schemaclass=NanoAODSchema).events()

        electrons, muons, photons, Jets, genPart = events.Electron, events.Muon, events.Photon, events.Jet, events.GenPart

        Jet_id = add_jetId(Jets, flattenUnflatten=True)
        events["Jet"] = ak.with_field(Jets, Jet_id, "jetId")
        jets = events.Jet

        electrons = electrons[select_electrons_bbgg(electrons)]
        muons = muons[select_muons_bbgg(muons)]
        photons = photon_preselection_bbgg(photons, events, electrons, muons)
        diphotons = build_diphoton_candidates(photons, 15.0)
        jets = jets[select_jets_bbgg(jets, diphotons, muons, electrons)]

        one_ele, zero_ele = ak.num(electrons) == 1, ak.num(electrons) == 0
        one_mu, zero_mu = ak.num(muons) == 1, ak.num(muons) == 0
        lepton_channel_mask = (one_ele & zero_mu) | (one_mu & zero_ele)

        b_jets = jets[jets.btagUParTAK4B > 0.1272]
        event_mask = lepton_channel_mask & (ak.num(photons) >= 2) & (ak.num(b_jets) >= 1)

        sel_events = ak.sum(event_mask)
        total_selected_events += sel_events

        photons, genPart = photons[event_mask], genPart[event_mask]
        photons = photons[ak.argsort(photons.pt, ascending=False)]

        photon_genparts = genPart[ak.where(photons.genPartIdx >= 0, photons.genPartIdx, 0)]
        valid_photon = photons.genPartIdx >= 0
        mother_idx = ak.where(valid_photon, photon_genparts.genPartIdxMother, -1)
        valid_mother = mother_idx >= 0
        mother_pdgId = ak.where(valid_mother, genPart[ak.where(valid_mother, mother_idx, 0)].pdgId, -999)
        gen_pdgId = ak.where(valid_photon, photon_genparts.pdgId, -999)

        lead_mother_pdgId, sublead_mother_pdgId = mother_pdgId[:, 0], mother_pdgId[:, 1]
        lead_gen_pdgId, sublead_gen_pdgId = gen_pdgId[:, 0], gen_pdgId[:, 1]

        lead_pdg = ak.to_numpy(lead_mother_pdgId)
        sublead_pdg = ak.to_numpy(sublead_mother_pdgId)
        lead_gen = ak.to_numpy(lead_gen_pdgId)
        sublead_gen = ak.to_numpy(sublead_gen_pdgId)

        same_genpart = photons.genPartIdx[:, 0] == photons.genPartIdx[:, 1]
        same_genpart = ak.fill_none(same_genpart, False)

        # -------------------------------------------------
        # Check (-11, -11) mother combination
        # -------------------------------------------------

        mask41 = (lead_mother_pdgId == -11) & (sublead_mother_pdgId == -11)
        mask41 = ak.fill_none(mask41, False)

        # -------------------------------------------------
        # Accumulate global counts
        # -------------------------------------------------

        same_genpart_count += int(ak.sum(same_genpart))
        mask41_count += int(ak.sum(mask41))
        same_genpart_mask41_count += int(ak.sum(same_genpart & mask41))

        ids, counts = np.unique(lead_pdg, return_counts=True)
        for pdgid, count in zip(ids, counts):
            pdgid = int(pdgid)
            lead_counts[pdgid] = lead_counts.get(pdgid, 0) + int(count)

        ids, counts = np.unique(sublead_pdg, return_counts=True)
        for pdgid, count in zip(ids, counts):
            pdgid = int(pdgid)
            sublead_counts[pdgid] = sublead_counts.get(pdgid, 0) + int(count)

        pairs = zip(lead_pdg, sublead_pdg)
        for lead, sublead in pairs:
            key = (int(lead), int(sublead))
            pair_counts[key] = pair_counts.get(key, 0) + 1

        for lg, lm, sg, sm in zip(lead_gen, lead_pdg, sublead_gen, sublead_pdg):
            lead_key = (int(lg), int(lm))
            sublead_key = (int(sg), int(sm))
            key = (lead_key, sublead_key)
            genmother_pair_counts[key] = genmother_pair_counts.get(key, 0) + 1

    except Exception as e:
        print(f"\nERROR processing:\n{file}\n{e}")


# =========================================================
# Process all ROOT files
# =========================================================

for i, file in enumerate(files):
    process_file(file)
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(files)} files")

# =========================================================
# Final results over ALL files
# =========================================================

print("\n==========================================")
print("GenPart consistency checks")
print("==========================================")

print(f"Total events with same lead/sublead GenPart: {same_genpart_count}")
print(f"Total (-11, -11) mother events:              {mask41_count}")
print(f"Total (-11, -11) events with same GenPart:   {same_genpart_mask41_count}")

print("\n------------------------------------------")
print(f"Fraction same GenPart: {same_genpart_count / mask41_count:.4f}" if mask41_count else "Fraction same GenPart: N/A")
print(f"Fraction same GenPart among (-11,-11): {same_genpart_mask41_count / mask41_count:.4f}" if mask41_count else "Fraction same GenPart among (-11,-11): N/A")
print("------------------------------------------")

print(f"Total selected events: {total_selected_events}")


# =========================================================
# Print distributions
# =========================================================

print("\n==========================================")
print("Lead photon mother PDG-ID distribution")
print("==========================================")
for pdgid, count in sorted(lead_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{pdgid:6d}  {pdgid_to_name(pdgid):15s}  {count}")

print("\n==========================================")
print("Sublead photon mother PDG-ID distribution")
print("==========================================")
for pdgid, count in sorted(sublead_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{pdgid:6d}  {pdgid_to_name(pdgid):15s}  {count}")


# =========================================================
# Top 20 for 1D
# =========================================================

lead_ids = np.array(sorted(lead_counts, key=lead_counts.get, reverse=True)[:20], dtype=int)
sublead_ids = np.array(sorted(sublead_counts, key=sublead_counts.get, reverse=True)[:20], dtype=int)

lead_values = np.array([lead_counts[x] for x in lead_ids])
sublead_values = np.array([sublead_counts[x] for x in sublead_ids])


# =========================================================
# 1D plots
# =========================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

x = np.arange(len(lead_ids))
axes[0].bar(x, lead_values)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{pdgid}\n{pdgid_to_name(pdgid)}" for pdgid in lead_ids], rotation=45, ha="right")
axes[0].set_ylabel("Events")
axes[0].set_title(f"Mother PDG ID of Lead Photon {args.label}")
axes[0].grid(axis="y", alpha=0.3)

x = np.arange(len(sublead_ids))
axes[1].bar(x, sublead_values)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{pdgid}\n{pdgid_to_name(pdgid)}" for pdgid in sublead_ids], rotation=45, ha="right")
axes[1].set_xlabel("Mother PDG ID")
axes[1].set_ylabel("Events")
axes[1].set_title(f"Mother PDG ID of Sublead Photon {args.label}")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(args.plot_name, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n1D plot saved to: {args.plot_name}")


# =========================================================
# Top 15 PDG-ID types for 2D
# =========================================================

all_pdg_ids = set(lead_counts) | set(sublead_counts)
total_counts = {pdgid: lead_counts.get(pdgid, 0) + sublead_counts.get(pdgid, 0) for pdgid in all_pdg_ids}
top15 = sorted(total_counts, key=total_counts.get, reverse=True)[:15]


# =========================================================
# Build 2D matrix
# =========================================================

matrix = np.zeros((len(top15), len(top15)), dtype=int)

for (lead, sublead), count in pair_counts.items():
    if lead in top15 and sublead in top15:
        matrix[top15.index(lead), top15.index(sublead)] = count


# =========================================================
# 2D plot
# =========================================================

fig, ax = plt.subplots(figsize=(13, 11))

im = ax.imshow(matrix, aspect="auto", origin ="lower")

labels = [f"{pdgid}\n{pdgid_to_name(pdgid)}" for pdgid in top15]

ax.set_xticks(np.arange(len(top15)))
ax.set_yticks(np.arange(len(top15)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)

ax.set_xlabel("Sublead photon mother PDG ID")
ax.set_ylabel("Lead photon mother PDG ID")
ax.set_title(f"Lead vs Sublead Photon Mother PDG ID {args.label}")

# for i in range(len(top15)):
#     for j in range(len(top15)):
#         if matrix[i, j] > 0:
#             ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

norm = colors.Normalize(vmin=matrix.min(), vmax=matrix.max())
cmap = plt.get_cmap("viridis")

for i in range(len(top15)):
    for j in range(len(top15)):
        if matrix[i, j] > 0:

            rgba = cmap(norm(matrix[i, j]))

            # Perceived brightness
            brightness = (
                0.299 * rgba[0] +
                0.587 * rgba[1] +
                0.114 * rgba[2]
            )

            text_color = "white" if brightness < 0.5 else "black"

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color
            )

fig.colorbar(im, ax=ax, label="Events")

plt.tight_layout()

base, ext = os.path.splitext(args.plot_name)
plot_2d_name = f"{base}_2D{ext}"

plt.savefig(plot_2d_name, dpi=300, bbox_inches="tight")
plt.close()

print(f"2D plot saved to: {plot_2d_name}")


# =========================================================
# Forced PDG-ID list for 2D plot
# =========================================================

# forced_pdgids = [23, 11, -11, 13, -13, 111, 22, 21, 2, -4, 1, 4, -5, 3, 5, -421, 221, 411, -413, 421]
forced_pdgids = [23, 11, -11, 13, -13, 111, 22, 21, 2, -4, 1, 4, -5, 3, 5, -421, 221, 411, -413, 421, -999]

x_pdgids = forced_pdgids
y_pdgids = forced_pdgids


# =========================================================
# Build 2D matrix
# =========================================================

matrix = np.zeros((len(y_pdgids), len(x_pdgids)), dtype=int)

for (lead, sublead), count in pair_counts.items():
    if lead in y_pdgids and sublead in x_pdgids:
        i = y_pdgids.index(lead)
        j = x_pdgids.index(sublead)
        matrix[i, j] = count


# =========================================================
# 2D plot
# =========================================================

fig, ax = plt.subplots(figsize=(18, 16))

im = ax.imshow(matrix, aspect="equal", origin="lower")

x_labels = ["Invalid/Unmatched" if pdgid == -999 else f"{pdgid}\n{pdgid_to_name(pdgid)}" for pdgid in x_pdgids]
y_labels = ["Invalid/Unmatched" if pdgid == -999 else f"{pdgid}\n{pdgid_to_name(pdgid)}" for pdgid in y_pdgids]

ax.set_xticks(np.arange(len(x_pdgids)))
ax.set_yticks(np.arange(len(y_pdgids)))

ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_yticklabels(y_labels)

ax.set_xlabel("Sublead photon mother PDG ID")
ax.set_ylabel("Lead photon mother PDG ID")
ax.set_title(f"Lead vs Sublead Photon Mother PDG ID {args.label}")

# for i in range(len(y_pdgids)):
#     for j in range(len(x_pdgids)):
#         ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9)

norm = colors.Normalize(vmin=matrix.min(), vmax=matrix.max())
cmap = plt.get_cmap("viridis")

for i in range(len(y_pdgids)):
    for j in range(len(x_pdgids)):
        if matrix[i, j] != 0:

            rgba = cmap(norm(matrix[i, j]))

            # Perceived brightness
            brightness = (
                0.299 * rgba[0] +
                0.587 * rgba[1] +
                0.114 * rgba[2]
            )

            text_color = "white" if brightness < 0.5 else "black"

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color
            )

fig.colorbar(im, ax=ax, label="Events")

plt.tight_layout()

base, ext = os.path.splitext(args.plot_name)
plot_forced2d_name = f"{base}_2Dforced{ext}"

plt.savefig(plot_forced2d_name, dpi=300, bbox_inches="tight")
plt.close()

print(f"2D plot saved to: {plot_forced2d_name}")


# =========================================================
# GenPart PDG ID + Mother PDG ID 2D plot
# =========================================================

allowed_pdgids = [22, 23, 11, -11, 13, -13]

def genmother_label(gen_pdgid, mother_pdgid):
    return f"{pdgid_to_name(mother_pdgid)} → {pdgid_to_name(gen_pdgid)}"

lead_combined_counts = {}
sublead_combined_counts = {}

for (lead, sublead), count in genmother_pair_counts.items():
    if lead[0] in allowed_pdgids and lead[1] in allowed_pdgids and sublead[0] in allowed_pdgids and sublead[1] in allowed_pdgids:
        lead_combined_counts[lead] = lead_combined_counts.get(lead, 0) + count
        sublead_combined_counts[sublead] = sublead_combined_counts.get(sublead, 0) + count

combined_types = sorted(set(lead_combined_counts) | set(sublead_combined_counts), key=lambda x: (-lead_combined_counts.get(x, 0) - sublead_combined_counts.get(x, 0), x))

matrix = np.zeros((len(combined_types), len(combined_types)), dtype=int)

for (lead, sublead), count in genmother_pair_counts.items():
    if lead in combined_types and sublead in combined_types:
        i = combined_types.index(lead)
        j = combined_types.index(sublead)
        matrix[i, j] += count

fig, ax = plt.subplots(figsize=(18, 16))

im = ax.imshow(matrix, aspect="equal", origin="lower")

labels = [genmother_label(g, m) for g, m in combined_types]

ax.set_xticks(np.arange(len(combined_types)))
ax.set_yticks(np.arange(len(combined_types)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)

ax.set_xlabel("Sublead: GenPart PDG ID → Mother PDG ID")
ax.set_ylabel("Lead: GenPart PDG ID → Mother PDG ID")
ax.set_title(f"Lead vs Sublead GenPart + Mother PDG ID {args.label}")

# for i in range(len(combined_types)):
#     for j in range(len(combined_types)):
#         if matrix[i, j] > 0:
#             ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)

norm = colors.Normalize(vmin=matrix.min(), vmax=matrix.max())
cmap = plt.get_cmap("viridis")

for i in range(len(combined_types)):
    for j in range(len(combined_types)):
        if matrix[i, j] > 0:

            rgba = cmap(norm(matrix[i, j]))

            # Perceived brightness
            brightness = (
                0.299 * rgba[0] +
                0.587 * rgba[1] +
                0.114 * rgba[2]
            )

            text_color = "white" if brightness < 0.5 else "black"

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color
            )

fig.colorbar(im, ax=ax, label="Events")

plt.tight_layout()

base, ext = os.path.splitext(args.plot_name)
plot_genmother_name = f"{base}_GenMother2D{ext}"

plt.savefig(plot_genmother_name, dpi=300, bbox_inches="tight")
plt.close()

print(f"GenPart + mother 2D plot saved to: {plot_genmother_name}")
