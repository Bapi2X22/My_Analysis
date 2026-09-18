import awkward as ak
import numpy as np
from pathlib import Path
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import correctionlib
import argparse
import gc
import sys
import subprocess
import shutil

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

def delta_r_mask(first, second, threshold):
    mval = first.metric_table(second)
    return ak.all(mval > threshold, axis=-1)

def build_diphoton_candidates(photons, min_pt_lead_photon):
    sorted_photons = photons[ak.argsort(photons.pt, ascending=False)]
    diphotons = ak.combinations(sorted_photons, 2, fields=["pho_lead", "pho_sublead"])
    diphotons = diphotons[diphotons["pho_lead"].pt > min_pt_lead_photon]
    diphoton_4mom = diphotons["pho_lead"] + diphotons["pho_sublead"]
    diphotons["pt"] = diphoton_4mom.pt
    diphotons["eta"] = diphoton_4mom.eta
    diphotons["phi"] = diphoton_4mom.phi
    diphotons["mass"] = diphoton_4mom.mass
    diphotons["charge"] = diphoton_4mom.charge
    diphoton_pz = diphoton_4mom.z
    diphoton_e = diphoton_4mom.energy
    diphotons["rapidity"] = 0.5 * np.log((diphoton_e + diphoton_pz) / (diphoton_e - diphoton_pz))
    diphotons = diphotons[ak.argsort(diphotons.pt, ascending=False)]
    return ak.with_name(diphotons, "PtEtaPhiMCandidate")

def photon_preselection_bbgg(photons, events, electrons, muons):
    dr_cut = delta_r_mask(photons, electrons, 0.2)
    dr_cut_muon = delta_r_mask(photons, muons, 0.2)
    return photons[(~photons.pixelSeed) & (photons.pt > 15.0) & (photons.isScEtaEB | photons.isScEtaEE) & dr_cut & dr_cut_muon & ak.where(photons.isScEtaEB, photons.mvaID > 0.0439603, photons.mvaID > -0.249526)]

def select_electrons_bbgg(electrons):
    return (electrons.pt > 30) & (abs(electrons.eta) < 2.5) & electrons.mvaIso_WP80

def select_muons_bbgg(muons):
    return (muons.pt > 26.0) & (abs(muons.eta) < 2.4) & muons.mediumId & (muons.pfIsoId >= 3) & muons.isGlobal

def select_jets_bbgg(jets, diphotons, muons, electrons, clean_jet_pho=True, clean_jet_ele=True, clean_jet_muo=True):
    jetId_cut = jets.jetId >= 2
    pt_cut = jets.pt > 20
    eta_cut = abs(jets.eta) < 2.4
    if clean_jet_pho and ak.any(ak.num(diphotons.pt) > 0):
        lead = ak.with_name(ak.zip({"pt": diphotons.pho_lead.pt, "eta": diphotons.pho_lead.eta, "phi": diphotons.pho_lead.phi, "mass": diphotons.pho_lead.mass, "charge": diphotons.pho_lead.charge}), "PtEtaPhiMCandidate")
        sublead = ak.with_name(ak.zip({"pt": diphotons.pho_sublead.pt, "eta": diphotons.pho_sublead.eta, "phi": diphotons.pho_sublead.phi, "mass": diphotons.pho_sublead.mass, "charge": diphotons.pho_sublead.charge}), "PtEtaPhiMCandidate")
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        dr_pho_lead_cut = delta_r_mask(jets, lead, 0.4)
        dr_pho_sublead_cut = delta_r_mask(jets, sublead, 0.4)
    else:
        dr_pho_lead_cut = jets.pt > -1
        dr_pho_sublead_cut = jets.pt > -1
    if clean_jet_ele and ak.any(ak.num(electrons.pt) > 0):
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        dr_electrons_cut = delta_r_mask(jets, electrons, 0.4)
    else:
        dr_electrons_cut = jets.pt > -1
    if clean_jet_muo and ak.any(ak.num(muons.pt) > 0):
        jets = ak.with_name(jets, "PtEtaPhiMCandidate")
        dr_muons_cut = delta_r_mask(jets, muons, 0.4)
    else:
        dr_muons_cut = jets.pt > -1
    return jetId_cut & pt_cut & eta_cut & dr_pho_lead_cut & dr_pho_sublead_cut & dr_electrons_cut & dr_muons_cut

def process_file(file):
    print(f"Processing: {file}", flush=True)
    factory = NanoEventsFactory.from_root(f"{file}:Events", schemaclass=NanoAODSchema)
    events = factory.events()
    electrons = events.Electron
    muons = events.Muon
    photons = events.Photon
    jets = events.Jet
    photons["mass"] = ak.zeros_like(photons.pt)
    photons["charge"] = ak.zeros_like(photons.pt)
    jet_id = add_jetId(jets, flattenUnflatten=True)
    events["Jet"] = ak.with_field(events.Jet, jet_id, "jetId")
    jets = events.Jet
    electrons = electrons[select_electrons_bbgg(electrons)]
    muons = muons[select_muons_bbgg(muons)]
    photons = photon_preselection_bbgg(photons, events, electrons, muons)
    diphotons = build_diphoton_candidates(photons, 15.0)
    jets = jets[select_jets_bbgg(jets, diphotons, muons, electrons)]
    one_ele = ak.num(electrons) == 1
    zero_ele = ak.num(electrons) == 0
    one_mu = ak.num(muons) == 1
    zero_mu = ak.num(muons) == 0
    electron_channel = one_ele & zero_mu
    muon_channel = one_mu & zero_ele
    lepton_channel_mask = electron_channel | muon_channel
    b_jets = jets[jets.btagUParTAK4B > 0.1272]
    at_least_one_bjet = ak.num(b_jets) >= 1
    at_least_two_photons = ak.num(photons) >= 2
    event_mask = lepton_channel_mask & at_least_two_photons & at_least_one_bjet
    selected_events = ak.Array(events[event_mask])
    print(f"Selected {len(selected_events)} events", flush=True)
    return selected_events

def process_batch(batch_files, output_file):
    print("=" * 80, flush=True)
    print(f"Starting batch with {len(batch_files)} files", flush=True)
    selected_events = []
    n_selected_total = 0
    n_failed = 0
    for file in batch_files:
        try:
            selected = process_file(file)
            if len(selected) > 0:
                selected_events.append(selected)
                n_selected_total += len(selected)
        except Exception as e:
            n_failed += 1
            print(f"ERROR processing {file}", flush=True)
            print(f"{type(e).__name__}: {e}", flush=True)
    if len(selected_events) > 0:
        batch_events = ak.concatenate(selected_events, axis=0)
        ak.to_parquet(batch_events, output_file, compression="snappy")
        print(f"Written {len(batch_events)} events to {output_file}", flush=True)
        del batch_events
    else:
        print("No events selected in this batch.", flush=True)
    del selected_events
    gc.collect()
    print(f"Batch finished: {n_selected_total} selected, {n_failed} failed", flush=True)

def main():
    if "--worker" in sys.argv:
        worker_parser = argparse.ArgumentParser()
        worker_parser.add_argument("--worker", action="store_true")
        worker_parser.add_argument("--file-list", required=True)
        worker_parser.add_argument("--worker-output", required=True)
        args = worker_parser.parse_args()
        with open(args.file_list) as f:
            batch_files = [line.strip() for line in f if line.strip()]
        process_batch(batch_files, args.worker_output)
        return

    parser = argparse.ArgumentParser(description="Apply BBGG selection to ROOT files and save selected events to Parquet.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input ROOT files.")
    parser.add_argument("--output", required=True, help="Output directory for Parquet files.")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of ROOT files per worker process.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(str(f) for f in input_dir.glob("*.root"))

    print(f"Found {len(files)} ROOT files")

    if len(files) == 0:
        print("No ROOT files found.")
        return

    batches = [files[i:i + args.chunk_size] for i in range(0, len(files), args.chunk_size)]

    print(f"Number of batches: {len(batches)}")

    batch_list_dir = output_dir / ".batch_lists"
    batch_list_dir.mkdir(parents=True, exist_ok=True)

    for batch_number, batch_files in enumerate(batches):
        output_file = output_dir / f"part_{batch_number:05d}.parquet"
        file_list = batch_list_dir / f"batch_{batch_number:05d}.txt"

        with open(file_list, "w") as f:
            f.write("\n".join(batch_files))

        print(f"\nBatch {batch_number + 1}/{len(batches)}: {len(batch_files)} files", flush=True)

        command = [sys.executable, sys.argv[0], "--worker", "--file-list", str(file_list), "--worker-output", str(output_file)]

        result = subprocess.run(command, check=False)

        if result.returncode != 0:
            print(f"WARNING: Batch {batch_number + 1} failed with return code {result.returncode}", flush=True)
        else:
            print(f"Batch {batch_number + 1} completed successfully", flush=True)

        file_list.unlink(missing_ok=True)

    shutil.rmtree(batch_list_dir, ignore_errors=True)

    print("\nAll batches finished.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
