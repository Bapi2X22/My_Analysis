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
import uproot
import pyarrow.parquet as pq
import os
import pyarrow as pa

from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.jetID import add_jetId
from coffea.analysis_tools import Weights

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

def process_file(file, dataset_name):
    print(f"Processing: {file}", flush=True)
    with uproot.open(file) as f:
        metadata = f["Metadata"]
        sum_genw_presel = metadata["sum_genw_presel"].array(library="np")[0]
    factory = NanoEventsFactory.from_root(f"{file}:Events", schemaclass=NanoAODSchema)
    events = factory.events()

    events["Photon"] = add_photon_SC_eta(events.Photon, events.PV)
    events["Electron"] = ak.with_field(events.Electron, events.Electron.eta + events.Electron.deltaEtaSC, "ScEta")
    Jets = events.Jet
    Jet_id = add_jetId(Jets, 15, "2024", flattenUnflatten=True)
    events["Jet"] = ak.with_field(Jets, Jet_id, "jetId")

    # Apply object corrections

    corrections_list = ["Smearing", "Material", "FNUF", "Electron_Smearing_EGM", "jec_jet_syst", "MuonScaRe", "energyErrShift"]

    for correction in corrections_list:

        varying_function = available_object_corrections[correction]
        print(f"Applying {correction}")
        events = varying_function(
            events=events, year="2024"
        )

    electrons = events.Electron
    muons = events.Muon
    photons = events.Photon
    jets = events.Jet
    photons["mass"] = ak.zeros_like(photons.pt)
    photons["charge"] = ak.zeros_like(photons.pt)
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

    # Apply weight corrections

    weight_corrections_list = ["TriggerSF_singleLep", "Pileup", "ElectronIdSFWP80iso", "MuonIsoMediumSF_IdMedium", "MuonIdMediumSF", "bTagFixedWP_UParTAK4Medium_bbgg"]

    event_weights = Weights(size=len(events[event_mask]))

    # corrections to event weights:
    for correction_name in weight_corrections_list:
        varying_function = available_weight_corrections[correction_name]

        try:

            event_weights = varying_function(
                events=selected_events,
                muons=muons[event_mask],
                electrons=electrons[event_mask],
                jets=jets[event_mask],
                weights=event_weights,
                dataset_name=dataset_name,
                year="2024",
            )
        except Exception:
            print(f"FAILED INSIDE {correction_name}", flush=True)
            import traceback
            traceback.print_exc()
            raise

    selected_events["weight_central"] = event_weights.weight()

    event_weights._weight = (selected_events["genWeight"] * selected_events["weight_central"])
    selected_events["weight"] = event_weights.weight()

    print(f"Selected {len(selected_events)} events", flush=True)
    return selected_events, sum_genw_presel

# def process_batch(batch_files, output_file, dataset_name):
#     print("=" * 80, flush=True)
#     print(f"Starting batch with {len(batch_files)} files", flush=True)
#     selected_events = []
#     n_selected_total = 0
#     n_failed = 0
#     total_sum_genw_presel = 0 
#     for file in batch_files:
#         try:
#             selected, sum_genw_presel = process_file(file, dataset_name)
#             total_sum_genw_presel += sum_genw_presel
#             if len(selected) > 0:
#                 selected_events.append(selected)
#                 n_selected_total += len(selected)
#         except Exception as e:
#             n_failed += 1
#             print(f"ERROR processing {file}", flush=True)
#             print(f"{type(e).__name__}: {e}", flush=True)
#     if len(selected_events) > 0:
#         batch_events = ak.concatenate(selected_events, axis=0)
#         pa_table = ak.to_arrow_table(batch_events, extensionarray=False)
#         metadata = {b"sum_genw_presel": str(total_sum_genw_presel).encode()}
#         merged_metadata = {**metadata, **(pa_table.schema.metadata or {})}
#         pa_table = pa_table.replace_schema_metadata(merged_metadata)
#         pq.write_table(pa_table, output_file, compression="snappy")
#         del batch_events
#     else:
#         print("No events selected in this batch.", flush=True)
#     del selected_events
#     gc.collect()
#     print(f"Batch finished: {n_selected_total} selected, {n_failed} failed", flush=True)

def process_batch(batch_files, output_file, dataset_name):

    print("=" * 80, flush=True)
    print(f"Starting batch with {len(batch_files)} files", flush=True)

    selected_events = []
    n_selected_total = 0
    n_failed = 0
    total_sum_genw_presel = 0

    empty_schema = None

    for file in batch_files:

        try:

            selected, sum_genw_presel = process_file(file, dataset_name)

            total_sum_genw_presel += sum_genw_presel

            # Keep the schema from the first successfully processed file
            if empty_schema is None:
                empty_schema = ak.to_arrow_table(selected, extensionarray=False).schema

            if len(selected) > 0:
                selected_events.append(selected)
                n_selected_total += len(selected)

        except Exception as e:

            n_failed += 1

            print(f"ERROR processing {file}", flush=True)
            print(f"{type(e).__name__}: {e}", flush=True)

    # ========================================================
    # Write output parquet
    # ========================================================

    if len(selected_events) > 0:

        # ----------------------------------------------------
        # Normal case: selected events exist
        # ----------------------------------------------------

        batch_events = ak.concatenate(selected_events, axis=0)

        pa_table = ak.to_arrow_table(batch_events, extensionarray=False)

        del batch_events

    else:

        # ----------------------------------------------------
        # Empty case: no selected events
        # ----------------------------------------------------

        print("No events selected in this batch. Writing empty parquet with metadata.", flush=True)

        if empty_schema is None:

            print("WARNING: No successfully processed files in this batch. Cannot determine parquet schema.", flush=True)

            return 1

        # Create an empty Arrow table with the same schema
        pa_table = pa.Table.from_arrays([pa.array([], type=field.type) for field in empty_schema], schema=empty_schema)

    # ========================================================
    # Add metadata
    # ========================================================

    metadata = {b"sum_genw_presel": str(total_sum_genw_presel).encode()}

    merged_metadata = {**(pa_table.schema.metadata or {}), **metadata}

    pa_table = pa_table.replace_schema_metadata(merged_metadata)

    # ========================================================
    # Write parquet
    # ========================================================

    pq.write_table(pa_table, output_file, compression="snappy")

    print(f"Written parquet: {output_file}", flush=True)

    del pa_table
    del selected_events

    gc.collect()

    print(f"Batch finished: {n_selected_total} selected, {n_failed} failed", flush=True)

    return 0

def main():
    if "--worker" in sys.argv:
        worker_parser = argparse.ArgumentParser()
        worker_parser.add_argument("--worker", action="store_true")
        worker_parser.add_argument("--file-list", required=True)
        worker_parser.add_argument("--worker-output", required=True)
        worker_parser.add_argument("--dataset-name", required=True)
        args = worker_parser.parse_args()
        with open(args.file_list) as f:
            batch_files = [line.strip() for line in f if line.strip()]
        # process_batch(batch_files, args.worker_output, args.dataset_name)
        return_code = process_batch(batch_files, args.worker_output, args.dataset_name)

        return return_code
    return 0

    parser = argparse.ArgumentParser(description="Apply BBGG selection to ROOT files and save selected events to Parquet.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input ROOT files.")
    parser.add_argument("--output", required=True, help="Output directory for Parquet files.")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of ROOT files per worker process.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_name = os.path.basename(os.path.normpath(input_dir))

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

        command = [sys.executable, sys.argv[0], "--worker", "--file-list", str(file_list), "--worker-output", str(output_file), "--dataset-name", sample_name]

        result = subprocess.run(command, check=False)

        if result.returncode != 0:
            print(f"WARNING: Batch {batch_number + 1} failed with return code {result.returncode}", flush=True)
        else:
            print(f"Batch {batch_number + 1} completed successfully", flush=True)

        file_list.unlink(missing_ok=True)

    shutil.rmtree(batch_list_dir, ignore_errors=True)

    print("\nAll batches finished.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    sys.exit(main())