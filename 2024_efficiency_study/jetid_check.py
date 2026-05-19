import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema


import awkward as ak
import correctionlib
import os
import logging

logger = logging.getLogger(__name__)

json = "jetid.json.gz"


def add_jetId(jets, nano_version, year, flattenUnflatten=False):
    """
    Add (or recompute) jet ID to the jets object based on the NanoAOD version.
    """
    abs_eta = abs(jets.eta)

    # Return the existing jetId for NanoAOD versions below 12
    if nano_version < 12:
        return jets.jetId

    # For NanoAOD version 12 and above, we recompute the jet ID criteria
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    else:
        if nano_version == 12:
            # Default tight
            passJetIdTight = ak.where(
                abs_eta <= 2.7,
                (jets.jetId & (1 << 1)) > 0,  # Tight criteria for abs_eta <= 2.7
                ak.where(
                    (abs_eta > 2.7) & (abs_eta <= 3.0),
                    ((jets.jetId & (1 << 1)) > 0) & (jets.neHEF < 0.99),  # Tight criteria for 2.7 < abs_eta <= 3.0
                    ((jets.jetId & (1 << 1)) > 0) & (jets.neEmEF < 0.4)  # Tight criteria for 3.0 < abs_eta
                )
            )

            # Default tight lepton veto
            passJetIdTightLepVeto = ak.where(
                abs_eta <= 2.7,
                passJetIdTight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8),  # add lepton veto for abs_eta <= 2.7
                passJetIdTight  # No lepton veto for 2.7 < abs_eta
            )

            return (passJetIdTight * (1 << 1)) | (passJetIdTightLepVeto * (1 << 2))

        else:
            # Example code: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jetidExample.py?ref_type=heads
            # Load CorrectionSet
            if year == "2025":
                logger.warning("There is no dedicated 2025 jetID. As the 2025 PUPPI tune is the same as the 2024 one, the 2024 jetID used! ")
                year = "2024"

            jerc_json = {
                "2022preEE": os.path.join(
                    os.path.dirname(__file__),
                    "../systematics/JSONs/POG/JME/2022_Summer22/jetid.json.gz",
                ),
                "2022postEE": os.path.join(
                    os.path.dirname(__file__),
                    "../systematics/JSONs/POG/JME/2022_Summer22EE/jetid.json.gz",
                ),
                "2023preBPix": os.path.join(
                    os.path.dirname(__file__),
                    "../systematics/JSONs/POG/JME/2023_Summer23/jetid.json.gz",
                ),
                "2023postBPix": os.path.join(
                    os.path.dirname(__file__),
                    "../systematics/JSONs/POG/JME/2023_Summer23BPix/jetid.json.gz",
                ),
                "2024": os.path.join(
                    os.path.dirname(__file__),
                    "../systematics/JSONs/POG/JME/2024_Summer24/jetid.json.gz",
                ),
            }

            cset = correctionlib.CorrectionSet.from_file(json)

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

file = "root://cms-xrd-global.cern.ch//store/mc/RunIII2024Summer24NanoAODv15/WHToAA-AToBB-AToGG_Par-M-15_TuneCP5_13p6TeV_madgraph-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/2550000/9faec35d-2846-4be9-80a0-b954c4ae0631.root"
#factory = NanoEventsFactory.from_root(
#    f"{file}:Events",
#    schemaclass=NanoAODSchema,
#)

factory = NanoEventsFactory.from_root(
    file,
    treepath="Events",
    schemaclass=NanoAODSchema,   # <--- use schemaclass, not schema
)

events = factory.events()

import correctionlib

json = "jetid.json"

cset = correctionlib.CorrectionSet.from_file(json)

jets = events.Jet


Jet_id = add_jetId(jets, 15, "2024", flattenUnflatten=True)
events["Jet"] = ak.with_field(jets, jets.pt, "jetId")

print(events.Jet.fields)
print(events.Photon.fields)

# eval_dict = {
#     "eta": jets.eta,
#     "chHEF": jets.chHEF,
#     "neHEF": jets.neHEF,
#     "chEmEF": jets.chEmEF,
#     "neEmEF": jets.neEmEF,
#     "muEF": jets.muEF,
#     "chMultiplicity": jets.chMultiplicity,
#     "neMultiplicity": jets.neMultiplicity,
#     "multiplicity": jets.chMultiplicity + jets.neMultiplicity
#             }

# idTight = cset["AK4PUPPI_Tight"]
# inputsTight = [eval_dict[input.name] for input in idTight.inputs]
# idTight_value = idTight.evaluate(*inputsTight) * 2  # equivalent to bit2

# # Default tight lepton veto
# idTightLepVeto = cset["AK4PUPPI_TightLeptonVeto"]
# inputsTightLepVeto = [eval_dict[input.name] for input in idTightLepVeto.inputs]
# idTightLepVeto_value = idTightLepVeto.evaluate(*inputsTightLepVeto) * 4  # equivalent to bit3

# # Default jet ID
# id_value = idTight_value + idTightLepVeto_value

# print(id_value)
