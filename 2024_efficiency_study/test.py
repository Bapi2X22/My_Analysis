import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

file = "root://cms-xrd-global.cern.ch//store/mc/RunIII2024Summer24NanoAODv15/WHToAA-AToBB-AToGG_Par-M-15_TuneCP5_13p6TeV_madgraph-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/2550000/9faec35d-2846-4be9-80a0-b954c4ae0631.root"
factory = NanoEventsFactory.from_root(
    f"{file}:Events",
    schemaclass=NanoAODSchema,
)
events = factory.events()

print(events.Jet.fields)


