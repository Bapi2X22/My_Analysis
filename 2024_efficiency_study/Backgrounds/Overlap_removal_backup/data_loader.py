import awkward as ak
import os
import glob
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

def load_parquet(filename):
    return ak.from_parquet(filename)

import os
import glob

import awkward as ak
import numpy as np

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema


def get_root_files(directory, n_files=None):
    files = sorted(glob.glob(os.path.join(directory, "*.root")))
    if n_files is not None:
        files = files[:n_files]
    return files


def get_sum_genweight(files):
    sum_genweight = 0.0
    for filename in files:
        events = NanoEventsFactory.from_root(filename,treepath="Events",schemaclass=NanoAODSchema).events()
        sum_genweight += ak.sum(events.genWeight)
    return float(sum_genweight)

def load_root_file(filename):
    events = NanoEventsFactory.from_root(filename,treepath="Events",schemaclass=NanoAODSchema).events()
    return events

def load_root_process(directories,n_files,xsecs,background_dir,lumi_fb):
    if isinstance(directories, str):
        directories = [directories]
    if isinstance(n_files, int):
        n_files = [n_files]
    if isinstance(xsecs, (int, float)):
        xsecs = [xsecs]
    results = []
    for directory, n_file, xsec_pb in zip(directories,n_files,xsecs):
        directory = os.path.join(background_dir, directory)
        files = get_root_files(directory, n_file)
        if not files:
            raise FileNotFoundError(f"No ROOT files found in {directory}")
        print(f"  Directory: {directory}")
        print(f"  Found {len(glob.glob(os.path.join(directory, '*.root')))} ROOT files")
        print(f"  Processing {len(files)} ROOT files")
        print(f"  Xsec: {xsec_pb} pb")
        sum_genweight = get_sum_genweight(files)
        print(f"  Sum genWeight: {sum_genweight:.6e}")
        for filename in files:
            events = load_root_file(filename)
            genweight = ak.to_numpy(events.genWeight)
            weights = (genweight/ sum_genweight* xsec_pb* lumi_fb* 1000.0)
            results.append((events, weights))
    return results


def get_expected_weight(events, xsec_pb, lumi_fb):
    return events.weight * xsec_pb * lumi_fb * 1000.0


def load_process(filename, xsec_pb, lumi_fb):
    events = load_parquet(filename)

    weights = get_expected_weight(events, xsec_pb, lumi_fb)

    return events, weights