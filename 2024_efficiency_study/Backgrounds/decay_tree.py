from particle import Particle
import numpy as np
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import os
import glob
import argparse


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


def main():
    parser = argparse.ArgumentParser(description="Print GenPart trees from a ROOT file")
    parser.add_argument("--input-file", required=True, help="Path to the input ROOT file")
    parser.add_argument("--n-events", type=int, default=20, help="Number of events to print")
    args = parser.parse_args()

    factory = NanoEventsFactory.from_root(f"{args.input_file}:Events", schemaclass=NanoAODSchema)
    events = factory.events()

    for i in range(min(args.n_events, len(events))):
        print(f"\n{'-' * 80}")
        print(f"EVENT {i}")
        print(f"{'-' * 80}")
        print_genpart_tree(events.GenPart, i)

if __name__ == "__main__":
    main()
