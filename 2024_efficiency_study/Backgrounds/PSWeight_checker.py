import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema


with open("TTto2L2Nu_2024.text") as f:
    files = [line.strip() for line in f if line.strip()]

for file in files:
    print(f"\nChecking file: {file}")

    try:

        events = NanoEventsFactory.from_root(
            f"{file}:Events",
            schemaclass=NanoAODSchema,
        ).events()

        ps = events.PSWeight

        # Check for None (missing) entries
        if ak.any(ak.is_none(ps)):
            print("Some PSWeight entries are None")

        # Check for empty lists (no weights in some events)
        elif ak.any(ak.num(ps) == 0):
            print("Some events have empty PSWeight")

        # Check for malformed entries (not length 4)
        elif ak.any(ak.num(ps) != 4):
            print("Some events do not have 4 PS weights")

        else:
            print("All events have valid PSWeight")

    except Exception as e:
        print(f"Skipping file (cannot open/read): {e}")
        continue