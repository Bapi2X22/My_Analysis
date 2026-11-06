import uproot

def get_root_file_info(filepath):
    with uproot.open(filepath) as root_file:
        root_uuid = str(root_file.file.uuid)
        n_events = root_file["Events"].num_entries

    return root_uuid, n_events
