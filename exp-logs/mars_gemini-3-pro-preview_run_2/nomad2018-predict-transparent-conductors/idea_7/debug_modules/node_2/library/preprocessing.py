import os
import numpy as np
import pandas as pd
import ase.io
from ase.neighborlist import neighbor_list
from library.config import (
    ATOM_MAP,
    CUTOFF_RADIUS,
    MAX_NEIGHBORS,
    INPUT_DIR,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    TARGET_COLS,
    WORKING_DIR,
)


def get_pbc_graph(atoms, cutoff=CUTOFF_RADIUS, max_neighbors=MAX_NEIGHBORS):
    """
    Computes the neighbor list and constructs the graph for a periodic structure.

    Args:
        atoms (ase.Atoms): The atomic structure.
        cutoff (float): Cutoff radius for neighbor search.
        max_neighbors (int): Maximum number of neighbors per node.

    Returns:
        dict: A dictionary containing:
            - node_feats (np.ndarray): Atomic indices [num_nodes].
            - edge_index (np.ndarray): Source and target indices [2, num_edges].
            - edge_dist (np.ndarray): Distances for each edge [num_edges].
    """
    # Get atomic indices based on ATOM_MAP
    symbols = atoms.get_chemical_symbols()
    node_feats = np.array([ATOM_MAP[s] for s in symbols], dtype=np.int64)

    # Compute neighbor list
    # i: center atom indices, j: neighbor atom indices, d: distances
    i, j, d = neighbor_list("ijd", atoms, cutoff)

    # If no edges, return empty arrays
    if len(i) == 0:
        return {
            "node_feats": node_feats,
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_dist": np.empty((0,), dtype=np.float32),
        }

    # Stack into (source, target, dist)
    edges = np.stack((i, j, d), axis=1)

    # Sort by source (primary) and distance (secondary)
    # lexsort sorts by last key passed first, so pass (dist, source)
    sort_indices = np.lexsort((d, i))
    sorted_edges = edges[sort_indices]

    # Filter to keep max_neighbors per node
    # Find unique source indices and their counts
    _, unique_indices, counts = np.unique(
        sorted_edges[:, 0].astype(int), return_index=True, return_counts=True
    )

    # Create a mask to keep only the first max_neighbors for each source node
    mask = np.zeros(len(sorted_edges), dtype=bool)

    current_idx = 0
    for count in counts:
        n_keep = min(count, max_neighbors)
        mask[current_idx : current_idx + n_keep] = True
        current_idx += count

    filtered_edges = sorted_edges[mask]

    edge_index = filtered_edges[:, :2].astype(np.int64).T
    edge_dist = filtered_edges[:, 2].astype(np.float32)

    return {"node_feats": node_feats, "edge_index": edge_index, "edge_dist": edge_dist}


def extract_global_features(atoms):
    """
    Extracts global features (lattice parameters and composition) from the structure.

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        np.ndarray: A 10-dimensional vector containing:
            [a, b, c, alpha, beta, gamma, frac_Al, frac_Ga, frac_In, frac_O]
    """
    # 1. Lattice Parameters (lengths in Angstrom, angles in degrees)
    # cellpar returns [a, b, c, alpha, beta, gamma]
    lattice_params = atoms.get_cell_lengths_and_angles()

    # 2. Composition Fractions
    symbols = atoms.get_chemical_symbols()
    num_atoms = len(symbols)

    # Count each atom type
    counts = {atom_type: 0 for atom_type in ATOM_MAP.keys()}
    for s in symbols:
        if s in counts:
            counts[s] += 1

    # Calculate fractions for Al, Ga, In, O in that order
    fractions = [counts[atom_type] / num_atoms for atom_type in ["Al", "Ga", "In", "O"]]

    # Combine into a single feature vector
    global_feats = np.concatenate([lattice_params, fractions]).astype(np.float32)

    return global_feats


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Loads metadata, processes geometry files into graphs and global features,
    and caches the results.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_path (str): Path to save/load the .npz cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary containing lists of:
            - ids
            - node_feats_list
            - edge_index_list
            - edge_dist_list
            - global_feats_list
            - targets (if available in metadata)
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # When loading object arrays, we need to ensure they are converted back to lists of arrays
            # if that's what the downstream code expects, or keep them as arrays of objects.
            # Here we convert back to lists for consistency with the processing path.
            return {
                "ids": data["ids"],
                "node_feats_list": list(data["node_feats_list"]),
                "edge_index_list": list(data["edge_index_list"]),
                "edge_dist_list": list(data["edge_dist_list"]),
                "global_feats_list": data["global_feats_list"],
                "targets": data["targets"] if "targets" in data else None,
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    ids = []
    node_feats_list = []
    edge_index_list = []
    edge_dist_list = []
    global_feats_list = []
    targets = []

    # Check if targets exist in the dataframe
    has_targets = all(col in df.columns for col in TARGET_COLS)

    for idx, row in df.iterrows():
        # Construct full file path
        # Metadata file_path is relative to INPUT_DIR, e.g., "train/1/geometry.xyz"
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            print(f"Warning: File not found {full_path}. Skipping.")
            continue

        # Load structure
        try:
            atoms = ase.io.read(full_path, format="aims")
        except Exception as e:
            print(f"Error reading {full_path}: {e}. Skipping.")
            continue

        # Graph Construction
        graph_data = get_pbc_graph(atoms)

        # Global Features
        g_feats = extract_global_features(atoms)

        # Collect data
        ids.append(row["id"])
        node_feats_list.append(graph_data["node_feats"])
        edge_index_list.append(graph_data["edge_index"])
        edge_dist_list.append(graph_data["edge_dist"])
        global_feats_list.append(g_feats)

        if has_targets:
            targets.append(row[TARGET_COLS].values.astype(np.float32))

    # Convert lists to numpy arrays for storage
    ids = np.array(ids, dtype=np.int64)
    global_feats_list = np.array(global_feats_list, dtype=np.float32)

    if has_targets:
        targets = np.array(targets, dtype=np.float32)
    else:
        targets = np.empty((len(ids), 0), dtype=np.float32)  # Empty placeholder

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving processed data to {cache_path}...")

    # Create object arrays for variable-length data to save correctly in npz
    nf_arr = np.empty(len(node_feats_list), dtype=object)
    nf_arr[:] = node_feats_list

    ei_arr = np.empty(len(edge_index_list), dtype=object)
    ei_arr[:] = edge_index_list

    ed_arr = np.empty(len(edge_dist_list), dtype=object)
    ed_arr[:] = edge_dist_list

    np.savez(
        cache_path,
        ids=ids,
        node_feats_list=nf_arr,
        edge_index_list=ei_arr,
        edge_dist_list=ed_arr,
        global_feats_list=global_feats_list,
        targets=targets,
    )

    return {
        "ids": ids,
        "node_feats_list": node_feats_list,
        "edge_index_list": edge_index_list,
        "edge_dist_list": edge_dist_list,
        "global_feats_list": global_feats_list,
        "targets": targets if has_targets else None,
    }
