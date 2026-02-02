import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from library.config import Config
from library.utils import set_seed


def get_atomic_number(symbol):
    """
    Maps element symbol to atomic number.
    """
    atom_map = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    return atom_map.get(symbol, 0)


def process_xyz(file_path):
    """
    Parses an XYZ file.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        atoms (np.ndarray): Array of atomic numbers (N,).
        coords (np.ndarray): Array of coordinates (N, 3).
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # First line is number of atoms
    try:
        # Some files might have whitespace
        if not lines:
            return None, None
        num_atoms = int(lines[0].strip())
    except ValueError:
        return None, None

    atoms = []
    coords = []

    # XYZ format:
    # Line 1: Number of atoms
    # Line 2: Comment/Blank
    # Line 3+: Symbol X Y Z

    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        symbol = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
            atoms.append(get_atomic_number(symbol))
            coords.append([x, y, z])
        except ValueError:
            continue

    return np.array(atoms, dtype=np.int32), np.array(coords, dtype=np.float32)


def build_graph(
    atoms, coords, cutoff=Config.CUTOFF_RADIUS, max_neighbors=Config.MAX_NEIGHBORS
):
    """
    Constructs a geometric graph from atoms and coordinates.

    Args:
        atoms (np.ndarray): Atomic numbers.
        coords (np.ndarray): Coordinates.
        cutoff (float): Radius for edge creation.
        max_neighbors (int): Maximum neighbors per atom.

    Returns:
        dict: Contains 'x', 'pos', 'edge_index', 'edge_attr'.
    """
    num_atoms = len(atoms)

    # Use KDTree for efficient neighbor search
    tree = cKDTree(coords)

    # Query neighbors
    # k = max_neighbors + 1 because the point itself is included in the result
    dists, indices = tree.query(
        coords, k=min(num_atoms, max_neighbors + 1), distance_upper_bound=cutoff
    )

    edge_sources = []
    edge_targets = []

    for i in range(num_atoms):
        # indices[i] are neighbors of atom i

        # Handle case where k=1 (dists is scalar or 1D array depending on version)
        # cKDTree query returns arrays of shape (N, k)

        row_indices = indices[i]
        row_dists = dists[i]

        if not isinstance(row_indices, (list, np.ndarray)):
            row_indices = [row_indices]
            row_dists = [row_dists]

        for k_idx, neighbor_idx in enumerate(row_indices):
            # cKDTree returns num_atoms (or inf) for missing neighbors if k > actual neighbors
            if neighbor_idx == num_atoms:
                continue
            if neighbor_idx == i:  # Skip self-loop
                continue

            d = row_dists[k_idx]
            if d > cutoff:  # strictly enforce cutoff (cKDTree upper_bound is <=)
                continue

            edge_sources.append(i)
            edge_targets.append(neighbor_idx)

    edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)

    # Compute edge attributes: Vector difference (Target - Source)
    if len(edge_sources) > 0:
        vecs = coords[edge_targets] - coords[edge_sources]
    else:
        # Handle isolated atoms or empty graph (unlikely in molecules but possible in cuts)
        vecs = np.zeros((0, 3), dtype=np.float32)
        # Ensure edge_index is correct shape even if empty
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return {
        "x": atoms,
        "pos": coords,
        "edge_index": edge_index,
        "edge_attr": vecs,  # Vectors (dx, dy, dz)
    }


def save_graphs_to_npz(graphs_dict, path):
    """
    Flattens the dictionary of graphs and saves to .npz to avoid pickle.
    """
    mol_names = sorted(list(graphs_dict.keys()))

    # Arrays to store concatenated data
    all_x = []
    all_pos = []
    all_edge_indices = []
    all_edge_attrs = []

    # Metadata to reconstruct
    meta_mol_names = []
    meta_num_atoms = []
    meta_num_edges = []

    for name in mol_names:
        g = graphs_dict[name]

        meta_mol_names.append(name)
        meta_num_atoms.append(len(g["x"]))
        meta_num_edges.append(g["edge_index"].shape[1])

        all_x.append(g["x"])
        all_pos.append(g["pos"])
        all_edge_indices.append(g["edge_index"])
        all_edge_attrs.append(g["edge_attr"])

    # Concatenate
    if len(all_x) > 0:
        flat_x = np.concatenate(all_x)
        flat_pos = np.concatenate(all_pos)
        flat_edge_index = np.concatenate(all_edge_indices, axis=1)
        flat_edge_attr = np.concatenate(all_edge_attrs)
    else:
        flat_x = np.array([], dtype=np.int32)
        flat_pos = np.array([], dtype=np.float32)
        flat_edge_index = np.array([[], []], dtype=np.int64)
        flat_edge_attr = np.array([], dtype=np.float32)

    np.savez_compressed(
        path,
        mol_names=np.array(meta_mol_names),
        num_atoms=np.array(meta_num_atoms, dtype=np.int32),
        num_edges=np.array(meta_num_edges, dtype=np.int32),
        flat_x=flat_x,
        flat_pos=flat_pos,
        flat_edge_index=flat_edge_index,
        flat_edge_attr=flat_edge_attr,
    )


def load_graphs_from_npz(path):
    """
    Loads graphs from .npz and reconstructs the dictionary.
    """
    data = np.load(path)

    mol_names = data["mol_names"]
    num_atoms = data["num_atoms"]
    num_edges = data["num_edges"]
    flat_x = data["flat_x"]
    flat_pos = data["flat_pos"]
    flat_edge_index = data["flat_edge_index"]
    flat_edge_attr = data["flat_edge_attr"]

    graphs_dict = {}

    atom_offset = 0
    edge_offset = 0

    for i, name in enumerate(mol_names):
        n_a = num_atoms[i]
        n_e = num_edges[i]

        x = flat_x[atom_offset : atom_offset + n_a]
        pos = flat_pos[atom_offset : atom_offset + n_a]

        # edge_index is (2, TotalEdges), we slice columns
        edge_index = flat_edge_index[:, edge_offset : edge_offset + n_e]
        edge_attr = flat_edge_attr[edge_offset : edge_offset + n_e]

        graphs_dict[name] = {
            "x": x,
            "pos": pos,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }

        atom_offset += n_a
        edge_offset += n_e

    return graphs_dict


def process_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Main function to process dataset.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_path (str): Path where the .npz cache should be stored/loaded.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary mapping molecule_name to graph data.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached graphs from {cache_path}...")
        try:
            return load_graphs_from_npz(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    if "structure_path" not in df.columns:
        df["structure_path"] = "structures/" + df["molecule_name"] + ".xyz"

    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling {Config.DEBUG_SIZE} rows...")
        df = df.iloc[: Config.DEBUG_SIZE]

    # Identify unique molecules
    unique_molecules = df["molecule_name"].unique()

    # Map molecule name to structure path
    # We drop duplicates to get unique molecule entries
    mol_df = df.drop_duplicates("molecule_name")
    mol_to_path = dict(zip(mol_df["molecule_name"], mol_df["structure_path"]))

    graphs_dict = {}

    print(f"Building graphs for {len(unique_molecules)} molecules...")

    count = 0
    for mol_name in unique_molecules:
        rel_path = mol_to_path[mol_name]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            # Fallback check if path is just the filename
            if os.path.exists(
                os.path.join(Config.STRUCTURES_DIR, os.path.basename(rel_path))
            ):
                full_path = os.path.join(
                    Config.STRUCTURES_DIR, os.path.basename(rel_path)
                )
            else:
                print(f"Warning: Structure file {full_path} not found. Skipping.")
                continue

        atoms, coords = process_xyz(full_path)
        if atoms is None:
            continue

        graph = build_graph(atoms, coords)
        graphs_dict[mol_name] = graph
        count += 1

    print(f"Successfully processed {count} molecules.")
    print(f"Saving processed graphs to {cache_path}...")
    save_graphs_to_npz(graphs_dict, cache_path)

    return graphs_dict
