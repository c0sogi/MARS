import os
import numpy as np
import pandas as pd
import torch
import ase.io
from ase.neighborlist import neighbor_list
from library.config import Config


class GaussianRBF(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, n_rbf=60, sigma=1.5):
        super().__init__()
        self.centers = torch.linspace(start, stop, n_rbf)
        self.sigma = torch.tensor(sigma)

    def forward(self, distances):
        # distances: shape (..., 1) or (...)
        # centers: shape (n_rbf,)
        if distances.dim() == 1:
            distances = distances.unsqueeze(-1)
        # Expand dimensions for broadcasting: (..., 1) - (n_rbf,) -> (..., n_rbf)
        return torch.exp(
            -((distances - self.centers.to(distances.device)) ** 2)
            / (2 * self.sigma**2)
        )


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        if isinstance(data, torch.Tensor):
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
            # Handle zero std to avoid division by zero
            self.std[self.std == 0] = 1.0
        else:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            # Handle zero std
            self.std[self.std == 0] = 1.0

    def transform(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")

        if isinstance(data, torch.Tensor):
            return (data - self.mean.to(data.device)) / self.std.to(data.device)
        else:
            return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")

        if isinstance(data, torch.Tensor):
            return data * self.std.to(data.device) + self.mean.to(data.device)
        else:
            return data * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def read_geometry(file_path):
    """
    Reads an XYZ file using ASE.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found: {full_path}")

    # ASE read usually handles 'xyz' format automatically
    atoms = ase.io.read(full_path, format="xyz")
    return atoms


def get_global_features(atoms):
    """
    Extracts global features: Lattice parameters (6) + Composition (4).
    Total dimension: 10.
    """
    # 1. Lattice parameters: a, b, c, alpha, beta, gamma
    # cellpar() returns [a, b, c, alpha, beta, gamma]
    # Lengths in Angstrom, angles in degrees
    lattice_params = atoms.cell.cellpar()

    # 2. Composition fractions
    # Elements of interest: Al, Ga, In, O
    # Atomic numbers: Al(13), Ga(31), In(49), O(8)
    atomic_numbers = atoms.get_atomic_numbers()
    total_atoms = len(atomic_numbers)

    # Count occurrences
    n_Al = np.sum(atomic_numbers == 13)
    n_Ga = np.sum(atomic_numbers == 31)
    n_In = np.sum(atomic_numbers == 49)
    n_O = np.sum(atomic_numbers == 8)

    frac_Al = n_Al / total_atoms
    frac_Ga = n_Ga / total_atoms
    frac_In = n_In / total_atoms
    frac_O = n_O / total_atoms

    composition = np.array([frac_Al, frac_Ga, frac_In, frac_O])

    # Combine
    global_feats = np.concatenate([lattice_params, composition])
    return global_feats.astype(np.float32)


def build_pbc_graph(atoms, cutoff=5.0, max_neighbors=50):
    """
    Constructs a graph representation of the crystal structure respecting PBC.
    """
    # Get neighbor list
    # i: index of center atom
    # j: index of neighbor atom
    # d: distance
    i_indices, j_indices, distances = neighbor_list("ijd", atoms, cutoff)

    # If no neighbors found
    if len(i_indices) == 0:
        return (
            atoms.get_atomic_numbers().astype(np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

    # Enforce max neighbors per node
    # Stack into a matrix: [i, j, d]
    edges = np.column_stack((i_indices, j_indices, distances))

    # Sort by i, then by d
    # lexsort sorts by last key first, so d then i
    sort_indices = np.lexsort((edges[:, 2], edges[:, 0]))
    edges_sorted = edges[sort_indices]

    # Filter to keep top K neighbors using pandas for convenience
    df_edges = pd.DataFrame(edges_sorted, columns=["i", "j", "d"])
    df_edges = df_edges.groupby("i").head(max_neighbors)

    src_indices = df_edges["i"].values.astype(np.int64)
    dst_indices = df_edges["j"].values.astype(np.int64)
    edge_dists = df_edges["d"].values.astype(np.float32)

    atom_numbers = atoms.get_atomic_numbers().astype(np.int64)

    return atom_numbers, src_indices, dst_indices, edge_dists


def process_dataset(metadata_df, load_cached_data=True, cache_prefix="train"):
    """
    Processes the dataset described by metadata_df.
    Handles caching of processed graphs and global features.

    Args:
        metadata_df: DataFrame containing 'file_path' and optionally targets.
        load_cached_data: Boolean, whether to load from cache.
        cache_prefix: String, prefix for cache files (e.g., 'train', 'val', 'test').

    Returns:
        A dictionary containing lists of features and numpy arrays for globals/targets.
    """

    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_graphs.npz")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)

            # Check if targets exist in cache (might be None for test set)
            targets = data["targets"] if "targets" in data else None
            # If targets was saved as None (0-d array with None), handle it
            if targets is not None and targets.shape == ():
                targets = None

            return {
                "atom_features_list": list(data["atom_features_list"]),
                "edge_index_list": list(data["edge_index_list"]),
                "edge_attr_list": list(data["edge_attr_list"]),
                "global_features": data["global_features"],
                "targets": targets,
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {len(metadata_df)} samples for {cache_prefix}...")

    atom_features_list = []
    edge_index_list = []
    edge_attr_list = []
    global_features_list = []
    targets_list = []
    ids_list = []

    # Optional sampling for debug
    if Config.SAMPLE_SIZE and Config.SAMPLE_SIZE < len(metadata_df):
        print(f"Subsampling {Config.SAMPLE_SIZE} records for debugging.")
        metadata_df = metadata_df.iloc[: Config.SAMPLE_SIZE]

    for idx, row in metadata_df.iterrows():
        file_path = row["file_path"]
        material_id = row["id"]

        try:
            # Read Geometry
            atoms = read_geometry(file_path)

            # Build Graph
            z, src, dst, dists = build_pbc_graph(
                atoms, cutoff=Config.CUTOFF_RADIUS, max_neighbors=Config.MAX_NEIGHBORS
            )

            # Global Features
            glob_feat = get_global_features(atoms)

            # Store
            atom_features_list.append(z)
            # Edge index: stack src and dst
            edge_index = np.vstack((src, dst))
            edge_index_list.append(edge_index)
            edge_attr_list.append(dists)
            global_features_list.append(glob_feat)
            ids_list.append(material_id)

            # Targets (if available)
            if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
                targets_list.append(
                    [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
                )

        except Exception as e:
            print(f"Error processing ID {material_id}: {e}")
            continue

    # Convert to numpy arrays where appropriate
    global_features = np.stack(global_features_list).astype(np.float32)
    targets = np.array(targets_list).astype(np.float32) if targets_list else None
    ids = np.array(ids_list)

    # 3. Save to Cache
    save_dict = {
        "atom_features_list": np.array(atom_features_list, dtype=object),
        "edge_index_list": np.array(edge_index_list, dtype=object),
        "edge_attr_list": np.array(edge_attr_list, dtype=object),
        "global_features": global_features,
        "ids": ids,
    }
    if targets is not None:
        save_dict["targets"] = targets

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez(cache_path, **save_dict)
    print(f"Saved processed data to {cache_path}")

    return {
        "atom_features_list": atom_features_list,
        "edge_index_list": edge_index_list,
        "edge_attr_list": edge_attr_list,
        "global_features": global_features,
        "targets": targets,
        "ids": ids,
    }
