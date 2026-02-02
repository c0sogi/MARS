import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from library.config import Config
from library.utils import get_target_stats
from library.features import RadialBasisFunctions, SphericalBasisFunctions


def read_xyz(path):
    """
    Parses a standard .xyz molecular structure file.

    Args:
        path (str): Full path to the .xyz file.

    Returns:
        tuple: (atom_types [LongTensor], positions [FloatTensor])
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # First line is number of atoms
    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        # Fallback for empty or malformed files
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float)

    # Skip comment line (line 1), read atoms (lines 2 to 2+num_atoms)
    atom_lines = lines[2 : 2 + num_atoms]

    atom_types = []
    positions = []

    for line in atom_lines:
        parts = line.split()
        if not parts:
            continue

        symbol = parts[0]
        # Parse coordinates
        coords = [float(x) for x in parts[1:4]]

        # Map symbol to integer
        if symbol in Config.ATOM_TYPES:
            atom_types.append(Config.ATOM_TYPES[symbol])
        else:
            # Handle unknown atoms (though unlikely in this dataset)
            atom_types.append(0)

        positions.append(coords)

    return torch.tensor(atom_types, dtype=torch.long), torch.tensor(
        positions, dtype=torch.float
    )


class CHAMPSDataset(InMemoryDataset):
    """
    PyG Dataset wrapper for CHAMPS molecular graphs.
    Handles loading from a pre-processed list of Data objects or a cached file.
    """

    def __init__(self, root, data_list=None, cache_path=None):
        super().__init__(root)

        if data_list is not None:
            # Collate the list of Data objects into the internal storage format
            self.data, self.slices = self.collate(data_list)

            # Save to cache if path provided
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save((self.data, self.slices), cache_path)

        elif cache_path is not None and os.path.exists(cache_path):
            # Load from cache
            self.data, self.slices = torch.load(cache_path)
        else:
            # This state implies initialization without data, which shouldn't happen
            # in the factory pattern used below.
            pass


def process_molecule(
    molecule_name, structure_rel_path, group_df, rbf_module, sbf_module, stats, is_test
):
    """
    Processes a single molecule into a PyG Data object.

    Args:
        molecule_name (str): ID of the molecule.
        structure_rel_path (str): Path to XYZ file relative to input dir.
        group_df (pd.DataFrame): Metadata rows corresponding to this molecule.
        rbf_module (nn.Module): Pre-initialized RBF module.
        sbf_module (nn.Module): Pre-initialized SBF module.
        stats (dict): Target statistics for normalization.
        is_test (bool): Whether this is test data (no targets).

    Returns:
        Data: PyG Data object containing graph and targets.
    """
    # 1. Load Structure
    full_path = os.path.join(Config.INPUT_DIR, structure_rel_path)
    z, pos = read_xyz(full_path)

    if z.size(0) == 0:
        return None

    # 2. Build Graph (Radius Graph)
    # Compute pairwise distances
    dist_matrix = torch.cdist(pos, pos)  # (N, N)

    # Create edge_index based on cutoff
    # Exclude self-loops (dist > 1e-6) and apply cutoff
    mask = (dist_matrix < Config.CUTOFF) & (dist_matrix > 1e-6)
    src, dst = torch.where(mask)
    edge_index = torch.stack([src, dst], dim=0)  # (2, E)

    # 3. Compute Edge Features
    # Vector difference: pos[dst] - pos[src]
    vec = pos[edge_index[1]] - pos[edge_index[0]]
    d = dist_matrix[edge_index[0], edge_index[1]]

    # RBF Expansion (Pre-computed)
    rbf_attr = rbf_module(d)

    # 4. Compute Triplet Features (for Directional Message Passing)
    # Identify triplets k -> j -> i
    # We find pairs of edges (e1, e2) where e1.dst == e2.src and e1.src != e2.dst

    # Broadcast to find matches
    dst_e1 = edge_index[1].unsqueeze(1)  # (E, 1)
    src_e2 = edge_index[0].unsqueeze(0)  # (1, E)

    # Filter for valid connections (j matches) and no backtracking (k != i)
    src_e1 = edge_index[0].unsqueeze(1)
    dst_e2 = edge_index[1].unsqueeze(0)

    match = (dst_e1 == src_e2) & (src_e1 != dst_e2)
    idx_kj, idx_ji = torch.where(match)

    triplet_index = torch.stack([idx_kj, idx_ji], dim=0)  # (2, T)

    # Compute Angles for SBF
    # Vectors for edges k->j and j->i
    vec_kj = vec[idx_kj]
    vec_ji = vec[idx_ji]

    # Angle is between bond j-k and bond j-i
    # vec_jk = -vec_kj
    vec_jk = -vec_kj

    # Cosine similarity
    norm_jk = torch.norm(vec_jk, dim=1)
    norm_ji = torch.norm(vec_ji, dim=1)

    dot = (vec_jk * vec_ji).sum(dim=1)
    cos_angle = dot / (norm_jk * norm_ji + 1e-7)
    # Clamp for numerical stability
    cos_angle = torch.clamp(cos_angle, -0.99999, 0.99999)
    angle = torch.acos(cos_angle)

    # SBF Expansion (Pre-computed)
    # Uses distance of incoming edge (d_kj) and angle
    d_kj = d[idx_kj]
    sbf_attr = sbf_module(d_kj, angle)

    # 5. Process Targets
    target_ids = group_df["id"].values
    target_indices = group_df[["atom_index_0", "atom_index_1"]].values
    target_types_str = group_df["type"].values

    # Map atom indices to graph node indices (identity mapping here)
    target_edge_index = torch.tensor(target_indices.T, dtype=torch.long)

    # Encode types
    type_indices = [Config.COUPLING_TYPES.index(t) for t in target_types_str]
    target_type = torch.tensor(type_indices, dtype=torch.long)
    ids = torch.tensor(target_ids, dtype=torch.long)

    if is_test:
        # No target values
        target_val = torch.zeros(len(target_ids), dtype=torch.float)
    else:
        # Normalize target values
        target_values_raw = group_df["scalar_coupling_constant"].values
        norm_values = []

        for t_idx, val in zip(type_indices, target_values_raw):
            m, s = stats[t_idx]
            norm_values.append((val - m) / s)

        target_val = torch.tensor(norm_values, dtype=torch.float)

    # Construct Data Object
    data = Data(
        x=z,
        pos=pos,
        edge_index=edge_index,
        edge_attr=rbf_attr,
        edge_vec=vec,
        triplet_index=triplet_index,
        triplet_attr=sbf_attr,
        target_edge_index=target_edge_index,
        target_type=target_type,
        target_val=target_val,
        id=ids,
        molecule_name=molecule_name,
    )

    return data


def get_molecular_data(split="train", load_cached_data=True):
    """
    Main entry point to get the dataset.
    Handles caching, metadata loading, and parallel processing.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from disk.

    Returns:
        CHAMPSDataset: The requested dataset.
    """
    # 1. Determine Paths and Config
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.CACHE_TRAIN_PATH
        is_test = False
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        cache_path = Config.CACHE_VAL_PATH
        is_test = False
    else:  # test
        meta_path = Config.TEST_METADATA_PATH
        cache_path = Config.CACHE_TEST_PATH
        is_test = True

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        return CHAMPSDataset(Config.WORKING_DIR, cache_path=cache_path)

    print(f"Processing {split} data from scratch...")

    # 3. Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    # 4. Handle Statistics (for normalization)
    stats = {}
    if not is_test:
        # If processing train, compute/load stats from this df
        if split == "train":
            stats = get_target_stats(df, load_cached_data=True)
        else:
            # If processing val, try to load cached stats.
            # If missing, we must load train metadata to compute them.
            stats = get_target_stats(None, load_cached_data=True)
            if not stats:
                print(
                    "Stats cache missing for validation processing. Loading train metadata..."
                )
                train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
                stats = get_target_stats(train_df, load_cached_data=False)
                del train_df

    # 5. Debugging Subset
    if Config.DEBUG:
        print(f"DEBUG MODE: Reducing dataset to {Config.DEBUG_SUBSET_SIZE} molecules.")
        molecules = df["molecule_name"].unique()[: Config.DEBUG_SUBSET_SIZE]
        df = df[df["molecule_name"].isin(molecules)]

    # 6. Initialize Feature Modules (CPU)
    # We use these to pre-compute features
    rbf = RadialBasisFunctions(Config.CUTOFF, Config.NUM_RBF)
    sbf = SphericalBasisFunctions(Config.CUTOFF, Config.NUM_RBF, Config.NUM_SBF)

    # 7. Process Molecules
    grouped = df.groupby("molecule_name")
    data_list = []

    for name, group in tqdm(grouped, desc=f"Processing {split}"):
        struct_path = group.iloc[0]["structure_path"]

        data = process_molecule(name, struct_path, group, rbf, sbf, stats, is_test)

        if data is not None:
            data_list.append(data)

    # 8. Create and Save Dataset
    dataset = CHAMPSDataset(
        Config.WORKING_DIR, data_list=data_list, cache_path=cache_path
    )
    return dataset
