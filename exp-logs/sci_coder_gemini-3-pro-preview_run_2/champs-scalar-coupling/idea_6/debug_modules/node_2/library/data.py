import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist, pdist, squareform
from library.config import Config
from library.utils import Standardizer, set_seed
from library.features import radial_basis_functions, spherical_basis_functions

# Global Constants
ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
COUPLING_TYPES = ["1JHC", "2JHC", "3JHC", "1JHN", "2JHN", "3JHN", "2JHH", "3JHH"]
TYPE_TO_ID = {t: i for i, t in enumerate(COUPLING_TYPES)}


class MolecularGraphDataset(Dataset):
    """
    PyTorch Dataset for Molecular Graphs.
    Stores data in flattened arrays and reconstructs individual graphs on demand.
    """

    def __init__(self, data_dict, standardizer=None):
        self.n_atoms = data_dict["n_atoms"]
        self.n_edges = data_dict["n_edges"]
        self.n_triplets = data_dict["n_triplets"]
        self.n_couplings = data_dict["n_couplings"]

        # Cumulative indices for efficient slicing
        self.cum_atoms = np.concatenate(([0], np.cumsum(self.n_atoms)))
        self.cum_edges = np.concatenate(([0], np.cumsum(self.n_edges)))
        self.cum_triplets = np.concatenate(([0], np.cumsum(self.n_triplets)))
        self.cum_couplings = np.concatenate(([0], np.cumsum(self.n_couplings)))

        # Feature Arrays
        self.atoms = data_dict["atoms"]
        self.edge_index = data_dict["edge_index"]
        self.edge_attr = data_dict["edge_attr"]
        self.triplet_index = data_dict["triplet_index"]
        self.triplet_attr = data_dict["triplet_attr"]

        # Target Arrays
        self.coupling_indices = data_dict["coupling_indices"]
        self.coupling_types = data_dict["coupling_types"]
        self.coupling_values = data_dict["coupling_values"]
        self.molecule_names = data_dict["molecule_names"]

        self.standardizer = standardizer
        self.num_molecules = len(self.n_atoms)

        # Pre-fetch standardizer stats for speed if available
        self.std_means = None
        self.std_stds = None
        if self.standardizer and self.standardizer.fitted:
            # Create lookup arrays for types 0..7
            self.std_means = np.zeros(len(COUPLING_TYPES))
            self.std_stds = np.ones(len(COUPLING_TYPES))
            for t_str, t_id in TYPE_TO_ID.items():
                self.std_means[t_id] = self.standardizer.means.get(t_str, 0.0)
                self.std_stds[t_id] = self.standardizer.stds.get(t_str, 1.0)

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        # Slice Atoms
        a_start, a_end = self.cum_atoms[idx], self.cum_atoms[idx + 1]
        x = torch.tensor(self.atoms[a_start:a_end], dtype=torch.long)

        # Slice Edges
        e_start, e_end = self.cum_edges[idx], self.cum_edges[idx + 1]
        edge_index = torch.tensor(
            self.edge_index[e_start:e_end], dtype=torch.long
        ).t()  # [2, E]
        edge_attr = torch.tensor(self.edge_attr[e_start:e_end], dtype=torch.float)

        # Slice Triplets
        t_start, t_end = self.cum_triplets[idx], self.cum_triplets[idx + 1]
        triplet_index = torch.tensor(
            self.triplet_index[t_start:t_end], dtype=torch.long
        ).t()  # [3, T]
        triplet_attr = torch.tensor(self.triplet_attr[t_start:t_end], dtype=torch.float)

        # Slice Couplings
        c_start, c_end = self.cum_couplings[idx], self.cum_couplings[idx + 1]
        coup_index = torch.tensor(
            self.coupling_indices[c_start:c_end], dtype=torch.long
        )  # [C, 2]
        coup_type = torch.tensor(
            self.coupling_types[c_start:c_end], dtype=torch.long
        )  # [C]
        raw_y = self.coupling_values[c_start:c_end]

        # Standardize Targets
        if self.std_means is not None:
            # Vectorized lookup
            means = self.std_means[coup_type.numpy()]
            stds = self.std_stds[coup_type.numpy()]
            y = (raw_y - means) / stds
        else:
            y = raw_y

        y = torch.tensor(y, dtype=torch.float)

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "triplet_index": triplet_index,
            "triplet_attr": triplet_attr,
            "coupling_index": coup_index,
            "coupling_type": coup_type,
            "y": y,
            "molecule_name": self.molecule_names[idx],
            "num_atoms": x.size(0),
        }


def collate_fn(batch):
    """
    Batches a list of molecular graphs into a single large graph.
    """
    batch_x = []
    batch_batch = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_triplet_index = []
    batch_triplet_attr = []
    batch_coupling_index = []
    batch_coupling_type = []
    batch_y = []
    batch_mol_names = []

    cum_atoms = 0

    for i, data in enumerate(batch):
        num_atoms = data["num_atoms"]

        # Nodes
        batch_x.append(data["x"])
        batch_batch.append(torch.full((num_atoms,), i, dtype=torch.long))

        # Edges (shift indices)
        batch_edge_index.append(data["edge_index"] + cum_atoms)
        batch_edge_attr.append(data["edge_attr"])

        # Triplets (shift indices)
        batch_triplet_index.append(data["triplet_index"] + cum_atoms)
        batch_triplet_attr.append(data["triplet_attr"])

        # Couplings (shift indices)
        batch_coupling_index.append(data["coupling_index"] + cum_atoms)
        batch_coupling_type.append(data["coupling_type"])
        batch_y.append(data["y"])

        batch_mol_names.append(data["molecule_name"])

        cum_atoms += num_atoms

    return {
        "x": torch.cat(batch_x),
        "batch": torch.cat(batch_batch),
        "edge_index": torch.cat(batch_edge_index, dim=1),
        "edge_attr": torch.cat(batch_edge_attr),
        "triplet_index": torch.cat(batch_triplet_index, dim=1),
        "triplet_attr": torch.cat(batch_triplet_attr),
        "coupling_index": torch.cat(batch_coupling_index),
        "coupling_type": torch.cat(batch_coupling_type),
        "y": torch.cat(batch_y),
        "molecule_names": batch_mol_names,
    }


def process_and_cache_data(mode, config, load_cached_data=True):
    """
    Loads raw data, processes it into graphs, and caches/loads from disk.
    """
    # Determine paths
    if mode == "train":
        meta_path = config.TRAIN_METADATA
        cache_path = config.TRAIN_CACHE
    elif mode == "val":
        meta_path = config.VAL_METADATA
        cache_path = config.VAL_CACHE
    else:
        meta_path = config.TEST_METADATA
        cache_path = config.TEST_CACHE

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            # Allow pickle=True is needed for string arrays in npz usually,
            # but we will try to stick to basic types.
            # np.load with allow_pickle=True is standard for npz containing strings.
            data = np.load(cache_path, allow_pickle=True)
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {mode} data from scratch...")

    # Load Metadata
    df_meta = pd.read_csv(meta_path)
    if config.DEBUG:
        df_meta = df_meta.iloc[:1000]  # Small subset for debug

    # Load Structures
    print("Loading structures...")
    df_struct = pd.read_csv(config.STRUCTURES_CSV)

    # Filter structures to only those in metadata
    relevant_mols = df_meta["molecule_name"].unique()
    df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)]

    # Group structures by molecule for fast access
    # Create dict: molecule_name -> (atom_types_array, coords_array)
    struct_dict = {}
    print("Grouping structures...")
    # Optimization: iterate groupby object
    for name, group in df_struct.groupby("molecule_name"):
        atoms = group["atom"].map(ATOM_MAP).values.astype(np.int8)
        coords = group[["x", "y", "z"]].values.astype(np.float32)
        struct_dict[name] = (atoms, coords)

    # Group metadata by molecule
    meta_grouped = df_meta.groupby("molecule_name")

    # Lists to collect data
    l_n_atoms = []
    l_n_edges = []
    l_n_triplets = []
    l_n_couplings = []

    l_atoms = []
    l_edge_index = []
    l_edge_attr = []
    l_triplet_index = []
    l_triplet_attr = []

    l_coup_indices = []
    l_coup_types = []
    l_coup_values = []
    l_mol_names = []

    print(f"Building graphs for {len(relevant_mols)} molecules...")

    # Iterate over molecules
    for mol_name in relevant_mols:
        if mol_name not in struct_dict:
            continue

        # Get Structure
        atoms, coords = struct_dict[mol_name]
        n_atoms = len(atoms)

        # Get Couplings
        if mol_name in meta_grouped.groups:
            group = meta_grouped.get_group(mol_name)
            c_idx = group[["atom_index_0", "atom_index_1"]].values.astype(np.int64)
            c_types = group["type"].map(TYPE_TO_ID).values.astype(np.int64)
            if "scalar_coupling_constant" in group.columns:
                c_vals = group["scalar_coupling_constant"].values.astype(np.float32)
            else:
                c_vals = np.zeros(len(group), dtype=np.float32)
        else:
            # Should not happen given logic above
            continue

        # --- Geometric Processing ---

        # 1. Distances & Edges
        dist_mat = cdist(coords, coords)
        # Mask: d < cutoff, d > 0 (no self loop)
        mask = (dist_mat < config.CUTOFF) & (dist_mat > 1e-6)
        src, dst = np.where(mask)

        # Edge Features
        dists = dist_mat[src, dst]
        # Compute RBF
        rbf = radial_basis_functions(
            torch.tensor(dists), start=0.0, end=config.CUTOFF, num_basis=config.RBF_SIZE
        ).numpy()

        # 2. Triplets (Angular)
        # We need triplets (j, i, k) where i is center, j, k are neighbors, j != k
        # Adjacency list
        adj = {i: [] for i in range(n_atoms)}
        for s, d in zip(src, dst):
            adj[s].append(d)

        t_indices = []
        t_dists = []
        t_angles = []

        for i in range(n_atoms):
            neighbors = adj[i]
            if len(neighbors) < 2:
                continue

            # Iterate pairs
            # Optimization: use numpy for vectorization if neighbors list is large?
            # Atoms usually < 30, neighbors < 10. Nested loop is fine.
            for j in neighbors:
                for k in neighbors:
                    if j == k:
                        continue

                    vec_ij = coords[j] - coords[i]
                    vec_ik = coords[k] - coords[i]

                    d_ij = np.linalg.norm(vec_ij)
                    d_ik = np.linalg.norm(vec_ik)

                    # Cosine
                    cos_theta = np.dot(vec_ij, vec_ik) / (d_ij * d_ik + 1e-8)
                    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                    t_indices.append([j, i, k])
                    t_dists.append(d_ij)  # Distance associated with incoming edge j->i
                    t_angles.append(theta)

        if len(t_indices) > 0:
            t_indices = np.array(t_indices, dtype=np.int64)
            t_dists_t = torch.tensor(t_dists, dtype=torch.float)
            t_angles_t = torch.tensor(t_angles, dtype=torch.float)

            # Compute SBF
            sbf = spherical_basis_functions(
                t_dists_t,
                t_angles_t,
                start=0.0,
                end=config.CUTOFF,
                num_basis=config.SBF_SIZE,
            ).numpy()
        else:
            t_indices = np.empty((0, 3), dtype=np.int64)
            sbf = np.empty((0, config.SBF_SIZE), dtype=np.float32)

        # --- Collect Data ---
        l_n_atoms.append(n_atoms)
        l_n_edges.append(len(src))
        l_n_triplets.append(len(t_indices))
        l_n_couplings.append(len(c_vals))

        l_atoms.append(atoms)
        l_edge_index.append(np.stack([src, dst], axis=1))
        l_edge_attr.append(rbf)
        l_triplet_index.append(t_indices)
        l_triplet_attr.append(sbf)

        l_coup_indices.append(c_idx)
        l_coup_types.append(c_types)
        l_coup_values.append(c_vals)
        l_mol_names.append(mol_name)

    # Concatenate
    print("Concatenating arrays...")
    if len(l_n_atoms) == 0:
        print(
            f"Warning: No valid graphs generated for mode {mode}. Returning empty dataset."
        )
        data_dict = {
            "n_atoms": np.array([], dtype=np.int64),
            "n_edges": np.array([], dtype=np.int64),
            "n_triplets": np.array([], dtype=np.int64),
            "n_couplings": np.array([], dtype=np.int64),
            "atoms": np.array([], dtype=np.int64),
            "edge_index": np.empty((0, 2), dtype=np.int64),
            "edge_attr": np.empty((0, config.RBF_SIZE), dtype=np.float32),
            "triplet_index": np.empty((0, 3), dtype=np.int64),
            "triplet_attr": np.empty((0, config.SBF_SIZE), dtype=np.float32),
            "coupling_indices": np.empty((0, 2), dtype=np.int64),
            "coupling_types": np.array([], dtype=np.int64),
            "coupling_values": np.array([], dtype=np.float32),
            "molecule_names": np.array([], dtype=object),
        }
    else:
        data_dict = {
            "n_atoms": np.array(l_n_atoms, dtype=np.int64),
            "n_edges": np.array(l_n_edges, dtype=np.int64),
            "n_triplets": np.array(l_n_triplets, dtype=np.int64),
            "n_couplings": np.array(l_n_couplings, dtype=np.int64),
            "atoms": np.concatenate(l_atoms).astype(np.int64),
            "edge_index": np.concatenate(l_edge_index).astype(np.int64),
            "edge_attr": np.concatenate(l_edge_attr).astype(np.float32),
            "triplet_index": (
                np.concatenate(l_triplet_index).astype(np.int64)
                if len(l_triplet_index) > 0 and l_triplet_index[0].size > 0
                else np.empty((0, 3), dtype=np.int64)
            ),
            "triplet_attr": (
                np.concatenate(l_triplet_attr).astype(np.float32)
                if len(l_triplet_attr) > 0 and l_triplet_attr[0].size > 0
                else np.empty((0, config.SBF_SIZE), dtype=np.float32)
            ),
            "coupling_indices": np.concatenate(l_coup_indices).astype(np.int64),
            "coupling_types": np.concatenate(l_coup_types).astype(np.int64),
            "coupling_values": np.concatenate(l_coup_values).astype(np.float32),
            "molecule_names": np.array(l_mol_names),
        }

    # Save Cache
    print(f"Saving cache to {cache_path}...")
    np.savez_compressed(cache_path, **data_dict)

    return data_dict


def get_dataloaders(config, load_cached_data=True):
    """
    Main entry point. Loads data, fits standardizer, returns dataloaders.
    """
    set_seed(config.SEED)

    # 1. Load Train Data (to fit standardizer)
    train_data = process_and_cache_data("train", config, load_cached_data)

    # 2. Fit Standardizer
    print("Fitting Standardizer...")
    # We need to reconstruct a DataFrame-like structure for the Standardizer
    # or just manually compute stats. The Standardizer expects a DF with 'type' and 'scalar_coupling_constant'
    # Let's create a temporary DF for fitting.
    train_types = train_data["coupling_types"]
    train_values = train_data["coupling_values"]

    # Inverse map types for Standardizer compatibility (it expects strings)
    # Actually, we can just use the integer types if we modify Standardizer,
    # but Standardizer is provided and immutable.
    # So we map ints back to strings.
    type_strings = [COUPLING_TYPES[t] for t in train_types]

    df_fit = pd.DataFrame(
        {"type": type_strings, "scalar_coupling_constant": train_values}
    )

    standardizer = Standardizer()
    standardizer.fit(df_fit)

    # 3. Create Train Dataset/Loader
    train_dataset = MolecularGraphDataset(train_data, standardizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 4. Val Data
    val_data = process_and_cache_data("val", config, load_cached_data)
    val_dataset = MolecularGraphDataset(val_data, standardizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 5. Test Data
    test_data = process_and_cache_data("test", config, load_cached_data)
    test_dataset = MolecularGraphDataset(test_data, standardizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, standardizer
