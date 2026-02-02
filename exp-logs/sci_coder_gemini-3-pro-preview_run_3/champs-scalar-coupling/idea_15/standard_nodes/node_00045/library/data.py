import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from library.config import (
    ATOM_MAP,
    TYPE_MAP,
    RBF_RADIUS,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    STRUCTURES_PATH,
    MULLIKEN_PATH,
    SHIELDING_PATH,
    TRAIN_CACHE_DIR,
    VAL_CACHE_DIR,
    TEST_CACHE_DIR,
    STATS_PATH,
    NUM_WORKERS,
    INPUT_DIR,
)
from library.utils import Standardizer


def process_data(metadata_path, cache_dir, mode="train", load_cached_data=True):
    """
    Processes raw CSV data into packed Numpy arrays for efficient loading.
    Implements caching and per-molecule graph construction.
    """
    os.makedirs(cache_dir, exist_ok=True)
    flag_path = os.path.join(cache_dir, "completed.flag")

    # 1. Load Cached Data if available
    if load_cached_data and os.path.exists(flag_path):
        # print(f"Loading cached {mode} data from {cache_dir}...")
        try:
            data = {}
            keys = [
                "node_types",
                "edge_indices",
                "edge_attrs",
                "coupling_pairs",
                "coupling_types",
                "coupling_ids",
                "coupling_dists",
                "slice_nodes",
                "slice_edges",
                "slice_couplings",
            ]
            if mode != "test":
                keys.append("coupling_values")
                keys.append("aux_charges")
                keys.append("aux_shielding")

            for k in keys:
                data[k] = np.load(os.path.join(cache_dir, f"{k}.npy"))

            # Load molecule names separately as they are strings
            data["molecule_names"] = np.load(
                os.path.join(cache_dir, "molecule_names.npy")
            )
            return data
        except Exception as e:
            print(f"Cache load failed: {e}. Reprocessing...")

    print(f"Processing {mode} data...")

    # 2. Load Raw Data
    df_meta = pd.read_csv(metadata_path)
    df_struct = pd.read_csv(STRUCTURES_PATH)

    # Load Auxiliary Data for Training
    if mode != "test":
        print("Loading auxiliary physics data...")
        df_mulliken = pd.read_csv(MULLIKEN_PATH)
        df_shielding = pd.read_csv(SHIELDING_PATH)

        # Merge aux data into structures
        # Both are indexed by molecule_name and atom_index
        df_struct = pd.merge(
            df_struct, df_mulliken, on=["molecule_name", "atom_index"], how="left"
        )
        df_struct = pd.merge(
            df_struct, df_shielding, on=["molecule_name", "atom_index"], how="left"
        )

        # Fill NaNs if any (shouldn't be for train)
        df_struct = df_struct.fillna(0.0)

    # Filter structures to only those in the metadata
    relevant_mols = df_meta["molecule_name"].unique()
    df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)]

    # Sort for consistent processing order
    df_meta = df_meta.sort_values("molecule_name")
    # Group structures by molecule name
    struct_grp = df_struct.groupby("molecule_name")
    meta_grp = df_meta.groupby("molecule_name")

    # Lists to accumulate data
    all_node_types = []
    all_edge_indices = []
    all_edge_attrs = []
    all_coupling_pairs = []
    all_coupling_types = []
    all_coupling_values = []
    all_coupling_ids = []
    all_coupling_dists = []

    # Aux lists
    all_aux_charges = []
    all_aux_shielding = []

    # Slice pointers (start_index, count)
    slice_nodes = []
    slice_edges = []
    slice_couplings = []
    molecule_names = []

    # Counters
    node_cnt = 0
    edge_cnt = 0
    coupling_cnt = 0

    # Iterate over molecules
    # We iterate over relevant_mols to ensure alignment
    for mol_name in relevant_mols:
        # --- Process Structure (Nodes & Edges) ---
        atoms = None
        coords = None
        mol_struct = None

        if mol_name in struct_grp.groups:
            mol_struct = struct_grp.get_group(mol_name).sort_values("atom_index")
            coords = mol_struct[["x", "y", "z"]].values
            atoms = mol_struct["atom"].values
        elif mol_name in meta_grp.groups:
            # Fallback: Load from XYZ file (Cite debug_lesson_11)
            meta_row = meta_grp.get_group(mol_name).iloc[0]
            if "structure_path" in meta_row:
                full_path = os.path.join(INPUT_DIR, meta_row["structure_path"])
                if os.path.exists(full_path):
                    try:
                        with open(full_path, "r") as f:
                            lines = f.readlines()
                        # XYZ format: Line 1=Count, Line 2=Comment, Line 3+=Atom X Y Z
                        data_lines = [l.split() for l in lines[2:] if l.strip()]
                        atoms = np.array([row[0] for row in data_lines])
                        coords = np.array(
                            [[float(val) for val in row[1:4]] for row in data_lines]
                        )
                    except Exception:
                        pass

        if atoms is None or coords is None:
            continue

        # Node Features
        # Map atom types to integers
        node_types = np.array([ATOM_MAP[a] for a in atoms], dtype=np.int64)
        n_atoms = len(node_types)

        # Aux Features
        if mode != "test":
            charges = mol_struct["mulliken_charge"].values.astype(np.float32)
            # Shielding columns: XX, YX, ZX, XY, YY, ZY, XZ, YZ, ZZ
            shielding_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
            shielding = mol_struct[shielding_cols].values.astype(np.float32)
            all_aux_charges.append(charges)
            all_aux_shielding.append(shielding)

        # Edge Features
        # Compute pairwise distances
        dists = squareform(pdist(coords))

        # Create sparse graph based on radius
        # Exclude self-loops (dist > 0) and enforce radius
        mask = (dists < RBF_RADIUS) & (dists > 1e-6)
        src, dst = np.where(mask)
        edge_dists = dists[src, dst]

        # --- Process Couplings ---
        if mol_name in meta_grp.groups:
            mol_meta = meta_grp.get_group(mol_name)

            c_pairs = (
                mol_meta[["atom_index_0", "atom_index_1"]].values.astype(np.int64).T
            )  # (2, K)
            c_types = np.array(
                [TYPE_MAP[t] for t in mol_meta["type"].values], dtype=np.int64
            )
            c_ids = mol_meta["id"].values.astype(np.int64)

            # Extract distances for coupling pairs (Cite solution_lesson_node_00011)
            # c_pairs is (2, K), so we index dists
            c_dists = dists[c_pairs[0], c_pairs[1]].astype(np.float32)

            if mode != "test":
                c_vals = mol_meta["scalar_coupling_constant"].values.astype(np.float32)
            else:
                c_vals = np.zeros(len(c_ids), dtype=np.float32)
        else:
            # Should not happen given split logic, but handle empty case
            c_pairs = np.zeros((2, 0), dtype=np.int64)
            c_types = np.zeros(0, dtype=np.int64)
            c_ids = np.zeros(0, dtype=np.int64)
            c_vals = np.zeros(0, dtype=np.float32)
            c_dists = np.zeros(0, dtype=np.float32)

        n_edges = len(edge_dists)
        n_couplings = len(c_types)

        # Append to lists
        all_node_types.append(node_types)
        all_edge_indices.append(np.stack([src, dst], axis=0))
        all_edge_attrs.append(edge_dists)
        all_coupling_pairs.append(c_pairs)
        all_coupling_types.append(c_types)
        all_coupling_values.append(c_vals)
        all_coupling_ids.append(c_ids)
        all_coupling_dists.append(c_dists)
        molecule_names.append(mol_name)

        # Record slices
        slice_nodes.append((node_cnt, n_atoms))
        slice_edges.append((edge_cnt, n_edges))
        slice_couplings.append((coupling_cnt, n_couplings))

        node_cnt += n_atoms
        edge_cnt += n_edges
        coupling_cnt += n_couplings

    # 3. Concatenate and Save
    data = {
        "node_types": np.concatenate(all_node_types),
        "edge_indices": np.concatenate(all_edge_indices, axis=1),
        "edge_attrs": np.concatenate(all_edge_attrs),
        "coupling_pairs": np.concatenate(all_coupling_pairs, axis=1),
        "coupling_types": np.concatenate(all_coupling_types),
        "coupling_ids": np.concatenate(all_coupling_ids),
        "coupling_values": np.concatenate(all_coupling_values),
        "coupling_dists": np.concatenate(all_coupling_dists),
        "slice_nodes": np.array(slice_nodes, dtype=np.int64),
        "slice_edges": np.array(slice_edges, dtype=np.int64),
        "slice_couplings": np.array(slice_couplings, dtype=np.int64),
        "molecule_names": np.array(molecule_names),
    }

    if mode != "test":
        data["aux_charges"] = np.concatenate(all_aux_charges)
        data["aux_shielding"] = np.concatenate(all_aux_shielding)

    # Save to disk
    for k, v in data.items():
        np.save(os.path.join(cache_dir, f"{k}.npy"), v)

    # Create success flag
    with open(flag_path, "w") as f:
        f.write("done")

    # 4. Fit Standardizer (Train only)
    if mode == "train":
        print("Fitting Standardizer on training data...")
        std = Standardizer(STATS_PATH)
        std.fit(data["coupling_types"], data["coupling_values"])

    return data


class MolecularGraphDataset(Dataset):
    """
    Dataset that serves molecular graphs from packed Numpy arrays.
    Each item is a dictionary representing one molecule and its couplings.
    """

    def __init__(self, data_dict):
        self.node_types = data_dict["node_types"]
        self.edge_indices = data_dict["edge_indices"]
        self.edge_attrs = data_dict["edge_attrs"]
        self.coupling_pairs = data_dict["coupling_pairs"]
        self.coupling_types = data_dict["coupling_types"]
        self.coupling_values = data_dict["coupling_values"]
        self.coupling_ids = data_dict["coupling_ids"]
        self.coupling_dists = data_dict["coupling_dists"]

        # Aux data
        self.aux_charges = data_dict.get("aux_charges", None)
        self.aux_shielding = data_dict.get("aux_shielding", None)

        self.slice_nodes = data_dict["slice_nodes"]
        self.slice_edges = data_dict["slice_edges"]
        self.slice_couplings = data_dict["slice_couplings"]
        self.molecule_names = data_dict["molecule_names"]

        self.num_molecules = len(self.molecule_names)

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        # Retrieve slices
        n_start, n_len = self.slice_nodes[idx]
        e_start, e_len = self.slice_edges[idx]
        c_start, c_len = self.slice_couplings[idx]

        # Slice arrays
        item = {
            "node_types": torch.from_numpy(
                self.node_types[n_start : n_start + n_len]
            ).long(),
            "edge_index": torch.from_numpy(
                self.edge_indices[:, e_start : e_start + e_len]
            ).long(),
            "edge_attr": torch.from_numpy(self.edge_attrs[e_start : e_start + e_len])
            .float()
            .unsqueeze(-1),
            "coupling_atom_index": torch.from_numpy(
                self.coupling_pairs[:, c_start : c_start + c_len]
            ).long(),
            "coupling_type": torch.from_numpy(
                self.coupling_types[c_start : c_start + c_len]
            ).long(),
            "coupling_value": torch.from_numpy(
                self.coupling_values[c_start : c_start + c_len]
            ).float(),
            "coupling_id": torch.from_numpy(
                self.coupling_ids[c_start : c_start + c_len]
            ).long(),
            "coupling_dist": torch.from_numpy(
                self.coupling_dists[c_start : c_start + c_len]
            )
            .float()
            .unsqueeze(-1),
            "num_nodes": n_len,
            "molecule_name": str(self.molecule_names[idx]),
        }

        if self.aux_charges is not None:
            item["aux_charges"] = torch.from_numpy(
                self.aux_charges[n_start : n_start + n_len]
            ).float()
            item["aux_shielding"] = torch.from_numpy(
                self.aux_shielding[n_start : n_start + n_len]
            ).float()

        return item


def collate_molecules(batch):
    """
    Collates a list of molecule dictionaries into a single batch.
    Offsets node indices to create a disjoint union of graphs.
    """
    # Initialize lists
    node_types_list = []
    edge_index_list = []
    edge_attr_list = []
    coupling_atom_index_list = []
    coupling_type_list = []
    coupling_value_list = []
    coupling_id_list = []
    coupling_dist_list = []
    aux_charges_list = []
    aux_shielding_list = []
    batch_index_list = []

    cumulative_nodes = 0

    for i, item in enumerate(batch):
        num_nodes = item["num_nodes"]

        # Nodes
        node_types_list.append(item["node_types"])

        # Batch index (for pooling if needed, though MP-IN is node-centric)
        batch_index_list.append(torch.full((num_nodes,), i, dtype=torch.long))

        # Edges (offset indices)
        edge_index_list.append(item["edge_index"] + cumulative_nodes)
        edge_attr_list.append(item["edge_attr"])

        # Couplings (offset indices)
        coupling_atom_index_list.append(item["coupling_atom_index"] + cumulative_nodes)
        coupling_type_list.append(item["coupling_type"])
        coupling_value_list.append(item["coupling_value"])
        coupling_id_list.append(item["coupling_id"])
        coupling_dist_list.append(item["coupling_dist"])

        # Aux
        if "aux_charges" in item:
            aux_charges_list.append(item["aux_charges"])
            aux_shielding_list.append(item["aux_shielding"])

        cumulative_nodes += num_nodes

    # Concatenate
    batch_data = {
        "node_types": torch.cat(node_types_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "coupling_atom_index": torch.cat(coupling_atom_index_list, dim=1),
        "coupling_type": torch.cat(coupling_type_list, dim=0),
        "coupling_value": torch.cat(coupling_value_list, dim=0),
        "coupling_id": torch.cat(coupling_id_list, dim=0),
        "coupling_dist": torch.cat(coupling_dist_list, dim=0),
        "batch": torch.cat(batch_index_list, dim=0),
        "num_graphs": len(batch),
    }

    if aux_charges_list:
        batch_data["aux_charges"] = torch.cat(aux_charges_list, dim=0)
        batch_data["aux_shielding"] = torch.cat(aux_shielding_list, dim=0)

    return batch_data


def get_train_val_datasets(load_cached=True, debug=False, debug_size=1000):
    """
    Returns (train_dataset, val_dataset).
    """
    train_data = process_data(
        TRAIN_METADATA_PATH, TRAIN_CACHE_DIR, mode="train", load_cached_data=load_cached
    )
    val_data = process_data(
        VAL_METADATA_PATH, VAL_CACHE_DIR, mode="val", load_cached_data=load_cached
    )

    train_dataset = MolecularGraphDataset(train_data)
    val_dataset = MolecularGraphDataset(val_data)

    if debug:
        print(f"DEBUG MODE: Trimming datasets to {debug_size} molecules.")
        # Simple slicing for debug (modifies the dataset object in place effectively)
        # Note: This doesn't resize the underlying huge arrays, just limits the indices accessed
        train_dataset.num_molecules = min(len(train_dataset), debug_size)
        val_dataset.num_molecules = min(len(val_dataset), debug_size)

    return train_dataset, val_dataset


def get_test_dataset(load_cached=True):
    """
    Returns test_dataset.
    """
    test_data = process_data(
        TEST_METADATA_PATH, TEST_CACHE_DIR, mode="test", load_cached_data=load_cached
    )
    return MolecularGraphDataset(test_data)
