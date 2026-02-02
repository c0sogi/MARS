import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import (
    ATOM_TO_INT,
    COUPLING_TO_INT,
    CUTOFF,
    NUM_RBF,
    NUM_SPHERICAL,
    NUM_RADIAL,
    CACHE_DIR,
    STRUCTURES_CSV,
    STRUCTURES_DIR,
    INPUT_DIR,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
    ENVELOPE_EXPONENT,
)
from library.geometry import RadialBasisFunctions, SphericalBasisFunctions
from library.utils import set_seed


def load_structures():
    """
    Loads structures.csv and returns a dictionary mapping molecule_name to
    (atoms, coords).
    """
    print(f"Loading structures from {STRUCTURES_CSV}...")
    df = pd.read_csv(STRUCTURES_CSV)

    # Sort by molecule_name and atom_index to ensure deterministic order
    df = df.sort_values(["molecule_name", "atom_index"])

    # Group by molecule_name
    grouped = df.groupby("molecule_name")

    structure_map = {}
    # Convert to dict of numpy arrays
    for name, group in grouped:
        structure_map[name] = {
            "atoms": group["atom"].values,
            "coords": group[["x", "y", "z"]].values.astype(np.float32),
        }

    return structure_map


def read_xyz(mol_name, custom_path=None):
    """
    Reads a single XYZ file as a fallback if the molecule is missing from structures.csv.
    """
    if custom_path:
        path = custom_path
    else:
        path = os.path.join(STRUCTURES_DIR, f"{mol_name}.xyz")

    if not os.path.exists(path):
        print(f"Warning: XYZ file not found at {path}")
        return None

    try:
        with open(path, "r") as f:
            lines = f.readlines()
            # Line 1: num atoms (skip)
            # Line 2: comment (skip)
            # Line 3+: atom x y z
            atoms = []
            coords = []
            for line in lines[2:]:
                parts = line.split()
                if not parts:
                    continue
                atoms.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])

        return {
            "atoms": np.array(atoms),
            "coords": np.array(coords, dtype=np.float32),
        }
    except Exception as e:
        print(f"Warning: Failed to read XYZ for {mol_name}: {e}")
        return None


class MolecularGraphDataset(Dataset):
    def __init__(self, metadata_path, split_name="train", load_cached_data=True):
        self.split_name = split_name
        self.metadata_path = metadata_path

        # Define cache path
        debug_suffix = "_debug" if DEBUG else ""
        self.cache_path = os.path.join(
            CACHE_DIR, f"cached_{split_name}{debug_suffix}.npz"
        )

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Load or Process
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached data from {self.cache_path}...")
            # Load into memory as a dictionary
            self.data = dict(np.load(self.cache_path))
        else:
            print(f"Processing data for {split_name} (Debug={DEBUG})...")
            self.process_and_cache()
            # Reload to ensure consistency
            self.data = dict(np.load(self.cache_path))

        # Extract offsets for indexing
        self.node_offsets = self.data["node_offsets"]
        self.edge_offsets = self.data["edge_offsets"]
        self.triplet_offsets = self.data["triplet_offsets"]
        self.target_offsets = self.data["target_offsets"]
        self.num_graphs = len(self.node_offsets) - 1

    def process_and_cache(self):
        # 1. Load Metadata
        df = pd.read_csv(self.metadata_path)

        if DEBUG:
            # Select a subset of molecules
            mols = df["molecule_name"].unique()[:DEBUG_SAMPLE_SIZE]
            df = df[df["molecule_name"].isin(mols)].copy()

        # Group by molecule to process graph-wise
        molecules = df["molecule_name"].unique()
        molecules.sort()  # Deterministic order

        # 2. Load Structures
        structure_map = load_structures()

        # 3. Initialize Basis Functions (CPU)
        rbf_fn = RadialBasisFunctions(NUM_RBF, CUTOFF)
        sbf_fn = SphericalBasisFunctions(
            NUM_SPHERICAL, NUM_RADIAL, CUTOFF, ENVELOPE_EXPONENT
        )

        # 4. Storage Lists
        all_atom_types = []
        all_coords = []
        all_edge_index = []
        all_edge_attr = []
        all_triplet_index = []
        all_triplet_attr = []
        all_target_val = []
        all_target_type = []
        all_target_edge_idx = []

        # Offsets
        node_offsets = [0]
        edge_offsets = [0]
        triplet_offsets = [0]
        target_offsets = [0]

        # Group metadata for fast lookup
        mol_groups = {k: v for k, v in df.groupby("molecule_name")}

        # Create path mapping from metadata if available
        mol_to_path = {}
        if "structure_path" in df.columns:
            # Create a mapping from molecule_name to structure_path
            # Use drop_duplicates to handle multiple rows per molecule efficiently
            path_df = df[["molecule_name", "structure_path"]].drop_duplicates()
            mol_to_path = dict(zip(path_df["molecule_name"], path_df["structure_path"]))

        print(f"Constructing graphs for {len(molecules)} molecules...")

        for i, mol_name in enumerate(molecules):
            if i % 2000 == 0 and i > 0:
                print(f"  Processed {i}/{len(molecules)}")

            # Get Structure
            struct = structure_map.get(mol_name)
            if struct is None:
                # Fallback to XYZ file using verified path from metadata if available
                custom_path = None
                rel_path = mol_to_path.get(mol_name)
                if rel_path:
                    custom_path = os.path.join(INPUT_DIR, rel_path)

                struct = read_xyz(mol_name, custom_path=custom_path)

            if struct is None:
                continue

            atoms = struct["atoms"]
            coords = torch.from_numpy(struct["coords"])  # (N, 3)
            n_atoms = len(atoms)

            # Atom Types
            z = [ATOM_TO_INT[a] for a in atoms]

            # --- Graph Construction ---
            # 1. Calculate Distances
            # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)
            dists = diff.norm(dim=-1)  # (N, N)

            # 2. Adjacency (Dist < Cutoff, excluding self)
            mask = (dists < CUTOFF) & (dists > 1e-4)
            src, dst = torch.where(mask)

            # Edge Attributes (RBF)
            edge_dists = dists[src, dst]
            edge_rbf = rbf_fn(edge_dists)  # (E, num_rbf)

            # Store Edges
            edge_index = torch.stack([src, dst], dim=0)
            num_edges = edge_index.shape[1]

            # --- Triplets ---
            # Find triplets k -> j -> i (incoming to j, outgoing from j)
            # Create adjacency lists for fast lookup
            incoming = [[] for _ in range(n_atoms)]
            outgoing = [[] for _ in range(n_atoms)]

            for e_idx in range(num_edges):
                u, v = src[e_idx].item(), dst[e_idx].item()
                incoming[v].append(e_idx)
                outgoing[u].append(e_idx)

            triplets_kji = []
            triplet_indices = []  # (edge_idx_kj, edge_idx_ji)

            for j in range(n_atoms):
                for e_kj in incoming[j]:
                    k = src[e_kj].item()
                    for e_ji in outgoing[j]:
                        i = dst[e_ji].item()
                        if k != i:
                            triplets_kji.append((k, j, i))
                            triplet_indices.append([e_kj, e_ji])

            if len(triplets_kji) > 0:
                triplet_indices = torch.tensor(
                    triplet_indices, dtype=torch.long
                ).t()  # (2, T)

                # Compute Angles
                triplets_tensor = torch.tensor(triplets_kji, dtype=torch.long)
                k_idx = triplets_tensor[:, 0]
                j_idx = triplets_tensor[:, 1]
                i_idx = triplets_tensor[:, 2]

                vec_ji = coords[i_idx] - coords[j_idx]
                vec_jk = coords[k_idx] - coords[j_idx]

                norm_ji = vec_ji.norm(dim=1)
                norm_jk = vec_jk.norm(dim=1)

                dot = (vec_ji * vec_jk).sum(dim=1)
                cos_theta = dot / (norm_ji * norm_jk + 1e-7)
                cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
                angles = torch.acos(cos_theta)

                # SBF Features
                # Pass distances of the k->j edges and the angles
                # edge_dists contains all edge distances. triplet_indices[0] are the k->j edge indices.
                sbf_feat = sbf_fn(edge_dists, angles, triplet_indices[0])  # (T, dim)

            else:
                triplet_indices = torch.zeros((2, 0), dtype=torch.long)
                sbf_feat = torch.zeros(
                    (0, NUM_SPHERICAL * NUM_RADIAL), dtype=torch.float32
                )

            # --- Targets ---
            # Map targets in metadata to edges
            # Create a dense lookup for edge indices: (u, v) -> edge_idx
            edge_lookup = torch.full((n_atoms, n_atoms), -1, dtype=torch.long)
            edge_lookup[src, dst] = torch.arange(num_edges)

            mol_df = mol_groups[mol_name]

            target_vals = []
            target_types = []
            target_e_indices = []

            for _, row in mol_df.iterrows():
                u = int(row["atom_index_0"])
                v = int(row["atom_index_1"])

                e_idx = edge_lookup[u, v].item()

                if e_idx != -1:
                    target_e_indices.append(e_idx)
                    target_types.append(COUPLING_TO_INT[row["type"]])
                    if "scalar_coupling_constant" in row:
                        target_vals.append(row["scalar_coupling_constant"])
                    else:
                        target_vals.append(0.0)
                else:
                    # Very rare case where coupling pair is > CUTOFF
                    pass

            # Convert to tensors
            target_vals = torch.tensor(target_vals, dtype=torch.float32)
            target_types = torch.tensor(target_types, dtype=torch.long)
            target_e_indices = torch.tensor(target_e_indices, dtype=torch.long)

            # --- Append to global lists ---
            all_atom_types.append(torch.tensor(z, dtype=torch.long))
            all_coords.append(coords)
            all_edge_index.append(edge_index)
            all_edge_attr.append(edge_rbf)
            all_triplet_index.append(triplet_indices)
            all_triplet_attr.append(sbf_feat)
            all_target_val.append(target_vals)
            all_target_type.append(target_types)
            all_target_edge_idx.append(target_e_indices)

            # Update offsets
            node_offsets.append(node_offsets[-1] + n_atoms)
            edge_offsets.append(edge_offsets[-1] + num_edges)
            triplet_offsets.append(triplet_offsets[-1] + triplet_indices.shape[1])
            target_offsets.append(target_offsets[-1] + len(target_vals))

        # Concatenate
        print("Concatenating arrays...")
        if not all_atom_types:
            raise RuntimeError(
                "No molecule data processed! Check structure files and metadata alignment."
            )

        data_dict = {
            "atom_types": torch.cat(all_atom_types).numpy(),
            "coords": torch.cat(all_coords).numpy(),
            "edge_index": torch.cat(all_edge_index, dim=1).numpy(),
            "edge_attr": torch.cat(all_edge_attr).numpy(),
            "triplet_index": torch.cat(all_triplet_index, dim=1).numpy(),
            "triplet_attr": torch.cat(all_triplet_attr).numpy(),
            "target_val": torch.cat(all_target_val).numpy(),
            "target_type": torch.cat(all_target_type).numpy(),
            "target_edge_idx": torch.cat(all_target_edge_idx).numpy(),
            "node_offsets": np.array(node_offsets),
            "edge_offsets": np.array(edge_offsets),
            "triplet_offsets": np.array(triplet_offsets),
            "target_offsets": np.array(target_offsets),
        }

        # Save
        print(f"Saving to {self.cache_path}...")
        np.savez(self.cache_path, **data_dict)
        print("Done.")

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        # Slice data for graph idx
        n_start, n_end = self.node_offsets[idx], self.node_offsets[idx + 1]
        e_start, e_end = self.edge_offsets[idx], self.edge_offsets[idx + 1]
        t_start, t_end = self.triplet_offsets[idx], self.triplet_offsets[idx + 1]
        tgt_start, tgt_end = self.target_offsets[idx], self.target_offsets[idx + 1]

        return {
            "x": torch.from_numpy(self.data["atom_types"][n_start:n_end]).long(),
            "pos": torch.from_numpy(self.data["coords"][n_start:n_end]).float(),
            "edge_index": torch.from_numpy(
                self.data["edge_index"][:, e_start:e_end]
            ).long(),
            "edge_attr": torch.from_numpy(
                self.data["edge_attr"][e_start:e_end]
            ).float(),
            "triplet_index": torch.from_numpy(
                self.data["triplet_index"][:, t_start:t_end]
            ).long(),
            "triplet_attr": torch.from_numpy(
                self.data["triplet_attr"][t_start:t_end]
            ).float(),
            "y": torch.from_numpy(self.data["target_val"][tgt_start:tgt_end]).float(),
            "target_type": torch.from_numpy(
                self.data["target_type"][tgt_start:tgt_end]
            ).long(),
            "target_edge_index": torch.from_numpy(
                self.data["target_edge_idx"][tgt_start:tgt_end]
            ).long(),
        }


def collate_graphs(batch):
    """
    Batches a list of graph dictionaries.
    """
    # Initialize lists
    x_list = []
    pos_list = []
    edge_index_list = []
    edge_attr_list = []
    triplet_index_list = []
    triplet_attr_list = []
    y_list = []
    target_type_list = []
    target_edge_index_list = []

    # Counters for offsets
    num_nodes_cumsum = 0
    num_edges_cumsum = 0

    batch_idx = []

    for i, data in enumerate(batch):
        num_nodes = data["x"].shape[0]
        num_edges = data["edge_index"].shape[1]

        # Nodes
        x_list.append(data["x"])
        pos_list.append(data["pos"])
        batch_idx.append(torch.full((num_nodes,), i, dtype=torch.long))

        # Edges (shift indices)
        edge_index_list.append(data["edge_index"] + num_nodes_cumsum)
        edge_attr_list.append(data["edge_attr"])

        # Triplets (shift indices refer to edges)
        triplet_index_list.append(data["triplet_index"] + num_edges_cumsum)
        triplet_attr_list.append(data["triplet_attr"])

        # Targets
        y_list.append(data["y"])
        target_type_list.append(data["target_type"])
        # Target edge indices also point to edges
        target_edge_index_list.append(data["target_edge_index"] + num_edges_cumsum)

        # Update counters
        num_nodes_cumsum += num_nodes
        num_edges_cumsum += num_edges

    # Concatenate
    batch_data = {
        "x": torch.cat(x_list, dim=0),
        "pos": torch.cat(pos_list, dim=0),
        "batch": torch.cat(batch_idx, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "triplet_index": torch.cat(triplet_index_list, dim=1),
        "triplet_attr": torch.cat(triplet_attr_list, dim=0),
        "y": torch.cat(y_list, dim=0),
        "target_type": torch.cat(target_type_list, dim=0),
        "target_edge_index": torch.cat(target_edge_index_list, dim=0),
    }

    return batch_data
