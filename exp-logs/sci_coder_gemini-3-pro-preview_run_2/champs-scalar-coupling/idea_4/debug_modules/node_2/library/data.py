import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
from library.config import ModelConfig, TrainConfig
from library.features import RadialBasisFunctions, SphericalBasisFunctions

# Constants
ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
TYPE_MAP = {
    "1JHC": 0,
    "2JHC": 1,
    "3JHC": 2,
    "1JHN": 3,
    "2JHN": 4,
    "3JHN": 5,
    "2JHH": 6,
    "3JHH": 7,
}


class MoleculeDataset(Dataset):
    """
    Dataset class for molecular graphs with geometric features.
    Handles loading, caching, and on-the-fly feature computation.
    """

    def __init__(self, split="train", config=None, load_cached_data=True):
        self.split = split
        self.config = config if config else TrainConfig()
        self.model_config = ModelConfig()

        # Paths
        self.metadata_path = os.path.join(
            self.config.metadata_dir, f"{split}_metadata.csv"
        )
        self.structures_path = os.path.join(self.config.input_dir, "structures.csv")
        self.cache_path = os.path.join(self.config.working_dir, f"cached_{split}.npz")

        # Load Metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Debug Mode: Sample subset
        if self.config.debug:
            molecules = self.df["molecule_name"].unique()
            if len(molecules) > 0:
                sample_size = min(len(molecules), self.config.debug_samples // 10 + 1)
                sampled_mols = np.random.choice(
                    molecules, size=sample_size, replace=False
                )
                self.df = self.df[self.df["molecule_name"].isin(sampled_mols)].copy()

        # Compute Normalization Stats (from current dataframe if targets exist)
        self.norm_stats = {}
        if "scalar_coupling_constant" in self.df.columns:
            for ctype in TYPE_MAP.keys():
                subset = self.df[self.df["type"] == ctype]
                if not subset.empty:
                    mean = subset["scalar_coupling_constant"].mean()
                    std = subset["scalar_coupling_constant"].std()
                    self.norm_stats[ctype] = {"mean": float(mean), "std": float(std)}

        # Initialize Feature Extractors
        self.rbf = RadialBasisFunctions(
            num_radial=self.model_config.num_rbf, cutoff=self.model_config.cutoff
        )
        self.sbf = SphericalBasisFunctions(
            num_radial=self.model_config.num_rbf,
            num_spherical=self.model_config.num_sbf,
            cutoff=self.model_config.cutoff,
        )

        # Load or Process Data
        self.data = {}
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                loaded = np.load(self.cache_path, allow_pickle=True)
                for k in loaded.files:
                    self.data[k] = loaded[k]
            except Exception:
                self.process_data()
        else:
            self.process_data()

    def process_data(self):
        """
        Reads structures and metadata, computes graphs, and caches to disk.
        """
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        # Load all structures into memory
        struct_df = pd.read_csv(self.structures_path)
        struct_grp = struct_df.groupby("molecule_name")

        # Group metadata by molecule
        meta_grp = self.df.groupby("molecule_name")
        unique_mols = self.df["molecule_name"].unique()

        # Storage
        all_node_z = []
        all_node_pos = []
        all_edge_index = []
        all_edge_dist = []
        all_edge_vec = []

        all_coupling_atom0 = []
        all_coupling_atom1 = []
        all_coupling_type = []
        all_coupling_val = []
        all_coupling_id = []

        mol_slices = {"node": [0], "edge": [0], "coupling": [0]}

        for mol_name in unique_mols:
            # Get atoms
            if mol_name not in struct_grp.groups:
                continue

            atoms_df = struct_grp.get_group(mol_name).sort_values("atom_index")
            coords = atoms_df[["x", "y", "z"]].values.astype(np.float32)
            atom_types = [ATOM_MAP[a] for a in atoms_df["atom"].values]

            # Compute Neighbors (Edges)
            tree = cKDTree(coords)
            rows, cols = tree.query_pairs(
                r=self.model_config.cutoff, output_type="ndarray"
            ).T

            if len(rows) > 0:
                # Bidirectional edges
                u = np.concatenate([rows, cols])
                v = np.concatenate([cols, rows])
            else:
                u = np.array([], dtype=int)
                v = np.array([], dtype=int)

            # Edge Features
            edge_vecs = coords[v] - coords[u]
            edge_dists = np.linalg.norm(edge_vecs, axis=1)

            # Get Couplings
            couplings_df = meta_grp.get_group(mol_name)
            c_a0 = couplings_df["atom_index_0"].values
            c_a1 = couplings_df["atom_index_1"].values
            c_type = [TYPE_MAP[t] for t in couplings_df["type"].values]
            c_ids = couplings_df["id"].values

            if "scalar_coupling_constant" in couplings_df.columns:
                c_vals = couplings_df["scalar_coupling_constant"].values
            else:
                c_vals = np.zeros(len(c_ids))

            # Append
            all_node_z.append(np.array(atom_types, dtype=np.int64))
            all_node_pos.append(coords)
            all_edge_index.append(
                np.stack([u, v], axis=0)
                if len(u) > 0
                else np.zeros((2, 0), dtype=np.int64)
            )
            all_edge_dist.append(edge_dists)
            all_edge_vec.append(edge_vecs)

            all_coupling_atom0.append(c_a0)
            all_coupling_atom1.append(c_a1)
            all_coupling_type.append(np.array(c_type, dtype=np.int64))
            all_coupling_val.append(c_vals.astype(np.float32))
            all_coupling_id.append(c_ids)

            # Update slices
            mol_slices["node"].append(mol_slices["node"][-1] + len(atom_types))
            mol_slices["edge"].append(mol_slices["edge"][-1] + len(u))
            mol_slices["coupling"].append(mol_slices["coupling"][-1] + len(c_ids))

        # Concatenate and Save
        self.data = {
            "node_z": np.concatenate(all_node_z),
            "node_pos": np.concatenate(all_node_pos),
            "edge_index": (
                np.concatenate(all_edge_index, axis=1)
                if len(all_edge_index) > 0
                else np.zeros((2, 0))
            ),
            "edge_dist": (
                np.concatenate(all_edge_dist) if len(all_edge_dist) > 0 else np.zeros(0)
            ),
            "edge_vec": (
                np.concatenate(all_edge_vec)
                if len(all_edge_vec) > 0
                else np.zeros((0, 3))
            ),
            "coupling_atom0": np.concatenate(all_coupling_atom0),
            "coupling_atom1": np.concatenate(all_coupling_atom1),
            "coupling_type": np.concatenate(all_coupling_type),
            "coupling_val": np.concatenate(all_coupling_val),
            "coupling_id": np.concatenate(all_coupling_id),
            "slice_node": np.array(mol_slices["node"]),
            "slice_edge": np.array(mol_slices["edge"]),
            "slice_coupling": np.array(mol_slices["coupling"]),
        }

        np.savez_compressed(self.cache_path, **self.data)

    def __len__(self):
        return len(self.data["slice_node"]) - 1

    def __getitem__(self, idx):
        # Slicing
        n_s, n_e = self.data["slice_node"][idx], self.data["slice_node"][idx + 1]
        e_s, e_e = self.data["slice_edge"][idx], self.data["slice_edge"][idx + 1]
        c_s, c_e = (
            self.data["slice_coupling"][idx],
            self.data["slice_coupling"][idx + 1],
        )

        # Basic Graph Data
        z = torch.from_numpy(self.data["node_z"][n_s:n_e]).long()
        pos = torch.from_numpy(self.data["node_pos"][n_s:n_e]).float()

        if e_e > e_s:
            edge_index = torch.from_numpy(self.data["edge_index"][:, e_s:e_e]).long()
            edge_dist = torch.from_numpy(self.data["edge_dist"][e_s:e_e]).float()
            edge_vec = torch.from_numpy(self.data["edge_vec"][e_s:e_e]).float()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_dist = torch.zeros(0, dtype=torch.float)
            edge_vec = torch.zeros((0, 3), dtype=torch.float)

        # Coupling Data
        c_atom0 = torch.from_numpy(self.data["coupling_atom0"][c_s:c_e]).long()
        c_atom1 = torch.from_numpy(self.data["coupling_atom1"][c_s:c_e]).long()
        c_type = torch.from_numpy(self.data["coupling_type"][c_s:c_e]).long()
        c_val = torch.from_numpy(self.data["coupling_val"][c_s:c_e]).float()
        c_id = torch.from_numpy(self.data["coupling_id"][c_s:c_e]).long()

        # Normalize Targets
        c_target = c_val.clone()
        if self.norm_stats:
            for i in range(len(c_type)):
                t_idx = c_type[i].item()
                t_str = list(TYPE_MAP.keys())[list(TYPE_MAP.values()).index(t_idx)]
                if t_str in self.norm_stats:
                    stats = self.norm_stats[t_str]
                    if stats["std"] > 1e-7:
                        c_target[i] = (c_target[i] - stats["mean"]) / stats["std"]

        # RBF Features
        edge_rbf = self.rbf(edge_dist)

        # Triplet Calculation for SBF
        # Find pairs (k->j, j->i)
        num_edges = edge_index.size(1)
        if num_edges > 0:
            src, dst = edge_index[0], edge_index[1]

            # Find triplets using broadcasting (efficient for small graphs)
            # e1: k->j, e2: j->i
            # dst[e1] == src[e2] AND src[e1] != dst[e2] (no backtrack)
            adj_mat = dst.unsqueeze(1) == src.unsqueeze(0)
            no_backtrack = src.unsqueeze(1) != dst.unsqueeze(0)
            valid = adj_mat & no_backtrack

            idx_k_j, idx_j_i = torch.nonzero(valid, as_tuple=True)

            # Calculate Angles
            # vec_ji = edge_vec[idx_j_i] (j->i)
            # vec_jk = -edge_vec[idx_k_j] (j->k, reversed k->j)
            r_ji = edge_vec[idx_j_i]
            r_jk = -edge_vec[idx_k_j]

            d_ji = edge_dist[idx_j_i]
            d_jk = edge_dist[idx_k_j]

            dot = (r_ji * r_jk).sum(dim=1)
            cos_theta = dot / (d_ji * d_jk + 1e-7)
            cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cos_theta)

            triplet_sbf = self.sbf(d_jk, theta)
            triplet_indices = torch.stack([idx_k_j, idx_j_i], dim=0)
        else:
            triplet_sbf = torch.zeros(
                (0, self.model_config.num_sbf * self.model_config.num_rbf)
            )
            triplet_indices = torch.zeros((2, 0), dtype=torch.long)

        return {
            "z": z,
            "pos": pos,
            "edge_index": edge_index,
            "edge_rbf": edge_rbf,
            "triplet_indices": triplet_indices,
            "triplet_sbf": triplet_sbf,
            "coupling_atom0": c_atom0,
            "coupling_atom1": c_atom1,
            "coupling_type": c_type,
            "coupling_target": c_target,
            "coupling_id": c_id,
            "num_nodes": z.size(0),
            "num_edges": edge_index.size(1),
        }


def collate_graphs(batch):
    """
    Batches a list of graph dictionaries into a single disjoint graph.
    Shifts indices for edges, triplets, and couplings.
    """
    # Accumulators
    node_z = []
    edge_rbf = []
    triplet_sbf = []

    c_target = []
    c_type = []
    c_id = []

    edge_indices = []
    triplet_indices = []
    c_atom0 = []
    c_atom1 = []
    batch_idx = []

    node_offset = 0
    edge_offset = 0

    for i, data in enumerate(batch):
        num_nodes = data["num_nodes"]
        num_edges = data["num_edges"]

        node_z.append(data["z"])
        batch_idx.append(torch.full((num_nodes,), i, dtype=torch.long))

        edge_rbf.append(data["edge_rbf"])
        edge_indices.append(data["edge_index"] + node_offset)

        triplet_sbf.append(data["triplet_sbf"])
        triplet_indices.append(data["triplet_indices"] + edge_offset)

        c_atom0.append(data["coupling_atom0"] + node_offset)
        c_atom1.append(data["coupling_atom1"] + node_offset)
        c_target.append(data["coupling_target"])
        c_type.append(data["coupling_type"])
        c_id.append(data["coupling_id"])

        node_offset += num_nodes
        edge_offset += num_edges

    return {
        "z": torch.cat(node_z),
        "batch": torch.cat(batch_idx),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_rbf": torch.cat(edge_rbf),
        "triplet_indices": torch.cat(triplet_indices, dim=1),
        "triplet_sbf": torch.cat(triplet_sbf),
        "coupling_atom0": torch.cat(c_atom0),
        "coupling_atom1": torch.cat(c_atom1),
        "coupling_type": torch.cat(c_type),
        "coupling_target": torch.cat(c_target),
        "coupling_id": torch.cat(c_id),
        "batch_size": len(batch),
    }
