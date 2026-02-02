import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import TypeSpecificStandardizer
from library.geometry import get_radius_graph, get_triplets, compute_angles


class ChampsDataset(Dataset):
    def __init__(self, metadata_path, mode="train", load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV (train/val/test).
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load pre-processed arrays from disk.
        """
        self.mode = mode
        self.metadata_path = metadata_path

        # Define cache paths with mode suffix to avoid conflicts between splits
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        suffix = f"_{mode}"
        self.paths = {
            "nodes": os.path.join(self.cache_dir, f"nodes{suffix}.npy"),
            "edges": os.path.join(self.cache_dir, f"edges{suffix}.npy"),
            "edge_attr": os.path.join(self.cache_dir, f"edge_attr{suffix}.npy"),
            "triplets": os.path.join(self.cache_dir, f"triplets{suffix}.npy"),
            "triplet_attr": os.path.join(self.cache_dir, f"triplet_attr{suffix}.npy"),
            "aux": os.path.join(self.cache_dir, f"aux{suffix}.npy"),
            "mol_map": os.path.join(self.cache_dir, f"mol_map{suffix}.parquet"),
            "couplings": os.path.join(self.cache_dir, f"couplings{suffix}.parquet"),
        }

        # Load or Process Data
        if load_cached_data and self._check_cache():
            self._load_cache()
        else:
            self._process_data()
            self._load_cache()

    def _check_cache(self):
        """Checks if all required cache files exist."""
        return all(os.path.exists(p) for p in self.paths.values())

    def _load_cache(self):
        """Loads data from cache files."""
        # Load index maps
        self.mol_map = pd.read_parquet(self.paths["mol_map"])
        self.couplings_df = pd.read_parquet(self.paths["couplings"])

        # Load arrays (loading into memory for speed, assuming sufficient RAM)
        self.nodes = np.load(self.paths["nodes"])
        self.edges = np.load(self.paths["edges"])
        self.edge_attr = np.load(self.paths["edge_attr"])
        self.triplets = np.load(self.paths["triplets"])
        self.triplet_attr = np.load(self.paths["triplet_attr"])
        self.aux = np.load(self.paths["aux"])

        # Create a map for fast coupling retrieval: molecule_name -> (start_idx, count)
        # We assume couplings_df is sorted by molecule_name (enforced in _process_data)
        mols = self.couplings_df["molecule_name"].values
        unique_mols, start_indices, counts = np.unique(
            mols, return_index=True, return_counts=True
        )
        self.coupling_slices = {
            m: (s, c) for m, s, c in zip(unique_mols, start_indices, counts)
        }

        # List of molecules to iterate over
        self.molecule_names = self.mol_map["molecule_name"].values

    def _process_data(self):
        """
        Reads raw CSVs, computes graphs, and saves to cache.
        """
        print(f"Processing data for mode: {self.mode}...")

        # 1. Load Inputs
        df_meta = pd.read_csv(self.metadata_path)
        df_struct = pd.read_csv(Config.STRUCTURES_CSV)

        # Filter structures to only those in metadata
        target_mols = df_meta["molecule_name"].unique()
        df_struct = df_struct[df_struct["molecule_name"].isin(target_mols)]

        # Load Aux Data if available (Train/Val)
        has_aux = self.mode in ["train", "val"]
        df_aux = None
        if has_aux:
            # Load and merge shielding and charges
            shield = pd.read_csv(Config.MAGNETIC_SHIELDING_CSV)
            shield = shield[shield["molecule_name"].isin(target_mols)]

            charges = pd.read_csv(Config.MULLIKEN_CHARGES_CSV)
            charges = charges[charges["molecule_name"].isin(target_mols)]

            # Merge on molecule_name and atom_index
            df_aux = pd.merge(shield, charges, on=["molecule_name", "atom_index"])
            # Ensure sorting matches structures
            df_aux.sort_values(["molecule_name", "atom_index"], inplace=True)

        # Sort structures and metadata to ensure consistent ordering
        df_struct.sort_values(["molecule_name", "atom_index"], inplace=True)
        df_meta.sort_values(["molecule_name"], inplace=True)

        # 2. Prepare Storage Lists
        all_nodes = []
        all_edges = []
        all_edge_attrs = []
        all_triplets = []
        all_triplet_attrs = []
        all_aux = []
        mol_map_rows = []

        # Grouping for iteration
        struct_grp = df_struct.groupby("molecule_name")
        if has_aux:
            aux_grp = df_aux.groupby("molecule_name")

        # Offsets for global arrays
        node_offset = 0
        edge_offset = 0
        triplet_offset = 0

        # Iterate over molecules
        for mol_name in target_mols:
            # Get atoms
            atoms = struct_grp.get_group(mol_name)
            coords = atoms[["x", "y", "z"]].values.astype(np.float32)
            atomic_nums = atoms["atom"].map(Config.ATOM_MAP).values.astype(np.int64)

            num_atoms = len(atomic_nums)

            # Get Aux Targets
            if has_aux:
                try:
                    aux_data = aux_grp.get_group(mol_name)
                    # Extract 9 shielding tensor components + 1 charge
                    shield_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
                    s_vals = aux_data[shield_cols].values
                    c_vals = aux_data["mulliken_charge"].values.reshape(-1, 1)
                    mol_aux = np.hstack([s_vals, c_vals]).astype(np.float32)
                except KeyError:
                    # Fallback (should not occur in valid training data)
                    mol_aux = np.zeros((num_atoms, 10), dtype=np.float32)
            else:
                # For test set, auxiliary targets are zeros
                mol_aux = np.zeros((num_atoms, 10), dtype=np.float32)

            # Compute Graph Topology
            pos_tensor = torch.tensor(coords)
            edge_index, edge_dist = get_radius_graph(pos_tensor, Config.CUTOFF)

            # Compute Triplets (Angles)
            triplets = get_triplets(edge_index, num_atoms)

            # Compute Angle Values
            if triplets.shape[0] > 0:
                angles = compute_angles(pos_tensor, edge_index, triplets)
            else:
                angles = torch.zeros((0,), dtype=torch.float32)

            # Convert to numpy for storage
            edge_index_np = edge_index.numpy().T  # [E, 2]
            edge_dist_np = edge_dist.numpy()  # [E]
            triplets_np = triplets.numpy()  # [T, 2]
            angles_np = angles.numpy()  # [T]

            num_edges = edge_index_np.shape[0]
            num_triplets = triplets_np.shape[0]

            # Append to lists
            all_nodes.append(atomic_nums)
            all_edges.append(edge_index_np)
            all_edge_attrs.append(edge_dist_np)
            all_triplets.append(triplets_np)
            all_triplet_attrs.append(angles_np)
            all_aux.append(mol_aux)

            # Record Metadata Map
            mol_map_rows.append(
                {
                    "molecule_name": mol_name,
                    "node_start": node_offset,
                    "node_count": num_atoms,
                    "edge_start": edge_offset,
                    "edge_count": num_edges,
                    "triplet_start": triplet_offset,
                    "triplet_count": num_triplets,
                }
            )

            node_offset += num_atoms
            edge_offset += num_edges
            triplet_offset += num_triplets

        # Concatenate into single large arrays
        print("Concatenating arrays...")
        nodes_arr = np.concatenate(all_nodes).reshape(-1, 1)  # [Total_N, 1]
        edges_arr = np.concatenate(all_edges).T  # [2, Total_E]
        edge_attr_arr = np.concatenate(all_edge_attrs).reshape(-1, 1)  # [Total_E, 1]

        if len(all_triplets) > 0 and triplet_offset > 0:
            triplets_arr = np.concatenate(all_triplets)  # [Total_T, 2]
            triplet_attr_arr = np.concatenate(all_triplet_attrs).reshape(
                -1, 1
            )  # [Total_T, 1]
        else:
            triplets_arr = np.zeros((0, 2), dtype=np.int64)
            triplet_attr_arr = np.zeros((0, 1), dtype=np.float32)

        aux_arr = np.concatenate(all_aux)  # [Total_N, 10]

        mol_map_df = pd.DataFrame(mol_map_rows)

        # Save to disk
        print("Saving to cache...")
        np.save(self.paths["nodes"], nodes_arr)
        np.save(self.paths["edges"], edges_arr)
        np.save(self.paths["edge_attr"], edge_attr_arr)
        np.save(self.paths["triplets"], triplets_arr)
        np.save(self.paths["triplet_attr"], triplet_attr_arr)
        np.save(self.paths["aux"], aux_arr)
        mol_map_df.to_parquet(self.paths["mol_map"])
        df_meta.to_parquet(self.paths["couplings"])

        print("Processing complete.")

    def __len__(self):
        return len(self.molecule_names)

    def __getitem__(self, idx):
        mol_name = self.molecule_names[idx]

        # 1. Retrieve Graph Data using Map
        row = self.mol_map.iloc[idx]

        # Safety check for alignment
        if row["molecule_name"] != mol_name:
            row = self.mol_map[self.mol_map["molecule_name"] == mol_name].iloc[0]

        n_start, n_cnt = int(row["node_start"]), int(row["node_count"])
        e_start, e_cnt = int(row["edge_start"]), int(row["edge_count"])
        t_start, t_cnt = int(row["triplet_start"]), int(row["triplet_count"])

        # Slice arrays and convert to tensors
        nodes = (
            torch.from_numpy(self.nodes[n_start : n_start + n_cnt]).long().squeeze(-1)
        )
        edge_index = torch.from_numpy(self.edges[:, e_start : e_start + e_cnt]).long()
        edge_attr = (
            torch.from_numpy(self.edge_attr[e_start : e_start + e_cnt])
            .float()
            .squeeze(-1)
        )
        triplets = torch.from_numpy(self.triplets[t_start : t_start + t_cnt]).long()
        triplet_attr = (
            torch.from_numpy(self.triplet_attr[t_start : t_start + t_cnt])
            .float()
            .squeeze(-1)
        )
        aux = torch.from_numpy(self.aux[n_start : n_start + n_cnt]).float()

        # 2. Retrieve Coupling Data
        c_start, c_cnt = self.coupling_slices.get(mol_name, (0, 0))

        if c_cnt > 0:
            subset = self.couplings_df.iloc[c_start : c_start + c_cnt]

            atom0 = torch.from_numpy(subset["atom_index_0"].values).long()
            atom1 = torch.from_numpy(subset["atom_index_1"].values).long()
            coupling_index = torch.stack([atom0, atom1], dim=0)  # [2, C]

            # Map type string to int
            types_str = subset["type"].values
            types_int = np.array([Config.COUPLING_TYPE_MAP[t] for t in types_str])
            coupling_type = torch.from_numpy(types_int).long()

            ids = torch.from_numpy(subset["id"].values).long()

            if "scalar_coupling_constant" in subset.columns:
                targets = torch.from_numpy(
                    subset["scalar_coupling_constant"].values
                ).float()
            else:
                targets = torch.zeros_like(ids).float()
        else:
            coupling_index = torch.zeros((2, 0), dtype=torch.long)
            coupling_type = torch.zeros((0,), dtype=torch.long)
            ids = torch.zeros((0,), dtype=torch.long)
            targets = torch.zeros((0,), dtype=torch.float)

        return {
            "x": nodes,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "triplets": triplets,
            "triplet_attr": triplet_attr,
            "aux": aux,
            "coupling_index": coupling_index,
            "coupling_type": coupling_type,
            "coupling_value": targets,
            "id": ids,
            "num_nodes": n_cnt,
            "num_edges": e_cnt,
            "num_triplets": t_cnt,
        }


def get_collate_fn():
    """
    Returns a collate function for batching variable-sized graphs.
    """

    def collate_fn(batch):
        # Initialize lists
        x_list = []
        edge_index_list = []
        edge_attr_list = []
        triplets_list = []
        triplet_attr_list = []
        aux_list = []

        coupling_index_list = []
        coupling_type_list = []
        coupling_value_list = []
        id_list = []

        batch_idx_list = []

        cum_nodes = 0
        cum_edges = 0

        for i, data in enumerate(batch):
            num_nodes = data["num_nodes"]
            num_edges = data["num_edges"]

            # Nodes & Aux
            x_list.append(data["x"])
            aux_list.append(data["aux"])
            batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))

            # Edges: Shift indices by cumulative nodes
            edge_index_list.append(data["edge_index"] + cum_nodes)
            edge_attr_list.append(data["edge_attr"])

            # Triplets: Shift indices by cumulative edges (since they point to edges)
            triplets_list.append(data["triplets"] + cum_edges)
            triplet_attr_list.append(data["triplet_attr"])

            # Couplings: Shift indices by cumulative nodes
            coupling_index_list.append(data["coupling_index"] + cum_nodes)
            coupling_type_list.append(data["coupling_type"])
            coupling_value_list.append(data["coupling_value"])
            id_list.append(data["id"])

            cum_nodes += num_nodes
            cum_edges += num_edges

        # Concatenate everything
        x = torch.cat(x_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)

        if len(triplets_list) > 0:
            triplets = torch.cat(triplets_list, dim=0)
            triplet_attr = torch.cat(triplet_attr_list, dim=0)
        else:
            triplets = torch.zeros((0, 2), dtype=torch.long)
            triplet_attr = torch.zeros((0,), dtype=torch.float)

        aux = torch.cat(aux_list, dim=0)

        coupling_index = torch.cat(coupling_index_list, dim=1)
        coupling_type = torch.cat(coupling_type_list, dim=0)
        coupling_value = torch.cat(coupling_value_list, dim=0)
        ids = torch.cat(id_list, dim=0)
        batch_idx = torch.cat(batch_idx_list, dim=0)

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "triplets": triplets,
            "triplet_attr": triplet_attr,
            "aux": aux,
            "coupling_index": coupling_index,
            "coupling_type": coupling_type,
            "coupling_value": coupling_value,
            "id": ids,
            "batch": batch_idx,
        }

    return collate_fn
