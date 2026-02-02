import os
import numpy as np
import pandas as pd
import torch
from library.config import Config

# Attempt to import torch_cluster for optimized graph generation
try:
    from torch_cluster import radius_graph
except ImportError:
    radius_graph = None


class DataProcessor:
    """
    Handles the Extraction, Transformation, and Loading (ETL) of raw molecular data
    into a flattened Structure-of-Arrays (SoA) format optimized for memory efficiency
    and high-performance machine learning.
    """

    def __init__(self):
        self.config = Config
        self.atom_map = Config.ATOM_MAP
        self.coupling_map = Config.COUPLING_TYPE_MAP

    def process_structures(self):
        """
        Parses structures.csv to create monolithic node arrays for the entire dataset.

        Returns:
            dict: Contains 'pos', 'node_type', 'mol_id' arrays and mapping helpers.
        """
        print("Loading and processing structures.csv...")
        df = pd.read_csv(self.config.STRUCTURES_CSV)

        # If debugging, limit the number of molecules
        if self.config.DEBUG:
            print(
                f"DEBUG MODE: limiting structures to {self.config.DEBUG_SAMPLE_SIZE} molecules."
            )
            unique_mols = df["molecule_name"].unique()[: self.config.DEBUG_SAMPLE_SIZE]
            df = df[df["molecule_name"].isin(unique_mols)].copy()

        # Ensure data is sorted by molecule and atom index to guarantee contiguous memory blocks
        # and consistent global indexing
        df = df.sort_values(["molecule_name", "atom_index"]).reset_index(drop=True)

        # Map atom types to integers (H=0, C=1, etc.)
        df["type_idx"] = df["atom"].map(self.atom_map).astype(np.int8)

        # Map molecule names to integer IDs
        # We create a mapping that will be used to link train/test CSVs to these structures
        unique_mols = df["molecule_name"].unique()
        mol_name_to_id = {name: i for i, name in enumerate(unique_mols)}
        df["mol_id"] = df["molecule_name"].map(mol_name_to_id).astype(np.int32)

        # Extract monolithic arrays
        pos = df[["x", "y", "z"]].values.astype(np.float32)
        node_type = df["type_idx"].values
        mol_id = df["mol_id"].values

        # Create a fast lookup for the start index of each molecule in the global arrays
        # Since data is sorted by mol_id, we can find unique indices
        _, start_indices = np.unique(mol_id, return_index=True)

        # Verify alignment
        # start_indices[i] should be the index in 'pos' where molecule i starts

        return {
            "pos": pos,
            "node_type": node_type,
            "mol_id": mol_id,
            "mol_start_indices": start_indices,
            "mol_name_to_id": mol_name_to_id,
            "total_atoms": len(df),
        }

    def compute_topology(self, pos, mol_id):
        """
        Computes the graph topology (edges) and angular triplets.

        Args:
            pos (np.array): Coordinates (N, 3).
            mol_id (np.array): Molecule ID for each atom (N,).

        Returns:
            dict: edge_index, edge_vec, edge_dist, triplet_index
        """
        print("Computing graph topology (Radius Graph & Triplets)...")

        # Convert to torch tensors
        pos_t = torch.from_numpy(pos)
        batch_t = torch.from_numpy(mol_id)

        # 1. Generate Radius Graph
        # We use torch_cluster.radius_graph which respects the 'batch' vector
        # (edges are only created between atoms in the same molecule)
        if radius_graph is None:
            raise ImportError("torch_cluster is required for graph generation.")

        # max_num_neighbors limits the degree to prevent explosive growth of triplets
        edge_index = radius_graph(
            pos_t,
            r=self.config.MAX_RADIUS,
            batch=batch_t,
            loop=False,
            max_num_neighbors=50,
            num_workers=self.config.NUM_WORKERS,
        )

        # 2. Compute Edge Attributes
        src, dst = edge_index[0], edge_index[1]
        edge_vec = pos_t[dst] - pos_t[src]
        edge_dist = edge_vec.norm(dim=-1)

        # Convert back to numpy for processing triplets
        edge_index_np = edge_index.numpy()
        edge_vec_np = edge_vec.numpy()
        edge_dist_np = edge_dist.numpy()

        print(f"  Generated {edge_index_np.shape[1]} edges.")

        # 3. Compute Triplets (k -> j -> i)
        # We identify pairs of edges (e_in, e_out) where e_in.dst == e_out.src
        # and e_in.src != e_out.dst (no backtracking/self-loop angles)
        print("  Generating triplets...")

        # Create a DataFrame to perform a self-join
        edges_df = pd.DataFrame(
            {
                "src": edge_index_np[0],
                "dst": edge_index_np[1],
                "edge_idx": np.arange(edge_index_np.shape[1], dtype=np.int32),
            }
        )

        # Inner join: target of incoming edge == source of outgoing edge
        triplets_df = pd.merge(
            edges_df, edges_df, left_on="dst", right_on="src", suffixes=("_in", "_out")
        )

        # Filter out cases where k == i (backtracking), as bond angles involve 3 distinct atoms
        triplets_df = triplets_df[triplets_df["src_in"] != triplets_df["dst_out"]]

        # Construct triplet index array: [edge_idx_in, edge_idx_out, atom_j]
        # atom_j is the center atom, which is dst_in (or src_out)
        triplet_index = np.vstack(
            [
                triplets_df["edge_idx_in"].values,
                triplets_df["edge_idx_out"].values,
                triplets_df["dst_in"].values,
            ]
        ).astype(np.int32)

        print(f"  Generated {triplet_index.shape[1]} triplets.")

        return {
            "edge_index": edge_index_np,
            "edge_vec": edge_vec_np,
            "edge_dist": edge_dist_np,
            "triplet_index": triplet_index,
        }

    def process_auxiliary(self, mol_name_to_id, mol_start_indices, total_atoms):
        """
        Loads auxiliary targets (Charges, Shielding) and maps them to the global node array.
        """
        print("Processing auxiliary targets...")

        # Initialize with zeros
        aux_charge = np.zeros(total_atoms, dtype=np.float32)
        aux_shielding = np.zeros((total_atoms, 9), dtype=np.float32)

        # Helper to map dataframe to global indices
        def get_global_indices(df):
            # Map molecule name to ID
            df["mol_id"] = df["molecule_name"].map(mol_name_to_id)
            df = df.dropna(subset=["mol_id"])
            df["mol_id"] = df["mol_id"].astype(int)

            # Global Index = Start Index of Molecule + Atom Index within Molecule
            start_idxs = mol_start_indices[df["mol_id"].values]
            return start_idxs + df["atom_index"].values

        # 1. Mulliken Charges
        if os.path.exists(self.config.MULLIKEN_CHARGES_CSV):
            df_c = pd.read_csv(self.config.MULLIKEN_CHARGES_CSV)
            # Filter if debug
            if self.config.DEBUG:
                df_c = df_c[df_c["molecule_name"].isin(mol_name_to_id.keys())]

            global_idxs = get_global_indices(df_c)
            aux_charge[global_idxs] = df_c["mulliken_charge"].values.astype(np.float32)

        # 2. Magnetic Shielding
        if os.path.exists(self.config.MAGNETIC_SHIELDING_CSV):
            df_s = pd.read_csv(self.config.MAGNETIC_SHIELDING_CSV)
            if self.config.DEBUG:
                df_s = df_s[df_s["molecule_name"].isin(mol_name_to_id.keys())]

            global_idxs = get_global_indices(df_s)
            cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
            aux_shielding[global_idxs] = df_s[cols].values.astype(np.float32)

        return aux_charge, aux_shielding

    def process_couplings(
        self, metadata_path, mol_name_to_id, mol_start_indices, is_test=False
    ):
        """
        Processes train/val/test metadata to extract coupling pairs and targets.
        """
        print(f"Processing couplings from {metadata_path}...")
        df = pd.read_csv(metadata_path)

        if self.config.DEBUG:
            # Only keep rows where molecule is in our loaded set
            df = df[df["molecule_name"].isin(mol_name_to_id.keys())].copy()

        # Map molecule names to IDs
        df["mol_id"] = df["molecule_name"].map(mol_name_to_id)
        df = df.dropna(
            subset=["mol_id"]
        )  # Should be empty drop if structures are consistent
        df["mol_id"] = df["mol_id"].astype(int)

        # Calculate global indices for the atom pair
        start_idxs = mol_start_indices[df["mol_id"].values]
        atom_0 = start_idxs + df["atom_index_0"].values
        atom_1 = start_idxs + df["atom_index_1"].values

        coupling_atom_index = np.vstack([atom_0, atom_1]).astype(np.int32)

        # Map Coupling Type
        coupling_type = df["type"].map(self.coupling_map).values.astype(np.int8)

        # IDs
        coupling_id = df["id"].values.astype(np.int32)

        # Targets
        if not is_test:
            coupling_value = df["scalar_coupling_constant"].values.astype(np.float32)
        else:
            coupling_value = np.zeros(len(df), dtype=np.float32)

        return {
            "coupling_atom_index": coupling_atom_index,
            "coupling_type": coupling_type,
            "coupling_value": coupling_value,
            "coupling_id": coupling_id,
        }

    def compute_statistics(self, coupling_value, coupling_type):
        """
        Computes Mean and Std for each coupling type for standardization.
        """
        print("Computing per-type target statistics...")
        stats = {}
        for t_name, t_idx in self.coupling_map.items():
            mask = coupling_type == t_idx
            if mask.sum() > 0:
                vals = coupling_value[mask]
                stats[t_idx] = {"mean": float(vals.mean()), "std": float(vals.std())}
            else:
                stats[t_idx] = {"mean": 0.0, "std": 1.0}
        return stats

    def run(self, load_cached_data=True):
        """
        Main driver function.
        Checks for cached data; if not found, runs the full ETL pipeline.
        """
        flag_path = os.path.join(self.config.PROCESSED_DIR, "completed.flag")

        if load_cached_data and os.path.exists(flag_path):
            print("Cached processed data found. Skipping ETL.")
            return

        print("Starting Data Processing Pipeline...")

        # 1. Process Structures (Nodes)
        struct_data = self.process_structures()

        # Save shared node data (using 'train' prefix as master/shared storage)
        np.save(self.config.get_processed_file_path("train", "pos"), struct_data["pos"])
        np.save(
            self.config.get_processed_file_path("train", "node_type"),
            struct_data["node_type"],
        )
        np.save(
            self.config.get_processed_file_path("train", "mol_id"),
            struct_data["mol_id"],
        )

        # 2. Compute Topology (Edges & Triplets)
        topo_data = self.compute_topology(struct_data["pos"], struct_data["mol_id"])

        np.save(
            self.config.get_processed_file_path("train", "edge_index"),
            topo_data["edge_index"],
        )
        np.save(
            self.config.get_processed_file_path("train", "edge_vec"),
            topo_data["edge_vec"],
        )
        np.save(
            self.config.get_processed_file_path("train", "edge_dist"),
            topo_data["edge_dist"],
        )
        np.save(
            self.config.get_processed_file_path("train", "triplet_index"),
            topo_data["triplet_index"],
        )

        # 3. Process Auxiliary Data
        aux_charge, aux_shielding = self.process_auxiliary(
            struct_data["mol_name_to_id"],
            struct_data["mol_start_indices"],
            struct_data["total_atoms"],
        )
        np.save(self.config.get_processed_file_path("train", "aux_charge"), aux_charge)
        np.save(
            self.config.get_processed_file_path("train", "aux_shielding"), aux_shielding
        )

        # 4. Process Train/Val/Test Splits
        splits = [
            ("train", self.config.TRAIN_METADATA, False),
            ("val", self.config.VAL_METADATA, False),
            ("test", self.config.TEST_METADATA, True),
        ]

        train_values = None
        train_types = None

        for split_name, meta_path, is_test in splits:
            data = self.process_couplings(
                meta_path,
                struct_data["mol_name_to_id"],
                struct_data["mol_start_indices"],
                is_test,
            )

            np.save(
                self.config.get_processed_file_path(split_name, "coupling_atom_index"),
                data["coupling_atom_index"],
            )
            np.save(
                self.config.get_processed_file_path(split_name, "coupling_type"),
                data["coupling_type"],
            )
            np.save(
                self.config.get_processed_file_path(split_name, "coupling_value"),
                data["coupling_value"],
            )
            np.save(
                self.config.get_processed_file_path(split_name, "coupling_id"),
                data["coupling_id"],
            )

            if split_name == "train":
                train_values = data["coupling_value"]
                train_types = data["coupling_type"]

        # 5. Compute and Save Statistics
        stats = self.compute_statistics(train_values, train_types)
        np.save(self.config.STATS_FILE, stats)

        # Mark completion
        with open(flag_path, "w") as f:
            f.write("done")

        print("Data Processing Complete.")
