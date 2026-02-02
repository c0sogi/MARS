import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph
from tqdm import tqdm
from library.config import Config
from library.utils import TargetScaler


class MolecularData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "line_edge_index":
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)


class MoleculeDataset(InMemoryDataset):
    """
    PyG Dataset for Scalar Coupling Prediction.
    Constructs Atom Graphs and Line Graphs (Edge Graphs) for geometric deep learning.
    """

    def __init__(
        self, root, mode, metadata_path, transform=None, pre_transform=None, debug=False
    ):
        self.mode = mode
        self.metadata_path = metadata_path
        self.debug = debug

        # Ensure processed directory exists
        os.makedirs(os.path.join(root, "processed"), exist_ok=True)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # We rely on Config paths and metadata, so we don't enforce raw file checks here
        return []

    @property
    def processed_file_names(self):
        # Define filename based on mode and debug status
        suffix = "_debug" if self.debug else ""
        return [f"{self.mode}_data{suffix}.pt"]

    def process(self):
        print(f"Processing {self.mode} dataset (Debug={self.debug})...")

        # 1. Load Metadata
        df_meta = pd.read_csv(self.metadata_path)
        if self.debug:
            df_meta = df_meta.iloc[: Config.DEBUG_SAMPLES]

        # 2. Load Structures (Global)
        # Loading entire structures.csv is efficient enough (approx 100MB)
        print("Loading structures...")
        df_struct = pd.read_csv(Config.STRUCTURES_PATH)
        # Index by molecule_name for fast lookup
        struct_grp = df_struct.groupby("molecule_name")

        # 3. Load Auxiliary Data (Only if available/needed, but we load to check coverage)
        # We treat these as targets. If missing (e.g. Test), we fill with NaNs.
        aux_shielding = None
        aux_charges = None

        if os.path.exists(Config.MAGNETIC_SHIELDING_PATH):
            print("Loading shielding tensors...")
            df_shield = pd.read_csv(Config.MAGNETIC_SHIELDING_PATH)
            shield_grp = df_shield.groupby("molecule_name")
        else:
            shield_grp = None

        if os.path.exists(Config.MULLIKEN_CHARGES_PATH):
            print("Loading mulliken charges...")
            df_charge = pd.read_csv(Config.MULLIKEN_CHARGES_PATH)
            charge_grp = df_charge.groupby("molecule_name")
        else:
            charge_grp = None

        # 4. Group Metadata by Molecule
        # This gives us all coupling targets for a single molecule
        meta_grp = df_meta.groupby("molecule_name")

        data_list = []

        # Iterate over molecules in the metadata
        # We use the unique molecules from metadata to drive the process
        unique_mols = df_meta["molecule_name"].unique()

        for mol_name in tqdm(unique_mols, desc=f"Building Graphs ({self.mode})"):
            # --- E. Coupling Targets (Primary) ---
            # Load targets first to ensure we have info for dummy creation if needed
            mol_targets = meta_grp.get_group(mol_name)

            coupling_atom0 = torch.tensor(
                mol_targets["atom_index_0"].values, dtype=torch.long
            )
            coupling_atom1 = torch.tensor(
                mol_targets["atom_index_1"].values, dtype=torch.long
            )
            coupling_indices = torch.stack(
                [coupling_atom0, coupling_atom1], dim=0
            )  # (2, num_couplings)

            # Coupling Types
            type_indices = [
                Config.get_coupling_type_id(t) for t in mol_targets["type"].values
            ]
            coupling_type = torch.tensor(type_indices, dtype=torch.long)

            # Submission IDs
            coupling_id = torch.tensor(mol_targets["id"].values, dtype=torch.long)

            # Target Values (if available)
            if "scalar_coupling_constant" in mol_targets.columns:
                y = torch.tensor(
                    mol_targets["scalar_coupling_constant"].values, dtype=torch.float
                )
            else:
                y = torch.zeros(
                    len(mol_targets), dtype=torch.float
                )  # Placeholder for test

            # --- A. Get Atom Features & Positions ---
            try:
                mol_struct = struct_grp.get_group(mol_name)
                # Sort by atom_index to ensure 0..N alignment
                mol_struct = mol_struct.sort_values("atom_index")

                # Positions
                pos = torch.tensor(
                    mol_struct[["x", "y", "z"]].values, dtype=torch.float
                )

                # Atom Types (Node Features)
                # Map atom symbols to indices
                atom_types = mol_struct["atom"].map(Config.ATOM_MAP).fillna(0).values
                x = torch.tensor(atom_types, dtype=torch.long)  # (Num_atoms,)

                num_atoms = len(x)

                # --- B. Get Auxiliary Targets (Node Level) ---
                # Shielding (Num_atoms, 9)
                if shield_grp and mol_name in shield_grp.groups:
                    s_data = shield_grp.get_group(mol_name).sort_values("atom_index")
                    # Columns XX..ZZ are 2..11
                    s_vals = s_data.iloc[:, 2:11].values
                    y_shield = torch.tensor(s_vals, dtype=torch.float)
                else:
                    y_shield = torch.zeros((num_atoms, 9), dtype=torch.float)

                # Charges (Num_atoms, 1)
                if charge_grp and mol_name in charge_grp.groups:
                    c_data = charge_grp.get_group(mol_name).sort_values("atom_index")
                    c_vals = c_data["mulliken_charge"].values
                    y_charge = torch.tensor(c_vals, dtype=torch.float).unsqueeze(1)
                else:
                    y_charge = torch.zeros((num_atoms, 1), dtype=torch.float)

                # --- C. Construct Atom Graph (Spatial) ---
                # Radius graph: connect atoms within cutoff
                edge_index = radius_graph(
                    pos,
                    r=Config.CUTOFF_RADIUS,
                    max_num_neighbors=Config.MAX_NUM_NEIGHBORS,
                )

                # Calculate Edge Distances (Atom Graph Features)
                row, col = edge_index
                dist = (pos[row] - pos[col]).norm(dim=-1).view(-1, 1)
                edge_attr = dist  # (Num_edges, 1)

                # --- D. Construct Line Graph (Angular) ---
                # Adjacency list for fast lookup
                adj = [[] for _ in range(num_atoms)]
                for idx, (u, v) in enumerate(edge_index.t().tolist()):
                    adj[u].append((v, idx))  # neighbor, edge_index_id

                line_src = []
                line_dst = []
                angles = []

                # Find triplets (i, j, k) where j is center
                for j in range(num_atoms):
                    neighbors = adj[j]
                    # All pairs of neighbors (i, k)
                    for i_idx, (i, e_ji) in enumerate(neighbors):
                        for k_idx, (k, e_jk) in enumerate(neighbors):
                            if i == k:
                                continue  # Skip self-loop angle (0 degrees)

                            # Line Graph Edge: from bond (j, i) to bond (j, k)
                            line_src.append(e_ji)
                            line_dst.append(e_jk)

                            # Calculate Angle i-j-k
                            vec_ji = pos[i] - pos[j]
                            vec_jk = pos[k] - pos[j]

                            # Cosine similarity
                            denom = (vec_ji.norm() * vec_jk.norm()) + 1e-8
                            cos_angle = torch.dot(vec_ji, vec_jk) / denom
                            angles.append(cos_angle)

                if len(line_src) > 0:
                    line_edge_index = torch.tensor(
                        [line_src, line_dst], dtype=torch.long
                    )
                    line_edge_attr = torch.tensor(angles, dtype=torch.float).view(-1, 1)
                else:
                    line_edge_index = torch.empty((2, 0), dtype=torch.long)
                    line_edge_attr = torch.empty((0, 1), dtype=torch.float)

            except KeyError:
                # Fallback for missing structure
                # Ensure num_atoms covers the indices in coupling_indices
                max_idx = max(coupling_atom0.max().item(), coupling_atom1.max().item())
                num_atoms = max_idx + 1

                x = torch.zeros(num_atoms, dtype=torch.long)
                pos = torch.zeros((num_atoms, 3), dtype=torch.float)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float)
                line_edge_index = torch.empty((2, 0), dtype=torch.long)
                line_edge_attr = torch.empty((0, 1), dtype=torch.float)
                y_shield = torch.zeros((num_atoms, 9), dtype=torch.float)
                y_charge = torch.zeros((num_atoms, 1), dtype=torch.float)

            # --- F. Create Data Object ---
            data = MolecularData(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                line_edge_index=line_edge_index,
                line_edge_attr=line_edge_attr,
                y=y,
                coupling_index=coupling_indices,
                coupling_type=coupling_type,
                coupling_id=coupling_id,
                aux_shielding=y_shield,
                aux_charge=y_charge,
                num_atoms=num_atoms,  # Explicitly set for batching
            )
            data_list.append(data)

        # 5. Save
        data, slices = self.collate(data_list)
        print(f"Saving processed {self.mode} data to {self.processed_paths[0]}...")
        torch.save((data, slices), self.processed_paths[0])


def get_data_loaders(
    batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, load_cached_data=True
):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    Also handles TargetScaler fitting.
    """

    # 1. Prepare Target Scaler
    # We need to fit the scaler on the training data.
    # To do this efficiently without loading the full dataset if cached,
    # we check if stats cache exists.
    scaler = TargetScaler(device=Config.DEVICE)
    stats_path = Config.STATS_CACHE

    # If we need to fit, we need the raw metadata
    if not os.path.exists(stats_path) or not load_cached_data:
        print("Fitting TargetScaler from metadata...")
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        scaler.fit(df_train)

        # For auxiliary targets, we need to load the full csvs or sample
        # Since aux files are large, we'll estimate from a sample or load if memory permits
        # Given 220GB RAM, we can load.
        print("Fitting Auxiliary Scaler...")
        df_shield = pd.read_csv(Config.MAGNETIC_SHIELDING_PATH)
        # Shielding columns XX..ZZ
        s_vals = df_shield.iloc[:, 2:11].values.flatten()

        df_charge = pd.read_csv(Config.MULLIKEN_CHARGES_PATH)
        c_vals = df_charge["mulliken_charge"].values

        scaler.fit_auxiliary(s_vals, c_vals)
        scaler.save(stats_path)
    else:
        print("Loading cached TargetScaler...")
        scaler.load(stats_path)

    # 2. Initialize Datasets
    # The Dataset class handles caching internally via 'processed_file_names' check
    # If load_cached_data is False, we can force reprocessing by deleting the file or
    # the user is expected to manage the 'processed' dir.
    # PyG doesn't natively support a 'force_process' flag easily without deleting files.
    # We will assume if the file exists, we use it, unless debug status changes.

    # Train
    train_dataset = MoleculeDataset(
        root=Config.WORK_DIR,
        mode="train",
        metadata_path=Config.TRAIN_META_PATH,
        debug=debug,
    )

    # Val
    val_dataset = MoleculeDataset(
        root=Config.WORK_DIR,
        mode="val",
        metadata_path=Config.VAL_META_PATH,
        debug=debug,
    )

    # Test
    test_dataset = MoleculeDataset(
        root=Config.WORK_DIR,
        mode="test",
        metadata_path=Config.TEST_META_PATH,
        debug=debug,
    )

    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, scaler
