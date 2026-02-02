import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from library.config import Config
from library.utils import (
    GaussianSmearing,
    TargetScaler,
    compute_bond_vectors,
    compute_bond_cosines,
)


class MoleculeDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for Scalar Coupling Prediction.
    Constructs Atom Graphs and Line Graphs for each molecule.
    """

    def __init__(
        self,
        root,
        split="train",
        load_cached_data=True,
        transform=None,
        pre_transform=None,
        debug=False,
    ):
        self.split = split
        self.debug = debug
        self.load_cached_data = load_cached_data

        # Define processed file path based on split
        if split == "train":
            self.processed_path_file = Config.CACHE_TRAIN_PATH
        elif split == "val":
            self.processed_path_file = Config.CACHE_VAL_PATH
        elif split == "test":
            self.processed_path_file = Config.CACHE_TEST_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        # Separate cache for debug mode to avoid overwriting full data
        if self.debug:
            self.processed_path_file = self.processed_path_file.replace(
                ".pt", "_debug.pt"
            )

        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

        # Load data
        if self.load_cached_data and os.path.exists(self.processed_path_file):
            print(f"Loading cached {split} data from {self.processed_path_file}...")
            self.data, self.slices = torch.load(self.processed_path_file)
        else:
            print(f"Processing {split} data from scratch...")
            self.process_data()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.processed_path_file]

    def download(self):
        pass

    def process_data(self):
        # 1. Load Metadata
        if self.split == "train":
            meta_path = Config.TRAIN_META_PATH
        elif self.split == "val":
            meta_path = Config.VAL_META_PATH
        elif self.split == "test":
            meta_path = Config.TEST_META_PATH

        df_meta = pd.read_csv(meta_path)

        # 2. Load Structures
        print("Loading structures...")
        df_struct = pd.read_csv(Config.STRUCTURES_CSV)

        # Filter metadata to ensure referential integrity with structures (Cite debug_lesson_2)
        # We perform this BEFORE sampling in debug mode to ensure we don't sample missing molecules (Cite debug_lesson_3)
        existing_mols = set(df_struct["molecule_name"].unique())
        df_meta = df_meta[df_meta["molecule_name"].isin(existing_mols)]

        if self.debug:
            print(f"DEBUG: Sampling subset of metadata for {self.split}...")
            mols = df_meta["molecule_name"].unique()
            if len(mols) > 0:
                sample_size = min(
                    len(mols), Config.DEBUG_SAMPLE_SIZE // 10
                )  # Approx 10 couplings per mol
                sample_mols = np.random.choice(mols, sample_size, replace=False)
                df_meta = df_meta[df_meta["molecule_name"].isin(sample_mols)].copy()

        # Filter structures to match the (possibly sampled) metadata
        relevant_mols = df_meta["molecule_name"].unique()
        df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)]

        # Group structures by molecule for fast access
        struct_grp = df_struct.groupby("molecule_name")

        # 3. Load Auxiliary Data
        # We load these for all splits. For test data, we fill with zeros if not available.
        print("Loading auxiliary data...")

        # Magnetic Shielding
        try:
            df_shield = pd.read_csv(Config.MAGNETIC_SHIELDING_CSV)
            df_shield = df_shield[df_shield["molecule_name"].isin(relevant_mols)]
            shield_grp = df_shield.groupby("molecule_name")
        except FileNotFoundError:
            shield_grp = None

        # Mulliken Charges
        try:
            df_charge = pd.read_csv(Config.MULLIKEN_CHARGES_CSV)
            df_charge = df_charge[df_charge["molecule_name"].isin(relevant_mols)]
            charge_grp = df_charge.groupby("molecule_name")
        except FileNotFoundError:
            charge_grp = None

        # 4. Prepare Feature Expanders (RBF)
        rbf_radial = GaussianSmearing(
            start=Config.RBF_START,
            stop=Config.RBF_END,
            num_gaussians=Config.NUM_RBF_RADIAL,
        )
        rbf_angular = GaussianSmearing(
            start=-1.0, stop=1.0, num_gaussians=Config.NUM_RBF_ANGULAR
        )

        # 5. Process Molecules
        data_list = []
        meta_grp = df_meta.groupby("molecule_name")
        mol_names = list(meta_grp.groups.keys())

        print(f"Processing {len(mol_names)} molecules for {self.split} set...")

        for mol_name in mol_names:
            # --- A. Atom Graph Construction ---
            mol_struct = struct_grp.get_group(mol_name).sort_values("atom_index")

            # Atom Features
            atoms = mol_struct["atom"].values
            x = torch.tensor([Config.ATOM_TO_IDX[a] for a in atoms], dtype=torch.long)
            pos = torch.tensor(mol_struct[["x", "y", "z"]].values, dtype=torch.float32)
            num_atoms = len(atoms)

            # Edges (Distance based)
            dist_matrix = torch.cdist(pos, pos)
            # Mask: dist < cutoff AND dist > 0 (no self-loops)
            mask = (dist_matrix < Config.RADIUS_CUTOFF) & (dist_matrix > 1e-6)
            row, col = torch.where(mask)
            edge_index = torch.stack([row, col], dim=0)

            # Edge Attributes (Radial RBF)
            edge_vec, edge_dist = compute_bond_vectors(pos, edge_index)
            edge_attr = rbf_radial(edge_dist)  # (E, num_rbf_radial)

            # --- B. Line Graph Construction ---
            # Nodes in Line Graph = Edges in Atom Graph
            # Edges in Line Graph = Pairs of Atom edges (i->j) and (j->k) sharing source j
            src, dst = edge_index

            # Find pairs where src[e1] == src[e2] (sharing the central atom)
            # This creates the connectivity for angular interactions
            # Expand dims: (E, 1) == (1, E) -> (E, E) adjacency
            adj_line = src.unsqueeze(1) == src.unsqueeze(0)
            adj_line.fill_diagonal_(False)  # Remove self-pairs

            line_row, line_col = torch.where(adj_line)
            line_edge_index = torch.stack([line_row, line_col], dim=0)

            # Line Edge Attributes (Angular RBF)
            if line_edge_index.shape[1] > 0:
                cosines = compute_bond_cosines(edge_vec, line_edge_index)
                line_edge_attr = rbf_angular(cosines)  # (E_line, num_rbf_angular)
            else:
                line_edge_attr = torch.empty(
                    (0, Config.NUM_RBF_ANGULAR), dtype=torch.float32
                )

            # --- C. Targets ---
            mol_meta = meta_grp.get_group(mol_name)

            # Coupling Targets (Main Task)
            types_str = mol_meta["type"].values
            types_idx = [Config.COUPLING_TYPES.index(t) for t in types_str]
            type_coupling = torch.tensor(types_idx, dtype=torch.long)

            atom_idx_0 = mol_meta["atom_index_0"].values
            atom_idx_1 = mol_meta["atom_index_1"].values
            edge_index_coupling = torch.tensor(
                np.stack([atom_idx_0, atom_idx_1], axis=0), dtype=torch.long
            )

            if "scalar_coupling_constant" in mol_meta.columns:
                y_coupling = torch.tensor(
                    mol_meta["scalar_coupling_constant"].values, dtype=torch.float32
                )
            else:
                y_coupling = torch.zeros(len(mol_meta), dtype=torch.float32)  # Test set

            id_coupling = torch.tensor(mol_meta["id"].values, dtype=torch.long)

            # Auxiliary Targets
            # Magnetic Shielding (N, 9)
            if shield_grp and mol_name in shield_grp.groups:
                s_data = shield_grp.get_group(mol_name).sort_values("atom_index")
                s_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
                y_shielding = torch.tensor(s_data[s_cols].values, dtype=torch.float32)
            else:
                y_shielding = torch.zeros((num_atoms, 9), dtype=torch.float32)

            # Mulliken Charges (N,)
            if charge_grp and mol_name in charge_grp.groups:
                c_data = charge_grp.get_group(mol_name).sort_values("atom_index")
                y_charge = torch.tensor(
                    c_data["mulliken_charge"].values, dtype=torch.float32
                )
            else:
                y_charge = torch.zeros(num_atoms, dtype=torch.float32)

            # --- D. Create Data Object ---
            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                line_edge_index=line_edge_index,
                line_edge_attr=line_edge_attr,
                edge_index_coupling=edge_index_coupling,
                type_coupling=type_coupling,
                y_coupling=y_coupling,
                id=id_coupling,
                y_shielding=y_shielding,
                y_charge=y_charge,
                num_atoms=num_atoms,
            )
            data_list.append(data)

        # Save processed data
        print(f"Saving {self.split} data to {self.processed_path_file}...")
        torch.save((data_list, None), self.processed_path_file)

        if not data_list:
            raise RuntimeError(
                f"Processed dataset {self.split} is empty. Check structure file integrity."
            )

        self.data, self.slices = self.collate(data_list)


def get_dataloaders(batch_size=Config.BATCH_SIZE, debug=False):
    """
    Creates DataLoaders for train, val, and test splits.
    Fits the TargetScaler on the training metadata to ensure correct standardization stats.
    """
    # 1. Fit Target Scaler (Compute Mean/Std per type)
    print("Initializing TargetScaler...")
    # We use the raw metadata file for fitting to ensure we capture the full distribution
    # even if debug mode loads a subset of graphs.
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    scaler = TargetScaler()
    scaler.fit(df_train_meta)

    # 2. Create Datasets
    train_dataset = MoleculeDataset(root=Config.WORKING_DIR, split="train", debug=debug)
    val_dataset = MoleculeDataset(root=Config.WORKING_DIR, split="val", debug=debug)
    test_dataset = MoleculeDataset(root=Config.WORKING_DIR, split="test", debug=debug)

    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
