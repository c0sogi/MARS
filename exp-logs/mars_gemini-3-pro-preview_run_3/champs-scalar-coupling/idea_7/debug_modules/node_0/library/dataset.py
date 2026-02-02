import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from library.config import Config
from library.utils import GaussianSmearing

# Mapping for atom types to integers
ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}


class MolecularGraphDataset(InMemoryDataset):
    """
    Graph dataset for Scalar Coupling Prediction.
    Constructs Atom Graphs (distances) and Line Graphs (angles) for S-GLGN.
    """

    def __init__(
        self, mode="train", load_cached_data=True, transform=None, pre_transform=None
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, try to load processed data from disk.
            transform (callable, optional): A function/transform that takes in an
                torch_geometric.data.Data object and returns a transformed version.
            pre_transform (callable, optional): A function/transform that takes in
                an torch_geometric.data.Data object and returns a transformed version.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Set root to working directory to allow writing processed files
        self.root = Config.WORKING_DIR

        # Ensure processed directory exists
        os.makedirs(os.path.join(self.root, "processed"), exist_ok=True)

        super().__init__(self.root, transform, pre_transform)

        # Load data
        processed_path = self.processed_paths[0]
        if self.load_cached_data and os.path.exists(processed_path):
            print(f"Loading cached {self.mode} data from {processed_path}...")
            self.data, self.slices = torch.load(processed_path)
        else:
            print(f"Processing {self.mode} data from scratch...")
            self.process()
            # Reload to ensure consistency and proper attribute setting
            self.data, self.slices = torch.load(processed_path)

    @property
    def raw_file_names(self):
        # We use absolute paths from Config, so this is just a placeholder
        return []

    @property
    def processed_file_names(self):
        return [f"{self.mode}_data.pt"]

    def process(self):
        # 1. Determine Input Files
        if self.mode == "train":
            meta_path = Config.TRAIN_METADATA
        elif self.mode == "val":
            meta_path = Config.VAL_METADATA
        elif self.mode == "test":
            meta_path = Config.TEST_METADATA
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # 2. Load Metadata
        print(f"Loading metadata from {meta_path}...")
        df_meta = pd.read_csv(meta_path)

        # Handle Debug Mode
        if Config.DEBUG:
            print(f"DEBUG MODE: Sampling subset of molecules.")
            mols = df_meta["molecule_name"].unique()
            sample_size = min(len(mols), Config.DEBUG_SAMPLE_SIZE)
            # Use fixed seed for debug consistency
            rng = np.random.RandomState(Config.SEED)
            sample_mols = rng.choice(mols, sample_size, replace=False)
            df_meta = df_meta[df_meta["molecule_name"].isin(sample_mols)].copy()

        # 3. Load Structures
        print("Loading structures...")
        df_struct = pd.read_csv(Config.STRUCTURES_CSV)
        relevant_mols = df_meta["molecule_name"].unique()
        df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)]

        # Group structures for fast access
        struct_group = df_struct.groupby("molecule_name")

        # 4. Load Auxiliary Data (Train/Val only)
        load_aux = self.mode in ["train", "val"]
        shield_group = None
        charge_group = None

        if load_aux:
            print("Loading auxiliary data (Shielding & Charges)...")
            df_shield = pd.read_csv(Config.MAGNETIC_SHIELDING_CSV)
            df_charge = pd.read_csv(Config.MULLIKEN_CHARGES_CSV)

            # Filter
            df_shield = df_shield[df_shield["molecule_name"].isin(relevant_mols)]
            df_charge = df_charge[df_charge["molecule_name"].isin(relevant_mols)]

            shield_group = df_shield.groupby("molecule_name")
            charge_group = df_charge.groupby("molecule_name")

        # 5. Initialize RBF Expanders
        # We use CPU tensors here to avoid GPU OOM during large loop
        dist_rbf = GaussianSmearing(
            start=Config.RBF_MIN_DIST,
            stop=Config.RBF_MAX_DIST,
            num_gaussians=Config.NUM_RBF_DIST,
        )
        angle_rbf = GaussianSmearing(
            start=-1.0, stop=1.0, num_gaussians=Config.NUM_RBF_ANGLE
        )

        data_list = []
        meta_group = df_meta.groupby("molecule_name")

        print("Constructing graphs...")
        for mol_name, group_df in tqdm(meta_group, total=len(meta_group)):
            if mol_name not in struct_group.groups:
                continue

            # --- Node Features & Positions ---
            mol_struct = struct_group.get_group(mol_name).sort_values("atom_index")
            coords = mol_struct[["x", "y", "z"]].values
            atoms = mol_struct["atom"].map(ATOM_MAP).values

            pos = torch.tensor(coords, dtype=torch.float)
            x = torch.tensor(atoms, dtype=torch.long)
            n_atoms = len(atoms)

            # --- Atom Graph (Distance-based) ---
            # Compute pairwise distances
            dist_matrix = torch.cdist(pos, pos)  # (N, N)

            # Create edges: distance < cutoff AND not self-loop
            mask = (dist_matrix < Config.ATOM_GRAPH_CUTOFF) & (dist_matrix > 1e-6)
            row, col = torch.where(mask)
            edge_index = torch.stack([row, col], dim=0)  # (2, E_atom)

            # Edge Attributes: RBF of distance
            distances = dist_matrix[row, col]
            edge_attr = dist_rbf(distances)  # (E_atom, num_rbf_dist)

            # --- Line Graph (Angle-based) ---
            # Nodes in Line Graph = Edges in Atom Graph
            # Edges in Line Graph = Pairs of Atom-Edges (u,v) and (v,w) sharing v

            # Find pairs: col[i] == row[j] AND row[i] != col[j] (no backtracking)
            # This logic connects edge i (u->v) to edge j (v->w)

            # Optimization: Use broadcasting for small graphs
            # num_edges is usually small (< 200)
            num_edges = edge_index.size(1)
            if num_edges > 0:
                # Indices of edges
                edge_indices = torch.arange(num_edges)

                # u, v for all edges
                u_all = row
                v_all = col

                # Expand to find matches
                # We want i, j such that v_all[i] == u_all[j]
                # (E, 1) == (1, E) -> (E, E) boolean matrix
                matches = v_all.unsqueeze(1) == u_all.unsqueeze(0)

                # Exclude backtracking: u_all[i] == v_all[j]
                # If u->v and v->u, then u=u.
                backtracks = u_all.unsqueeze(1) == v_all.unsqueeze(0)

                valid_pairs = matches & (~backtracks)

                # Indices in the edge_index tensor
                src_edge_idx, dst_edge_idx = torch.where(valid_pairs)
                line_edge_index = torch.stack([src_edge_idx, dst_edge_idx], dim=0)

                # Compute Angles
                # Edge i: u->v. Vector v-u? No, standard angle def is vectors FROM central atom.
                # Central atom is v.
                # Vector 1: u - v
                # Vector 2: w - v

                # Get atom indices for the triplets
                idx_u = row[src_edge_idx]
                idx_v = col[src_edge_idx]  # central
                idx_w = col[dst_edge_idx]

                vec1 = pos[idx_u] - pos[idx_v]
                vec2 = pos[idx_w] - pos[idx_v]

                # Normalize
                norm1 = torch.norm(vec1, dim=1, keepdim=True)
                norm2 = torch.norm(vec2, dim=1, keepdim=True)

                # Cosine
                dot_prod = (vec1 * vec2).sum(dim=1, keepdim=True)
                cosine = dot_prod / (norm1 * norm2 + 1e-8)
                cosine = torch.clamp(cosine, -1.0, 1.0)

                line_edge_attr = angle_rbf(cosine)
            else:
                line_edge_index = torch.empty((2, 0), dtype=torch.long)
                line_edge_attr = torch.empty(
                    (0, Config.NUM_RBF_ANGLE), dtype=torch.float
                )

            # --- Targets ---
            # Coupling targets
            target_atom_0 = group_df["atom_index_0"].values
            target_atom_1 = group_df["atom_index_1"].values
            target_types_str = group_df["type"].values

            # Map types
            target_types = torch.tensor(
                [Config.COUPLING_TYPE_MAP[t] for t in target_types_str],
                dtype=torch.long,
            )
            target_edge_index = torch.tensor(
                np.stack([target_atom_0, target_atom_1]), dtype=torch.long
            )

            # Target Values
            if "scalar_coupling_constant" in group_df.columns:
                y = torch.tensor(
                    group_df["scalar_coupling_constant"].values, dtype=torch.float
                )
            else:
                y = torch.zeros(len(target_types), dtype=torch.float)

            # Auxiliary Targets
            if load_aux:
                # Shielding
                if mol_name in shield_group.groups:
                    s_df = shield_group.get_group(mol_name).sort_values("atom_index")
                    s_cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
                    y_shield = torch.tensor(s_df[s_cols].values, dtype=torch.float)
                else:
                    y_shield = torch.zeros((n_atoms, 9), dtype=torch.float)

                # Charges
                if mol_name in charge_group.groups:
                    c_df = charge_group.get_group(mol_name).sort_values("atom_index")
                    y_charge = torch.tensor(
                        c_df["mulliken_charge"].values, dtype=torch.float
                    )
                else:
                    y_charge = torch.zeros((n_atoms), dtype=torch.float)
            else:
                y_shield = torch.zeros((n_atoms, 9), dtype=torch.float)
                y_charge = torch.zeros((n_atoms), dtype=torch.float)

            # Construct Data Object
            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_attr,
                line_edge_index=line_edge_index,
                line_edge_attr=line_edge_attr,
                target_edge_index=target_edge_index,
                target_type=target_types,
                y=y,
                y_shield=y_shield,
                y_charge=y_charge,
                molecule_name=mol_name,
                num_atoms=n_atoms,
            )

            data_list.append(data)

        print(
            f"Saving {len(data_list)} processed graphs to {self.processed_paths[0]}..."
        )
        # Collate and save using standard PyG method
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
