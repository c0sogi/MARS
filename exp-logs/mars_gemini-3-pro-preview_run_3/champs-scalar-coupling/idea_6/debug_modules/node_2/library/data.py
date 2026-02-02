import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from library.config import Config


class MolecularGraphDataset(InMemoryDataset):
    """
    Graph dataset for scalar coupling prediction.
    Constructs a dual-graph representation:
    1. Atom Graph: Nodes=Atoms, Edges=Distances (cutoff based)
    2. Line Graph: Nodes=Bonds, Edges=Angles (triplets)
    """

    def __init__(
        self,
        root,
        mode="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        load_cached_data=True,
    ):
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Mappings
        self.atom_map = {k: v for v, k in enumerate(Config.ATOM_TYPES)}
        self.type_map = {k: v for v, k in enumerate(Config.COUPLING_TYPES)}

        # Override processed directory to use the cache directory defined in Config
        self._processed_dir = Config.PROCESSED_CACHE_DIR

        super().__init__(root, transform, pre_transform, pre_filter)

        # Caching Logic
        if self.load_cached_data and os.path.exists(self.processed_paths[0]):
            try:
                self.data, self.slices = torch.load(self.processed_paths[0])
            except Exception as e:
                print(
                    f"Failed to load cached data for {self.mode}: {e}. Reprocessing..."
                )
                self.process()
        else:
            self.process()

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def raw_file_names(self):
        # Not used directly as we load from Config paths
        return []

    @property
    def processed_file_names(self):
        return [f"{self.mode}_data.pt"]

    def download(self):
        pass

    def process(self):
        print(f"Processing {self.mode} dataset...")

        # 1. Load Metadata
        if self.mode == "train":
            df_meta = pd.read_csv(Config.TRAIN_META_PATH)
        elif self.mode == "val":
            df_meta = pd.read_csv(Config.VAL_META_PATH)
        elif self.mode == "test":
            df_meta = pd.read_csv(Config.TEST_META_PATH)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Debug Slicing
        if Config.DEBUG:
            print(
                f"DEBUG: Slicing {self.mode} data to {Config.DEBUG_SAMPLE_SIZE} molecules."
            )
            unique_mols = df_meta["molecule_name"].unique()
            if len(unique_mols) > Config.DEBUG_SAMPLE_SIZE:
                # Use random choice to avoid hitting blocks of missing data
                selected_mols = np.random.choice(
                    unique_mols, Config.DEBUG_SAMPLE_SIZE, replace=False
                )
                df_meta = df_meta[df_meta["molecule_name"].isin(selected_mols)].copy()

        relevant_mols = df_meta["molecule_name"].unique()

        # 2. Load Structures
        print("Loading structures...")
        df_structures = pd.read_csv(Config.STRUCTURES_CSV)
        df_structures = df_structures[
            df_structures["molecule_name"].isin(relevant_mols)
        ]
        # Group by molecule_name for O(1) access
        structures_grp = df_structures.groupby("molecule_name")

        # 3. Load Auxiliary Data (Train/Val only)
        mulliken_grp = None
        shielding_grp = None
        if self.mode in ["train", "val"]:
            print("Loading auxiliary targets...")
            df_mull = pd.read_csv(Config.MULLIKEN_CHARGES_CSV)
            df_mull = df_mull[df_mull["molecule_name"].isin(relevant_mols)]
            mulliken_grp = df_mull.groupby("molecule_name")

            df_shield = pd.read_csv(Config.MAGNETIC_SHIELDING_CSV)
            df_shield = df_shield[df_shield["molecule_name"].isin(relevant_mols)]
            shielding_grp = df_shield.groupby("molecule_name")

        # 4. Group Metadata (Couplings)
        meta_grp = df_meta.groupby("molecule_name")

        data_list = []
        print(f"Constructing graphs for {len(relevant_mols)} molecules...")

        for mol_name in tqdm(relevant_mols):
            # --- Structure ---
            try:
                mol_struct = structures_grp.get_group(mol_name).sort_values(
                    "atom_index"
                )
            except KeyError:
                continue

            pos = mol_struct[["x", "y", "z"]].values.astype(np.float32)
            atoms = mol_struct["atom"].values
            num_atoms = len(atoms)

            # Node Features: Atom Type Index
            x = torch.tensor([self.atom_map[a] for a in atoms], dtype=torch.long)
            pos_t = torch.tensor(pos, dtype=torch.float32)

            # --- Atom Graph (Radius Graph) ---
            # Calculate pairwise distances
            dist_matrix = torch.cdist(pos_t, pos_t)
            # Create edges for atoms within cutoff (excluding self-loops)
            mask = (dist_matrix < Config.RBF_CUTOFF) & (dist_matrix > 1e-6)
            edge_index = mask.nonzero(as_tuple=False).t()

            row, col = edge_index
            edge_attr = dist_matrix[row, col].unsqueeze(-1)  # Distance feature

            # --- Line Graph (Angles) ---
            # Nodes in Line Graph are edges of Atom Graph (indices 0 to E-1).
            # Edges in Line Graph connect adjacent bonds (u->v and v->w).

            # Use pandas for efficient join on small graphs
            edges_df = pd.DataFrame(
                {"u": row.numpy(), "v": col.numpy(), "idx": np.arange(len(row))}
            )

            # Join on v == u (head matches tail)
            triplets = pd.merge(
                edges_df, edges_df, left_on="v", right_on="u", suffixes=("_1", "_2")
            )
            # Remove backtracking (u -> v -> u)
            triplets = triplets[triplets["u_1"] != triplets["v_2"]]

            if len(triplets) > 0:
                line_src = torch.tensor(triplets["idx_1"].values, dtype=torch.long)
                line_dst = torch.tensor(triplets["idx_2"].values, dtype=torch.long)
                line_edge_index = torch.stack([line_src, line_dst], dim=0)

                # Calculate Cosine Angle
                # Bond 1: v -> u (vector = pos[u] - pos[v])
                # Bond 2: v -> w (vector = pos[w] - pos[v])
                u_idx = triplets["u_1"].values
                v_idx = triplets["v_1"].values
                w_idx = triplets["v_2"].values

                vec1 = pos_t[u_idx] - pos_t[v_idx]
                vec2 = pos_t[w_idx] - pos_t[v_idx]

                norm1 = torch.norm(vec1, dim=1, keepdim=True)
                norm2 = torch.norm(vec2, dim=1, keepdim=True)

                dot = (vec1 * vec2).sum(dim=1, keepdim=True)
                cosine = dot / (norm1 * norm2 + 1e-7)
                # Clamp for numerical stability
                cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

                line_edge_attr = cosine
            else:
                line_edge_index = torch.empty((2, 0), dtype=torch.long)
                line_edge_attr = torch.empty((0, 1), dtype=torch.float32)

            # --- Coupling Targets ---
            mol_couplings = meta_grp.get_group(mol_name)

            # Coupling Indices (Atom Pairs)
            c_idx = torch.tensor(
                mol_couplings[["atom_index_0", "atom_index_1"]].values.T,
                dtype=torch.long,
            )

            # Coupling Types
            c_types = torch.tensor(
                [self.type_map[t] for t in mol_couplings["type"].values],
                dtype=torch.long,
            )

            # Coupling Values
            if "scalar_coupling_constant" in mol_couplings.columns:
                c_vals = torch.tensor(
                    mol_couplings["scalar_coupling_constant"].values,
                    dtype=torch.float32,
                )
            else:
                c_vals = torch.zeros(len(mol_couplings), dtype=torch.float32)

            # Coupling IDs (for submission)
            if "id" in mol_couplings.columns:
                c_ids = torch.tensor(mol_couplings["id"].values, dtype=torch.long)
            else:
                c_ids = torch.zeros(len(mol_couplings), dtype=torch.long)

            # --- Auxiliary Targets ---
            if self.mode in ["train", "val"]:
                # Mulliken Charges
                try:
                    m_data = mulliken_grp.get_group(mol_name).sort_values("atom_index")
                    y_mull = torch.tensor(
                        m_data["mulliken_charge"].values, dtype=torch.float32
                    )
                except KeyError:
                    y_mull = torch.zeros(num_atoms, dtype=torch.float32)

                # Shielding Tensors
                try:
                    s_data = shielding_grp.get_group(mol_name).sort_values("atom_index")
                    cols = ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
                    y_shield = torch.tensor(s_data[cols].values, dtype=torch.float32)
                except KeyError:
                    y_shield = torch.zeros((num_atoms, 9), dtype=torch.float32)
            else:
                y_mull = torch.zeros(num_atoms, dtype=torch.float32)
                y_shield = torch.zeros((num_atoms, 9), dtype=torch.float32)

            data = Data(
                x=x,
                pos=pos_t,
                edge_index=edge_index,
                edge_attr=edge_attr,
                line_edge_index=line_edge_index,
                line_edge_attr=line_edge_attr,
                coupling_atom_index=c_idx,
                coupling_type=c_types,
                coupling_value=c_vals,
                coupling_id=c_ids,
                y_mulliken=y_mull,
                y_shielding=y_shield,
                num_atoms=num_atoms,
                molecule_name=mol_name,
            )
            data_list.append(data)

        if not data_list:
            raise RuntimeError(
                f"No valid graphs constructed for {self.mode} dataset. "
                "Check structure files or debug sampling."
            )

        print(f"Saving processed {self.mode} data...")
        torch.save(self.collate(data_list), self.processed_paths[0])
        self.data, self.slices = self.collate(data_list)


def get_datasets(load_cached_data=True):
    """
    Factory function to initialize train, val, and test datasets.
    """
    os.makedirs(Config.PROCESSED_CACHE_DIR, exist_ok=True)

    train_ds = MolecularGraphDataset(
        Config.WORKING_DIR, mode="train", load_cached_data=load_cached_data
    )
    val_ds = MolecularGraphDataset(
        Config.WORKING_DIR, mode="val", load_cached_data=load_cached_data
    )
    test_ds = MolecularGraphDataset(
        Config.WORKING_DIR, mode="test", load_cached_data=load_cached_data
    )

    return train_ds, val_ds, test_ds
