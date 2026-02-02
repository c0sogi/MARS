import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
from library.config import Config


class TargetStandardizer:
    """
    Handles standardization of primary coupling targets and auxiliary targets.
    Persists statistics to disk to ensure consistency between training and inference.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.stats = {}
        self.stats_path = os.path.join(Config.WORKING_DIR, "stats.npy")

    def fit(self, df_train, aux_shielding=None, aux_charge=None):
        """
        Compute mean and std for coupling types and auxiliary targets.

        Args:
            df_train (pd.DataFrame): Training metadata containing 'type' and 'scalar_coupling_constant'.
            aux_shielding (np.ndarray): Array of shape (N, 9) for shielding tensors.
            aux_charge (np.ndarray): Array of shape (N,) for mulliken charges.
        """
        # Primary Target Statistics per Coupling Type
        for ctype in Config.COUPLING_TYPES:
            subset = df_train[df_train["type"] == ctype]
            vals = subset["scalar_coupling_constant"].values
            self.stats[f"mean_{ctype}"] = float(np.mean(vals))
            self.stats[f"std_{ctype}"] = float(np.std(vals))

        # Auxiliary Statistics
        if aux_shielding is not None:
            self.stats["mean_shielding"] = np.mean(aux_shielding, axis=0)  # (9,)
            self.stats["std_shielding"] = np.std(aux_shielding, axis=0)  # (9,)

        if aux_charge is not None:
            self.stats["mean_charge"] = float(np.mean(aux_charge))
            self.stats["std_charge"] = float(np.std(aux_charge))

        # Save statistics
        np.save(self.stats_path, self.stats)

    def load(self):
        """Load statistics from disk."""
        if os.path.exists(self.stats_path):
            loaded = np.load(self.stats_path, allow_pickle=True).item()
            self.stats = loaded
        else:
            # Fallback or error if not found (should be fit first)
            pass

    def transform(self, values, types):
        """
        Standardize coupling constants based on their type.
        values: Tensor (N,)
        types: Tensor (N,) of integer type indices
        """
        # Create output tensor
        out = torch.zeros_like(values)

        for i, ctype in enumerate(Config.COUPLING_TYPES):
            type_idx = Config.TYPE_MAP[ctype]
            mask = types == type_idx
            if mask.sum() > 0:
                mu = self.stats.get(f"mean_{ctype}", 0.0)
                sigma = self.stats.get(f"std_{ctype}", 1.0)
                out[mask] = (values[mask] - mu) / sigma
        return out

    def inverse_transform(self, values, types):
        """
        Inverse standardize predictions to original scale.
        """
        out = torch.zeros_like(values)
        for i, ctype in enumerate(Config.COUPLING_TYPES):
            type_idx = Config.TYPE_MAP[ctype]
            mask = types == type_idx
            if mask.sum() > 0:
                mu = self.stats.get(f"mean_{ctype}", 0.0)
                sigma = self.stats.get(f"std_{ctype}", 1.0)
                out[mask] = (values[mask] * sigma) + mu
        return out

    def transform_aux(self, shielding, charge):
        """
        Standardize auxiliary targets.
        """
        s_mean = torch.tensor(
            self.stats.get("mean_shielding", np.zeros(9)),
            device=shielding.device,
            dtype=shielding.dtype,
        )
        s_std = torch.tensor(
            self.stats.get("std_shielding", np.ones(9)),
            device=shielding.device,
            dtype=shielding.dtype,
        )

        c_mean = self.stats.get("mean_charge", 0.0)
        c_std = self.stats.get("std_charge", 1.0)

        s_out = (shielding - s_mean) / s_std
        c_out = (charge - c_mean) / c_std

        return s_out, c_out


class SoADataset(Dataset):
    """
    Structure-of-Arrays Dataset.
    Loads monolithic numpy arrays and slices them on-the-fly to produce graph objects.
    """

    def __init__(self, split, mode="train"):
        self.split = split
        self.mode = mode
        self.data_dir = os.path.join(Config.WORKING_DIR, "processed")

        # Load index map
        self.mol_map = np.load(
            os.path.join(self.data_dir, f"{split}_mol_map.npy"), allow_pickle=True
        ).item()
        self.mol_names = list(self.mol_map.keys())

        # Load monolithic arrays (memory mapped for efficiency if large, but here we load to RAM)
        # Using mmap_mode='r' allows us to not load everything if RAM is tight,
        # but 220GB RAM is plenty for this dataset.
        self.node_types = np.load(
            os.path.join(self.data_dir, f"{split}_node_types.npy")
        )
        self.node_coords = np.load(
            os.path.join(self.data_dir, f"{split}_node_coords.npy")
        )

        self.edge_indices = np.load(
            os.path.join(self.data_dir, f"{split}_edge_indices.npy")
        )
        self.edge_dists = np.load(
            os.path.join(self.data_dir, f"{split}_edge_dists.npy")
        )

        self.coupling_pairs = np.load(
            os.path.join(self.data_dir, f"{split}_coupling_pairs.npy")
        )
        self.coupling_types = np.load(
            os.path.join(self.data_dir, f"{split}_coupling_types.npy")
        )
        self.coupling_values = np.load(
            os.path.join(self.data_dir, f"{split}_coupling_values.npy")
        )
        self.coupling_ids = np.load(
            os.path.join(self.data_dir, f"{split}_coupling_ids.npy")
        )

        if mode != "test":
            self.aux_shielding = np.load(
                os.path.join(self.data_dir, f"{split}_aux_shielding.npy")
            )
            self.aux_charge = np.load(
                os.path.join(self.data_dir, f"{split}_aux_charge.npy")
            )
        else:
            self.aux_shielding = None
            self.aux_charge = None

    def __len__(self):
        return len(self.mol_names)

    def __getitem__(self, idx):
        mol_name = self.mol_names[idx]
        indices = self.mol_map[mol_name]

        # Unpack indices
        n_start, n_cnt = indices["node"]
        e_start, e_cnt = indices["edge"]
        c_start, c_cnt = indices["coupling"]

        # Slice Data
        # Nodes
        z = torch.from_numpy(self.node_types[n_start : n_start + n_cnt]).long()
        pos = torch.from_numpy(self.node_coords[n_start : n_start + n_cnt]).float()

        # Edges
        # edge_index stored as (2, E), we need to slice columns
        edge_index = torch.from_numpy(
            self.edge_indices[:, e_start : e_start + e_cnt]
        ).long()
        edge_attr = (
            torch.from_numpy(self.edge_dists[e_start : e_start + e_cnt])
            .float()
            .unsqueeze(-1)
        )

        # Couplings
        # coupling_pairs stored as (2, C)
        coupling_index = torch.from_numpy(
            self.coupling_pairs[:, c_start : c_start + c_cnt]
        ).long()
        coupling_type = torch.from_numpy(
            self.coupling_types[c_start : c_start + c_cnt]
        ).long()
        coupling_value = torch.from_numpy(
            self.coupling_values[c_start : c_start + c_cnt]
        ).float()
        coupling_id = torch.from_numpy(
            self.coupling_ids[c_start : c_start + c_cnt]
        ).long()

        # Aux
        if self.aux_shielding is not None:
            aux_s = torch.from_numpy(
                self.aux_shielding[n_start : n_start + n_cnt]
            ).float()
            aux_c = torch.from_numpy(self.aux_charge[n_start : n_start + n_cnt]).float()
        else:
            # For test set, return zeros matching node count
            aux_s = torch.zeros((n_cnt, 9), dtype=torch.float)
            aux_c = torch.zeros((n_cnt,), dtype=torch.float)

        return {
            "z": z,
            "pos": pos,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "coupling_index": coupling_index,
            "coupling_type": coupling_type,
            "y": coupling_value,
            "id": coupling_id,
            "aux_s": aux_s,
            "aux_c": aux_c,
            "num_nodes": n_cnt,
        }


def collate_fn(batch):
    """
    Custom collate function to batch graphs.
    Shifts indices (edge_index, coupling_index) by cumulative node counts.
    """
    z_list = []
    pos_list = []
    edge_index_list = []
    edge_attr_list = []
    coupling_index_list = []
    coupling_type_list = []
    y_list = []
    id_list = []
    aux_s_list = []
    aux_c_list = []

    node_offset = 0
    batch_idx_list = []

    for i, data in enumerate(batch):
        num_nodes = data["num_nodes"]

        z_list.append(data["z"])
        pos_list.append(data["pos"])

        # Shift edge indices
        edge_index_list.append(data["edge_index"] + node_offset)
        edge_attr_list.append(data["edge_attr"])

        # Shift coupling indices
        coupling_index_list.append(data["coupling_index"] + node_offset)
        coupling_type_list.append(data["coupling_type"])
        y_list.append(data["y"])
        id_list.append(data["id"])

        aux_s_list.append(data["aux_s"])
        aux_c_list.append(data["aux_c"])

        # Create batch index for nodes
        batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))

        node_offset += num_nodes

    return {
        "z": torch.cat(z_list, dim=0),
        "pos": torch.cat(pos_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "coupling_index": torch.cat(coupling_index_list, dim=1),
        "coupling_type": torch.cat(coupling_type_list, dim=0),
        "y": torch.cat(y_list, dim=0),
        "id": torch.cat(id_list, dim=0),
        "aux_s": torch.cat(aux_s_list, dim=0),
        "aux_c": torch.cat(aux_c_list, dim=0),
        "batch": torch.cat(batch_idx_list, dim=0),
    }


def preprocess_dataset(split, load_cached_data=True):
    """
    Preprocesses the dataset for a given split (train/val/test).
    Generates radius graphs and stores them in flattened numpy arrays.
    """
    processed_dir = os.path.join(Config.WORKING_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    flag_file = os.path.join(processed_dir, f"{split}_completed.flag")

    if load_cached_data and os.path.exists(flag_file):
        return

    print(f"Preprocessing {split} set...")

    # 1. Load Metadata
    if split == "train":
        meta_path = Config.TRAIN_META_PATH
    elif split == "val":
        meta_path = Config.VAL_META_PATH
    else:
        meta_path = Config.TEST_META_PATH

    df_meta = pd.read_csv(meta_path)

    # 2. Load Structures
    # We load all structures but filter for molecules in this split
    df_struct = pd.read_csv(Config.STRUCTURES_CSV)
    relevant_mols = df_meta["molecule_name"].unique()
    df_struct = df_struct[df_struct["molecule_name"].isin(relevant_mols)]

    # 3. Load Aux Data (Only for train/val if available, else ignored)
    # Note: We use aux data for val to compute val loss, but not for test.
    has_aux = split != "test"
    if has_aux:
        df_shield = pd.read_csv(Config.MAGNETIC_SHIELDING_CSV)
        df_charge = pd.read_csv(Config.MULLIKEN_CHARGES_CSV)
        # Filter
        df_shield = df_shield[df_shield["molecule_name"].isin(relevant_mols)]
        df_charge = df_charge[df_charge["molecule_name"].isin(relevant_mols)]

    # 4. Prepare Lists for SoA construction
    # We will build lists then concat to numpy

    # Sort structures to ensure contiguous atoms per molecule
    df_struct = df_struct.sort_values(["molecule_name", "atom_index"])

    # Grouping
    struct_grp = df_struct.groupby("molecule_name")
    meta_grp = df_meta.groupby("molecule_name")

    if has_aux:
        shield_grp = df_shield.groupby("molecule_name")
        charge_grp = df_charge.groupby("molecule_name")

    # Containers
    all_node_types = []
    all_node_coords = []
    all_edge_indices = []
    all_edge_dists = []
    all_coupling_pairs = []
    all_coupling_types = []
    all_coupling_values = []
    all_coupling_ids = []
    all_aux_shielding = []
    all_aux_charge = []

    mol_map = (
        {}
    )  # name -> {node: (start, len), edge: (start, len), coupling: (start, len)}

    node_offset_counter = 0
    edge_offset_counter = 0
    coupling_offset_counter = 0

    # Iterate over molecules
    # We iterate over relevant_mols to maintain order
    for mol_name in relevant_mols:
        # -- Nodes --
        if mol_name in struct_grp.groups:
            mol_struct = struct_grp.get_group(mol_name)
            atoms = mol_struct["atom"].map(Config.ATOM_MAP).values.astype(np.int64)
            coords = mol_struct[["x", "y", "z"]].values.astype(np.float32)
        else:
            # Fallback for missing structures (Cite debug_lesson_2)
            # Ensure dummy structure has enough atoms to cover indices in metadata (Cite debug_lesson_14)
            n_dummy = 1
            if mol_name in meta_grp.groups:
                mol_meta = meta_grp.get_group(mol_name)
                if "atom_index_0" in mol_meta.columns:
                    max_idx = max(
                        mol_meta["atom_index_0"].max(), mol_meta["atom_index_1"].max()
                    )
                    n_dummy = int(max_idx) + 1

            # Create dummy atoms (Type 0=H, Coords 0,0,0)
            atoms = np.zeros(n_dummy, dtype=np.int64)
            coords = np.zeros((n_dummy, 3), dtype=np.float32)

        n_atoms = len(atoms)

        all_node_types.append(atoms)
        all_node_coords.append(coords)

        # -- Edges (Radius Graph) --
        # Brute force distance for small molecules is fast
        dists = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
        # Mask: dist < cutoff and not self loop
        mask = (dists < Config.CUTOFF) & (dists > 1e-6)
        src, dst = np.where(mask)
        edge_d = dists[src, dst]

        n_edges = len(src)
        all_edge_indices.append(np.stack([src, dst], axis=0))
        all_edge_dists.append(edge_d)

        # -- Couplings --
        if mol_name in meta_grp.groups:
            mol_meta = meta_grp.get_group(mol_name)
            c_idx0 = mol_meta["atom_index_0"].values
            c_idx1 = mol_meta["atom_index_1"].values
            c_types = mol_meta["type"].map(Config.TYPE_MAP).values
            c_ids = mol_meta["id"].values

            if "scalar_coupling_constant" in mol_meta.columns:
                c_vals = mol_meta["scalar_coupling_constant"].values.astype(np.float32)
            else:
                c_vals = np.zeros(len(c_idx0), dtype=np.float32)

            n_couplings = len(c_idx0)
            all_coupling_pairs.append(np.stack([c_idx0, c_idx1], axis=0))
            all_coupling_types.append(c_types)
            all_coupling_values.append(c_vals)
            all_coupling_ids.append(c_ids)
        else:
            n_couplings = 0
            # Append empty arrays to keep types consistent if needed,
            # but usually every molecule in metadata has couplings.
            # Just in case:
            all_coupling_pairs.append(np.zeros((2, 0), dtype=np.int64))
            all_coupling_types.append(np.zeros((0,), dtype=np.int64))
            all_coupling_values.append(np.zeros((0,), dtype=np.float32))
            all_coupling_ids.append(np.zeros((0,), dtype=np.int64))

        # -- Aux --
        if has_aux:
            # Shielding
            # Columns XX..ZZ are 3..12 in csv?
            # CSV cols: molecule_name, atom_index, XX, YX, ZX, XY, YY, ZY, XZ, YZ, ZZ
            # We assume sorted by atom_index same as structures.
            mol_shield = shield_grp.get_group(mol_name)
            # Ensure sorting
            mol_shield = mol_shield.sort_values("atom_index")
            s_vals = mol_shield[
                ["XX", "YX", "ZX", "XY", "YY", "ZY", "XZ", "YZ", "ZZ"]
            ].values.astype(np.float32)

            # Charge
            mol_charge = charge_grp.get_group(mol_name)
            mol_charge = mol_charge.sort_values("atom_index")
            ch_vals = mol_charge["mulliken_charge"].values.astype(np.float32)

            all_aux_shielding.append(s_vals)
            all_aux_charge.append(ch_vals)

        # -- Map --
        mol_map[mol_name] = {
            "node": (node_offset_counter, n_atoms),
            "edge": (edge_offset_counter, n_edges),
            "coupling": (coupling_offset_counter, n_couplings),
        }

        node_offset_counter += n_atoms
        edge_offset_counter += n_edges
        coupling_offset_counter += n_couplings

    # 5. Concatenate and Save
    print(f"Saving {split} arrays...")
    np.save(
        os.path.join(processed_dir, f"{split}_node_types.npy"),
        np.concatenate(all_node_types),
    )
    np.save(
        os.path.join(processed_dir, f"{split}_node_coords.npy"),
        np.concatenate(all_node_coords),
    )

    np.save(
        os.path.join(processed_dir, f"{split}_edge_indices.npy"),
        np.concatenate(all_edge_indices, axis=1),
    )
    np.save(
        os.path.join(processed_dir, f"{split}_edge_dists.npy"),
        np.concatenate(all_edge_dists),
    )

    np.save(
        os.path.join(processed_dir, f"{split}_coupling_pairs.npy"),
        np.concatenate(all_coupling_pairs, axis=1),
    )
    np.save(
        os.path.join(processed_dir, f"{split}_coupling_types.npy"),
        np.concatenate(all_coupling_types),
    )
    np.save(
        os.path.join(processed_dir, f"{split}_coupling_values.npy"),
        np.concatenate(all_coupling_values),
    )
    np.save(
        os.path.join(processed_dir, f"{split}_coupling_ids.npy"),
        np.concatenate(all_coupling_ids),
    )

    if has_aux:
        np.save(
            os.path.join(processed_dir, f"{split}_aux_shielding.npy"),
            np.concatenate(all_aux_shielding),
        )
        np.save(
            os.path.join(processed_dir, f"{split}_aux_charge.npy"),
            np.concatenate(all_aux_charge),
        )

    np.save(os.path.join(processed_dir, f"{split}_mol_map.npy"), mol_map)

    # Create flag
    with open(flag_file, "w") as f:
        f.write("done")


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    Handles preprocessing and standardization fitting.
    """
    # 1. Preprocess
    preprocess_dataset("train", load_cached_data)
    preprocess_dataset("val", load_cached_data)
    preprocess_dataset("test", load_cached_data)

    # 2. Fit Standardizer (if not exists)
    standardizer = TargetStandardizer()
    if not os.path.exists(standardizer.stats_path) or not load_cached_data:
        print("Fitting Standardizer on Train data...")
        # Load train arrays to compute stats
        data_dir = os.path.join(Config.WORKING_DIR, "processed")

        # Load metadata df for coupling types/targets
        df_train = pd.read_csv(Config.TRAIN_META_PATH)

        # Load aux arrays
        aux_s = np.load(os.path.join(data_dir, "train_aux_shielding.npy"))
        aux_c = np.load(os.path.join(data_dir, "train_aux_charge.npy"))

        standardizer.fit(df_train, aux_s, aux_c)
    else:
        standardizer.load()

    # 3. Create Datasets
    train_ds = SoADataset("train", mode="train")
    val_ds = SoADataset("val", mode="val")
    test_ds = SoADataset("test", mode="test")

    # 4. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, standardizer
