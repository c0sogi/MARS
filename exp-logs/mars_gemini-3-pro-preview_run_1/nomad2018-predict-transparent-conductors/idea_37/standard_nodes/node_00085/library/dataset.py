import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.geometry_processor import process_dataset


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for material structures.
    """

    def __init__(self, node_feats_list, global_feats, targets=None, ids=None):
        """
        Args:
            node_feats_list (list of np.ndarray): List where each element is (N_atoms, 9) features for a crystal.
            global_feats (np.ndarray): (N_samples, 12) global features.
            targets (np.ndarray, optional): (N_samples, 2) targets.
            ids (np.ndarray, optional): (N_samples,) IDs.
        """
        self.node_feats_list = node_feats_list
        self.global_feats = torch.tensor(global_feats, dtype=torch.float32)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        if ids is not None:
            self.ids = ids
        else:
            self.ids = None

    def __len__(self):
        return len(self.node_feats_list)

    def __getitem__(self, idx):
        # Convert specific crystal's atoms to tensor
        atom_feats = torch.tensor(self.node_feats_list[idx], dtype=torch.float32)
        global_feat = self.global_feats[idx]

        target = self.targets[idx] if self.targets is not None else torch.zeros(2)
        sample_id = self.ids[idx] if self.ids is not None else -1

        return atom_feats, global_feat, target, sample_id


def collate_materials(batch):
    """
    Collate function to handle variable number of atoms per crystal.

    Args:
        batch: List of tuples (atom_feats, global_feat, target, sample_id)

    Returns:
        batch_atom_feats: (Total_Batch_Atoms, Atom_Feat_Dim)
        batch_indices: (Total_Batch_Atoms,) mapping atoms to sample index in batch
        batch_global_feats: (Batch_Size, Global_Feat_Dim)
        batch_targets: (Batch_Size, Target_Dim)
        batch_ids: (Batch_Size,)
    """
    atom_feats_list = []
    batch_indices_list = []
    global_feats_list = []
    targets_list = []
    ids_list = []

    for i, (atoms, glob, target, sid) in enumerate(batch):
        n_atoms = atoms.shape[0]

        atom_feats_list.append(atoms)
        # Create an index vector [i, i, ..., i] for this sample
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        global_feats_list.append(glob)
        targets_list.append(target)
        ids_list.append(sid)

    # Concatenate all atoms into one big tensor
    batch_atom_feats = torch.cat(atom_feats_list, dim=0)
    batch_indices = torch.cat(batch_indices_list, dim=0)

    # Stack globals and targets
    batch_global_feats = torch.stack(global_feats_list, dim=0)
    batch_targets = torch.stack(targets_list, dim=0)
    batch_ids = torch.tensor(ids_list, dtype=torch.long)

    return batch_atom_feats, batch_indices, batch_global_feats, batch_targets, batch_ids


class MaterialScaler:
    """
    Wrapper for scaling atomic and global features.
    Scales continuous atomic features (indices 4-8) and all global features.
    """

    def __init__(self):
        self.atom_scaler = StandardScaler()
        self.global_scaler = StandardScaler()

    def fit(self, node_feats_list, global_feats):
        # Flatten node feats list for fitting
        all_nodes = np.vstack(node_feats_list)
        # Scale only continuous parts: coords (4,5,6) and shells (7,8)
        # Indices 0-3 are one-hot encodings
        self.atom_scaler.fit(all_nodes[:, 4:])
        self.global_scaler.fit(global_feats)

    def transform(self, node_feats_list, global_feats):
        # Transform global
        scaled_global = self.global_scaler.transform(global_feats)

        # Transform nodes
        scaled_nodes_list = []
        for nodes in node_feats_list:
            nodes_copy = nodes.copy()
            nodes_copy[:, 4:] = self.atom_scaler.transform(nodes[:, 4:])
            scaled_nodes_list.append(nodes_copy)

        return scaled_nodes_list, scaled_global

    def save(self, path):
        np.savez(
            path,
            atom_mean=self.atom_scaler.mean_,
            atom_scale=self.atom_scaler.scale_,
            global_mean=self.global_scaler.mean_,
            global_scale=self.global_scaler.scale_,
        )

    def load(self, path):
        data = np.load(path)
        self.atom_scaler.mean_ = data["atom_mean"]
        self.atom_scaler.scale_ = data["atom_scale"]
        self.global_scaler.mean_ = data["global_mean"]
        self.global_scaler.scale_ = data["global_scale"]


def _reconstruct_list_from_flat(flat_nodes, batch_indices, num_samples):
    """
    Helper to convert flat node array back to list of arrays per sample.
    """
    node_list = []
    # We assume batch_indices are sorted 0..N-1, which they are from process_dataset
    # But to be safe, we can use split.
    # However, np.split requires split points.

    # Fast method assuming sorted indices:
    # Find change points
    if len(batch_indices) == 0:
        return [np.zeros((0, flat_nodes.shape[1])) for _ in range(num_samples)]

    # Count atoms per sample
    counts = np.bincount(batch_indices, minlength=num_samples)

    # Create split indices
    split_indices = np.cumsum(counts)[:-1]
    node_list = np.split(flat_nodes, split_indices)

    return node_list


def prepare_data(load_cached_data=True):
    """
    Main function to prepare DataLoaders.

    1. Loads/Computes features via geometry_processor.
    2. Fits scalers on Train.
    3. Transforms Train, Val, Test.
    4. Log-transforms targets.
    5. Returns DataLoaders.
    """
    Config.setup_directories()

    # 1. Process Data
    train_raw = process_dataset(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_DATA_PATH, load_cached_data
    )
    val_raw = process_dataset(
        Config.VAL_METADATA_PATH, Config.VAL_DATA_PATH, load_cached_data
    )
    test_raw = process_dataset(
        Config.TEST_METADATA_PATH, Config.TEST_DATA_PATH, load_cached_data
    )

    # Reconstruct lists from flat arrays for Dataset compatibility
    train_nodes_list = _reconstruct_list_from_flat(
        train_raw["node_feats"], train_raw["batch_indices"], len(train_raw["ids"])
    )
    val_nodes_list = _reconstruct_list_from_flat(
        val_raw["node_feats"], val_raw["batch_indices"], len(val_raw["ids"])
    )
    test_nodes_list = _reconstruct_list_from_flat(
        test_raw["node_feats"], test_raw["batch_indices"], len(test_raw["ids"])
    )

    # 2. Scaling
    scaler = MaterialScaler()

    if load_cached_data and os.path.exists(Config.SCALERS_PATH):
        print(f"Loading scalers from {Config.SCALERS_PATH}...")
        scaler.load(Config.SCALERS_PATH)
    else:
        print("Fitting scalers on training data...")
        scaler.fit(train_nodes_list, train_raw["global_feats"])
        scaler.save(Config.SCALERS_PATH)

    train_nodes_scaled, train_global_scaled = scaler.transform(
        train_nodes_list, train_raw["global_feats"]
    )
    val_nodes_scaled, val_global_scaled = scaler.transform(
        val_nodes_list, val_raw["global_feats"]
    )
    test_nodes_scaled, test_global_scaled = scaler.transform(
        test_nodes_list, test_raw["global_feats"]
    )

    # 3. Target Transformation (Log1p)
    # y -> log(1 + y)
    # Inverse will be exp(y) - 1
    train_targets_log = np.log1p(train_raw["targets"])
    val_targets_log = np.log1p(val_raw["targets"])
    # Test targets are placeholders, no need to transform

    # 4. Create Datasets
    train_dataset = MaterialDataset(
        train_nodes_scaled, train_global_scaled, train_targets_log, train_raw["ids"]
    )

    val_dataset = MaterialDataset(
        val_nodes_scaled, val_global_scaled, val_targets_log, val_raw["ids"]
    )

    test_dataset = MaterialDataset(
        test_nodes_scaled,
        test_global_scaled,
        None,  # No targets for test
        test_raw["ids"],
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_materials,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_materials,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_materials,
        pin_memory=True,
    )

    print(f"Data preparation complete.")
    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
