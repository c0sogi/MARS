import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import cache_data
from library.geometry import get_geometry_processor


class CrystalDataset(Dataset):
    """
    PyTorch Dataset for crystal structures.
    """

    def __init__(self, data_dict):
        self.ids = data_dict["ids"]
        self.species_indices = data_dict["species_indices"]
        self.centered_coords = data_dict["centered_coords"]
        self.d_min = data_dict["d_min"]
        self.d_mean = data_dict["d_mean"]
        self.global_features = data_dict["global_features"]

        # Targets might not exist for test set
        if "targets" in data_dict:
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Atomic features
        # One-hot encoding for species
        n_atoms = len(self.species_indices[idx])
        species_onehot = np.zeros((n_atoms, len(Config.ATOM_TYPES)), dtype=np.float32)
        species_onehot[np.arange(n_atoms), self.species_indices[idx]] = 1.0

        # Concatenate atomic features: [OneHot(4), Coords(3), d_min(1), d_mean(1)]
        # Ensure shapes are (N, 1) for scalars
        coords = self.centered_coords[idx].astype(np.float32)
        d_min = self.d_min[idx].reshape(-1, 1).astype(np.float32)
        d_mean = self.d_mean[idx].reshape(-1, 1).astype(np.float32)

        atomic_feats = np.concatenate([species_onehot, coords, d_min, d_mean], axis=1)

        # Global features
        global_feats = self.global_features[idx].astype(np.float32)

        item = {
            "atomic_features": torch.from_numpy(atomic_feats),
            "global_features": torch.from_numpy(global_feats),
            "id": self.ids[idx],
            "n_atoms": n_atoms,
        }

        if self.targets is not None:
            item["targets"] = torch.from_numpy(self.targets[idx].astype(np.float32))

        return item


def collate_crystals(batch):
    """
    Custom collate function to handle variable number of atoms.
    Creates a packed representation for atomic features.
    """
    ids = []
    batch_atomic_feats = []
    batch_global_feats = []
    batch_targets = []
    batch_indices = []

    for i, item in enumerate(batch):
        ids.append(item["id"])

        # Atomic features
        n_atoms = item["n_atoms"]
        batch_atomic_feats.append(item["atomic_features"])
        # Create batch index vector (e.g., [0, 0, 0, 1, 1, ...])
        batch_indices.append(torch.full((n_atoms,), i, dtype=torch.long))

        # Global features
        batch_global_feats.append(item["global_features"])

        # Targets
        if "targets" in item:
            batch_targets.append(item["targets"])

    # Concatenate all
    atomic_features = torch.cat(batch_atomic_feats, dim=0)
    batch_indices = torch.cat(batch_indices, dim=0)
    global_features = torch.stack(batch_global_feats, dim=0)

    result = {
        "atomic_features": atomic_features,
        "batch_indices": batch_indices,
        "global_features": global_features,
        "ids": ids,
    }

    if batch_targets:
        result["targets"] = torch.stack(batch_targets, dim=0)

    return result


def _compute_scalers(train_data):
    """
    Computes mean and std for continuous features from training data.
    """
    scalers = {}

    # 1. Atomic Coordinates (indices 4, 5, 6 in atomic_features)
    # We flatten all atoms from all crystals
    all_coords = np.concatenate(train_data["centered_coords"], axis=0)
    scalers["coords_mean"] = np.mean(all_coords, axis=0)
    scalers["coords_std"] = np.std(all_coords, axis=0) + 1e-8

    # 2. Geometric Scalars (d_min, d_mean)
    all_d_min = np.concatenate(train_data["d_min"], axis=0)
    all_d_mean = np.concatenate(train_data["d_mean"], axis=0)

    scalers["d_min_mean"] = np.mean(all_d_min)
    scalers["d_min_std"] = np.std(all_d_min) + 1e-8

    scalers["d_mean_mean"] = np.mean(all_d_mean)
    scalers["d_mean_std"] = np.std(all_d_mean) + 1e-8

    # 3. Global Features
    # Construct global feature matrix: [lengths(3), angles(3), vol(1), dens(1), stoich(3), num(1)]
    # We need to construct it exactly as it is done in GeometryProcessor to compute stats
    # Actually, GeometryProcessor returns them as separate arrays in the dict,
    # but we want to scale the concatenated vector.
    # Let's stack them first to match the input dim of 12.

    # Extract arrays
    lat_len = train_data["lattice_lengths"]
    lat_ang = train_data["lattice_angles"]
    vol = train_data["volume"].reshape(-1, 1)
    dens = train_data["density"].reshape(-1, 1)
    stoich = train_data["stoichiometry"]
    num = train_data["num_atoms"].reshape(-1, 1)

    global_mat = np.concatenate([lat_len, lat_ang, vol, dens, stoich, num], axis=1)

    scalers["global_mean"] = np.mean(global_mat, axis=0)
    scalers["global_std"] = np.std(global_mat, axis=0) + 1e-8

    return scalers


def preprocess_features(data, scalers=None, fit_scalers=False):
    """
    Applies scaling to features.
    If fit_scalers is True, computes scalers from data (assumed to be training set).
    Otherwise, uses provided scalers.

    Also assembles the 'global_features' array in the data dictionary.
    """
    # Assemble global features matrix first
    lat_len = data["lattice_lengths"]
    lat_ang = data["lattice_angles"]
    vol = data["volume"].reshape(-1, 1)
    dens = data["density"].reshape(-1, 1)
    stoich = data["stoichiometry"]
    num = data["num_atoms"].reshape(-1, 1)

    global_mat = np.concatenate([lat_len, lat_ang, vol, dens, stoich, num], axis=1)

    if fit_scalers:
        scalers = _compute_scalers(data)

    # Apply scaling
    # 1. Atomic Coords
    # Process item by item to keep object array structure
    norm_coords = []
    for arr in data["centered_coords"]:
        norm = (arr - scalers["coords_mean"]) / scalers["coords_std"]
        norm_coords.append(norm)
    data["centered_coords"] = np.array(norm_coords, dtype=object)

    # 2. Geometric Scalars
    norm_d_min = []
    for arr in data["d_min"]:
        norm = (arr - scalers["d_min_mean"]) / scalers["d_min_std"]
        norm_d_min.append(norm)
    data["d_min"] = np.array(norm_d_min, dtype=object)

    norm_d_mean = []
    for arr in data["d_mean"]:
        norm = (arr - scalers["d_mean_mean"]) / scalers["d_mean_std"]
        norm_d_mean.append(norm)
    data["d_mean"] = np.array(norm_d_mean, dtype=object)

    # 3. Global Features
    data["global_features"] = (global_mat - scalers["global_mean"]) / scalers[
        "global_std"
    ]

    # 4. Log transform targets if they exist
    if "targets" in data:
        # log(1 + y)
        data["targets"] = np.log1p(data["targets"])

    return data, scalers


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Main entry point to get data loaders.
    """
    processor = get_geometry_processor()

    # 1. Load Raw Data (Cached via GeometryProcessor)
    print("Loading raw data...")
    train_raw = processor.process_data("train", load_cached_data=load_cached_data)
    val_raw = processor.process_data("val", load_cached_data=load_cached_data)
    test_raw = processor.process_data("test", load_cached_data=load_cached_data)

    # 2. Preprocess (Scale & Transform)
    # We need to cache scalers to ensure consistency across runs and inference
    scaler_path = os.path.join(Config.WORKING_DIR, "scalers.npz")

    if load_cached_data and os.path.exists(scaler_path):
        print("Loading cached scalers...")
        scalers = dict(np.load(scaler_path))
        # Process train with loaded scalers (no fit)
        train_data, _ = preprocess_features(
            train_raw, scalers=scalers, fit_scalers=False
        )
    else:
        print("Computing scalers from training data...")
        train_data, scalers = preprocess_features(train_raw, fit_scalers=True)
        np.savez(scaler_path, **scalers)

    # Process val and test using training scalers
    val_data, _ = preprocess_features(val_raw, scalers=scalers, fit_scalers=False)
    test_data, _ = preprocess_features(test_raw, scalers=scalers, fit_scalers=False)

    # 3. Create Datasets
    train_dataset = CrystalDataset(train_data)
    val_dataset = CrystalDataset(val_data)
    test_dataset = CrystalDataset(test_data)

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_crystals,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_crystals,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
