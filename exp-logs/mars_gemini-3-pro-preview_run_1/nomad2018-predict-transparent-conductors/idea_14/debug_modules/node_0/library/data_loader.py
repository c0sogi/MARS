import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config, set_seed
from library.geometry_processor import extract_atomic_features

# Set seed for reproducibility
set_seed(Config.SEED)


class SimpleScaler:
    """
    A simple StandardScaler implementation that can be saved/loaded
    without pickle by storing mean and scale as numpy arrays.
    """

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.scale = np.std(data, axis=0)
        # Handle constant features to avoid division by zero
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, data):
        if self.mean is None or self.scale is None:
            # If not fitted, return data as is (or raise error)
            return data
        return (data - self.mean) / self.scale

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale)

    def load(self, path):
        if os.path.exists(path):
            data = np.load(path)
            self.mean = data["mean"]
            self.scale = data["scale"]
        else:
            print(f"Warning: Scaler file {path} not found.")


class MaterialDataset(Dataset):
    def __init__(self, atomic_features, atom_counts, global_features, targets, ids):
        self.atomic_features = atomic_features
        self.atom_counts = atom_counts
        self.global_features = global_features
        self.targets = targets
        self.ids = ids

        # Pre-calculate start indices for fast slicing
        self.cumulative_counts = np.concatenate(([0], np.cumsum(atom_counts)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        start = self.cumulative_counts[idx]
        end = self.cumulative_counts[idx + 1]

        # Get atomic features for this crystal
        atoms = torch.from_numpy(self.atomic_features[start:end]).float()

        # Global features
        glob = torch.from_numpy(self.global_features[idx]).float()

        # Targets (if available)
        if self.targets is not None:
            target = torch.from_numpy(self.targets[idx]).float()
        else:
            target = torch.tensor([])

        sample_id = self.ids[idx]

        return atoms, glob, target, sample_id


def calculate_cell_volume(lattice_lengths, lattice_angles):
    a, b, c = lattice_lengths
    alpha, beta, gamma = np.radians(lattice_angles)

    term = (
        1
        - np.cos(alpha) ** 2
        - np.cos(beta) ** 2
        - np.cos(gamma) ** 2
        + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )
    return a * b * c * np.sqrt(max(0, term))


def process_data_internal(csv_path, scaler_path, mode="train", debug_size=None):
    """
    Internal function to process raw data, extract features, and apply scaling.
    """
    df = pd.read_csv(csv_path)
    if debug_size:
        df = df.head(debug_size)

    all_atomic_feats = []
    atom_counts = []
    all_global_feats = []
    all_targets = []
    all_ids = []

    print(f"Processing {len(df)} samples from {csv_path} (Mode: {mode})...")

    for _, row in df.iterrows():
        sample_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # 1. Extract Geometry Features
        geo_data = extract_atomic_features(full_path)

        # Atomic features construction
        # One-hot encoding
        atom_types = geo_data["atom_types"]
        one_hot = np.zeros((len(atom_types), Config.NUM_ATOM_TYPES))
        for i, atom in enumerate(atom_types):
            if atom in Config.ATOM_TO_IDX:
                one_hot[i, Config.ATOM_TO_IDX[atom]] = 1.0

        # Combine atomic features:
        # One-Hot (4) + Centered Cart (3) + Frac (3) + NN Dist (1) + Potential (1) = 12 dims
        atomic_feat = np.hstack(
            [
                one_hot,
                geo_data["centered_cart_coords"],
                geo_data["frac_coords"],
                geo_data["nn_dist"],
                geo_data["potential_proxy"],
            ]
        )

        all_atomic_feats.append(atomic_feat)
        atom_counts.append(len(atom_types))

        # 2. Global Features
        lat_lens = np.array(
            [
                row["lattice_vector_1_ang"],
                row["lattice_vector_2_ang"],
                row["lattice_vector_3_ang"],
            ]
        )
        lat_angs = np.array(
            [
                row["lattice_angle_alpha_degree"],
                row["lattice_angle_beta_degree"],
                row["lattice_angle_gamma_degree"],
            ]
        )
        stoich = np.array(
            [row["percent_atom_al"], row["percent_atom_ga"], row["percent_atom_in"]]
        )
        n_atoms = row["number_of_total_atoms"]

        # Derived global features
        vol = calculate_cell_volume(lat_lens, lat_angs)
        density = n_atoms / vol if vol > 1e-6 else 0

        # Combine global: Lat Len (3) + Lat Ang (3) + Vol (1) + Density (1) + Stoich (3) + N_atoms (1) = 12 dims
        global_feat = np.concatenate(
            [lat_lens, lat_angs, [vol, density], stoich, [n_atoms]]
        )
        all_global_feats.append(global_feat)

        # 3. Targets (log1p transformed)
        if mode in ["train", "val"]:
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            all_targets.append([t1, t2])

        all_ids.append(sample_id)

    # Concatenate all atomic features for scaling
    flat_atomic = np.vstack(all_atomic_feats)
    stacked_global = np.vstack(all_global_feats)

    if mode in ["train", "val"]:
        stacked_targets = np.vstack(all_targets)
    else:
        stacked_targets = None

    # Scaling
    atomic_scaler = SimpleScaler()
    global_scaler = SimpleScaler()

    if mode == "train":
        # Fit scalers on training data
        # Note: indices 0-3 are one-hot, don't scale them. Scale indices 4:12.
        atomic_continuous = flat_atomic[:, 4:]
        atomic_scaler.fit(atomic_continuous)
        global_scaler.fit(stacked_global)

        # Save scalers
        atomic_scaler.save(
            scaler_path
        )  # We will save combined in get_data_loaders or just use this
        # Actually, let's save them properly in one file in get_data_loaders, or here.
        # To keep it simple and robust, we save here.
        np.savez(
            scaler_path,
            atomic_mean=atomic_scaler.mean,
            atomic_scale=atomic_scaler.scale,
            global_mean=global_scaler.mean,
            global_scale=global_scaler.scale,
        )
    else:
        # Load scalers for val/test
        if os.path.exists(scaler_path):
            data = np.load(scaler_path)
            atomic_scaler.mean = data["atomic_mean"]
            atomic_scaler.scale = data["atomic_scale"]
            global_scaler.mean = data["global_mean"]
            global_scaler.scale = data["global_scale"]
        else:
            print("Warning: Scaler not found for inference. Data will be unscaled.")

    # Apply Transform
    if atomic_scaler.mean is not None:
        flat_atomic[:, 4:] = atomic_scaler.transform(flat_atomic[:, 4:])
        stacked_global = global_scaler.transform(stacked_global)

    return (
        flat_atomic,
        np.array(atom_counts),
        stacked_global,
        stacked_targets,
        np.array(all_ids),
    )


def collate_materials(batch):
    """
    Collate function for the DataLoader.
    Batch is a list of tuples: (atoms, glob, target, sample_id)
    """
    atoms_list = []
    batch_indices = []
    global_list = []
    target_list = []
    ids_list = []

    for i, (atoms, glob, target, sample_id) in enumerate(batch):
        atoms_list.append(atoms)
        # Create batch index vector: [i, i, i, ...] for the number of atoms in this sample
        batch_indices.append(torch.full((atoms.shape[0],), i, dtype=torch.long))
        global_list.append(glob)
        if target.numel() > 0:
            target_list.append(target)
        ids_list.append(sample_id)

    # Concatenate
    atomic_batch = torch.cat(atoms_list, dim=0)
    batch_idx = torch.cat(batch_indices, dim=0)
    global_batch = torch.stack(global_list, dim=0)

    if target_list:
        target_batch = torch.stack(target_list, dim=0)
    else:
        target_batch = None

    return atomic_batch, batch_idx, global_batch, target_batch, ids_list


def get_data_loaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, debug_sample_size=None
):
    """
    Main entry point to get DataLoaders. Handles caching logic using .npz files.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Map .pt paths from Config to .npz for numpy storage (strict no-pickle compliance)
    train_cache = Config.TRAIN_CACHE_PATH.replace(".pt", ".npz")
    val_cache = Config.VAL_CACHE_PATH.replace(".pt", ".npz")
    test_cache = Config.TEST_CACHE_PATH.replace(".pt", ".npz")
    scaler_cache = Config.SCALER_CACHE_PATH.replace(".pt", ".npz")

    # ---------------------------------------------------------
    # Train Data
    # ---------------------------------------------------------
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(scaler_cache)
    ):
        print("Loading cached training data...")
        data = np.load(train_cache)
        train_dataset = MaterialDataset(
            data["atomic_features"],
            data["atom_counts"],
            data["global_features"],
            data["targets"],
            data["ids"],
        )
    else:
        # Train: Fit scalers, extract targets
        t_atomic, t_counts, t_global, t_targets, t_ids = process_data_internal(
            Config.TRAIN_CSV, scaler_cache, mode="train", debug_size=debug_sample_size
        )
        np.savez(
            train_cache,
            atomic_features=t_atomic,
            atom_counts=t_counts,
            global_features=t_global,
            targets=t_targets,
            ids=t_ids,
        )
        train_dataset = MaterialDataset(t_atomic, t_counts, t_global, t_targets, t_ids)

    # ---------------------------------------------------------
    # Validation Data
    # ---------------------------------------------------------
    if load_cached_data and os.path.exists(val_cache):
        print("Loading cached validation data...")
        data = np.load(val_cache)
        val_dataset = MaterialDataset(
            data["atomic_features"],
            data["atom_counts"],
            data["global_features"],
            data["targets"],
            data["ids"],
        )
    else:
        # Val: Load scalers, extract targets
        v_atomic, v_counts, v_global, v_targets, v_ids = process_data_internal(
            Config.VAL_CSV, scaler_cache, mode="val", debug_size=debug_sample_size
        )
        np.savez(
            val_cache,
            atomic_features=v_atomic,
            atom_counts=v_counts,
            global_features=v_global,
            targets=v_targets,
            ids=v_ids,
        )
        val_dataset = MaterialDataset(v_atomic, v_counts, v_global, v_targets, v_ids)

    # ---------------------------------------------------------
    # Test Data
    # ---------------------------------------------------------
    if load_cached_data and os.path.exists(test_cache):
        print("Loading cached test data...")
        data = np.load(test_cache)
        test_dataset = MaterialDataset(
            data["atomic_features"],
            data["atom_counts"],
            data["global_features"],
            None,
            data["ids"],
        )
    else:
        # Test: Load scalers, no targets
        te_atomic, te_counts, te_global, _, te_ids = process_data_internal(
            Config.TEST_CSV, scaler_cache, mode="test", debug_size=debug_sample_size
        )
        np.savez(
            test_cache,
            atomic_features=te_atomic,
            atom_counts=te_counts,
            global_features=te_global,
            ids=te_ids,
        )
        test_dataset = MaterialDataset(te_atomic, te_counts, te_global, None, te_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_materials,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_materials,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_materials,
        num_workers=Config.NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader
