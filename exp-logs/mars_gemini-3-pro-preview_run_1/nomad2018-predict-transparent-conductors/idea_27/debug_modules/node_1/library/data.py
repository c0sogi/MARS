import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import load_molecule, get_pbc_distances, calculate_cell_volume


class CrystalDataset(Dataset):
    def __init__(
        self, atomic_features, global_features, batch_indices, targets=None, ids=None
    ):
        """
        Args:
            atomic_features: List of np.arrays, one per crystal.
            global_features: np.array of shape (N_crystals, Global_Dim).
            batch_indices: This might be redundant if we store as list of arrays,
                           but useful for collate. Actually, storing as list of arrays
                           is better for the Dataset class.
            targets: np.array of shape (N_crystals, 2).
            ids: List of crystal IDs.
        """
        self.atomic_features = atomic_features
        self.global_features = global_features.astype(np.float32)
        self.targets = targets.astype(np.float32) if targets is not None else None
        self.ids = ids

    def __len__(self):
        return len(self.global_features)

    def __getitem__(self, idx):
        sample = {
            "atomic_features": torch.tensor(
                self.atomic_features[idx], dtype=torch.float32
            ),
            "global_features": torch.tensor(
                self.global_features[idx], dtype=torch.float32
            ),
            "id": self.ids[idx],
        }
        if self.targets is not None:
            sample["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sample


def collate_fn(batch):
    """
    Custom collate function to handle variable number of atoms.
    Merges atomic features into a single tensor and creates a batch index vector.
    """
    atomic_features_list = []
    global_features_list = []
    targets_list = []
    batch_indices_list = []
    ids_list = []

    for i, sample in enumerate(batch):
        # Atomic features
        atoms_feats = sample["atomic_features"]
        atomic_features_list.append(atoms_feats)

        # Batch index for each atom in this crystal
        # i is the index within this batch
        num_atoms = atoms_feats.shape[0]
        batch_indices_list.append(torch.full((num_atoms,), i, dtype=torch.long))

        # Global features
        global_features_list.append(sample["global_features"])

        # IDs
        ids_list.append(sample["id"])

        # Targets
        if "targets" in sample:
            targets_list.append(sample["targets"])

    # Concatenate everything
    atomic_features_batch = torch.cat(atomic_features_list, dim=0)
    batch_indices_batch = torch.cat(batch_indices_list, dim=0)
    global_features_batch = torch.stack(global_features_list, dim=0)

    result = {
        "atomic_features": atomic_features_batch,
        "batch_indices": batch_indices_batch,
        "global_features": global_features_batch,
        "ids": ids_list,
    }

    if targets_list:
        result["targets"] = torch.stack(targets_list, dim=0)

    return result


def extract_features(df, scaler_atomic=None, scaler_global=None, fit_scalers=False):
    """
    Extracts atomic and global features for all crystals in the dataframe.
    """
    all_atomic_features = []
    all_global_features = []
    all_targets = []
    all_ids = []

    # Pre-compute atom type mapping for speed
    # Config.ATOMIC_MAP = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
    atom_type_map = Config.ATOMIC_MAP
    num_atom_types = Config.NUM_ATOM_TYPES

    print(f"Extracting features for {len(df)} samples...")

    # Lists to hold raw data for scaling later
    raw_atomic_data = []  # Will be a list of arrays, then concatenated for fit
    raw_global_data = []

    # Store the number of atoms per crystal to reconstruct the list structure after scaling
    atoms_per_crystal = []

    for idx, row in df.iterrows():
        crystal_id = row["id"]
        file_path = row["file_path"]

        # 1. Load Geometry
        try:
            atoms = load_molecule(file_path)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue

        # 2. Atomic Features
        # ------------------

        # a. One-hot encoding
        symbols = atoms.get_chemical_symbols()
        n_atoms = len(atoms)
        one_hot = np.zeros((n_atoms, num_atom_types))
        for i, s in enumerate(symbols):
            if s in atom_type_map:
                one_hot[i, atom_type_map[s]] = 1.0

        # b. Centered Coordinates
        positions = atoms.get_positions()
        centroid = np.mean(positions, axis=0)
        centered_pos = positions - centroid

        # c. Distances & Reciprocal Proximity
        # Get PBC distances matrix
        dist_matrix = get_pbc_distances(atoms)
        # Fill diagonal with infinity to ignore self-distance in min()
        np.fill_diagonal(dist_matrix, np.inf)

        recip_prox = np.zeros((n_atoms, num_atom_types))

        # Identify indices for each atom type
        type_indices = {t: [] for t in atom_type_map}
        for i, s in enumerate(symbols):
            if s in atom_type_map:
                type_indices[s].append(i)

        for t_name, t_idx in atom_type_map.items():
            indices = type_indices[t_name]
            if len(indices) > 0:
                # For each atom (row), find min distance to any atom of type t_name (cols)
                # dist_matrix shape: (n_atoms, n_atoms)
                # subset columns: dist_matrix[:, indices]
                d_min = np.min(dist_matrix[:, indices], axis=1)

                # Avoid division by zero (shouldn't happen due to diagonal inf, but safety check)
                # d_min can be very small? minimal bond length is usually > 1A.
                with np.errstate(divide="ignore"):
                    recip = 1.0 / d_min
                # If d_min is inf (no neighbor found?), recip is 0.
                recip[d_min == np.inf] = 0.0
                recip_prox[:, t_idx] = recip
            else:
                # No atoms of this type in crystal
                recip_prox[:, t_idx] = 0.0

        # d. Local Packing Density
        # Mean distance to K nearest neighbors
        k = min(Config.K_NEIGHBORS, n_atoms - 1)
        if k > 0:
            # Sort distances for each atom
            sorted_dists = np.sort(dist_matrix, axis=1)
            # Take first k (columns 0 to k-1)
            nearest_k = sorted_dists[:, :k]
            packing_density = np.mean(nearest_k, axis=1).reshape(-1, 1)
        else:
            packing_density = np.zeros((n_atoms, 1))

        # Combine Atomic Features
        # 4 (one-hot) + 3 (coords) + 4 (recip) + 1 (packing) = 12
        atom_feats = np.hstack([one_hot, centered_pos, recip_prox, packing_density])

        raw_atomic_data.append(atom_feats)
        atoms_per_crystal.append(n_atoms)

        # 3. Global Features
        # ------------------
        # Lattice lengths and angles
        lat_len = np.array(
            [
                row["lattice_vector_1_ang"],
                row["lattice_vector_2_ang"],
                row["lattice_vector_3_ang"],
            ]
        )
        lat_ang = np.array(
            [
                row["lattice_angle_alpha_degree"],
                row["lattice_angle_beta_degree"],
                row["lattice_angle_gamma_degree"],
            ]
        )

        # Volume
        vol = calculate_cell_volume(lat_len, lat_ang)

        # Atomic Density
        density = n_atoms / vol if vol > 1e-6 else 0.0

        # Stoichiometry
        stoich = np.array(
            [row["percent_atom_al"], row["percent_atom_ga"], row["percent_atom_in"]]
        )

        # Total Atoms
        total_atoms = np.array([row["number_of_total_atoms"]])

        # Combine Global Features
        # 3 + 3 + 1 + 1 + 3 + 1 = 12
        glob_feats = np.concatenate(
            [lat_len, lat_ang, [vol], [density], stoich, total_atoms]
        )
        raw_global_data.append(glob_feats)

        # 4. Targets & ID
        if "formation_energy_ev_natom" in row and "bandgap_energy_ev" in row:
            # Log transform targets: log(1 + y)
            # Ensure non-negative before log? Energies can be negative?
            # Formation energy can be negative (stable). Bandgap is usually positive.
            # Task description metric is RMSLE. Usually implies targets are positive.
            # Let's check data analysis... min formation energy is 0.0.
            # So log1p is safe.
            t1 = np.log1p(row["formation_energy_ev_natom"])
            t2 = np.log1p(row["bandgap_energy_ev"])
            all_targets.append([t1, t2])
        else:
            # Placeholder for test set
            all_targets.append([0.0, 0.0])

        all_ids.append(crystal_id)

    # Convert to numpy arrays
    raw_global_array = np.array(raw_global_data)

    # Scaling
    if fit_scalers:
        # Concatenate all atomic features to fit scaler
        all_atoms_concat = np.vstack(raw_atomic_data)
        scaler_atomic = StandardScaler()
        scaler_atomic.fit(all_atoms_concat)

        scaler_global = StandardScaler()
        scaler_global.fit(raw_global_array)

    # Apply scaling
    # Atomic
    scaled_atomic_features = []
    for af in raw_atomic_data:
        if scaler_atomic:
            scaled_atomic_features.append(scaler_atomic.transform(af))
        else:
            scaled_atomic_features.append(af)

    # Global
    if scaler_global:
        scaled_global_features = scaler_global.transform(raw_global_array)
    else:
        scaled_global_features = raw_global_array

    return (
        scaled_atomic_features,
        scaled_global_features,
        np.array(all_targets),
        all_ids,
        scaler_atomic,
        scaler_global,
    )


def process_and_cache_data(
    df, cache_path, scaler_atomic=None, scaler_global=None, fit_scalers=False
):
    """
    Wrapper to handle extraction and caching.
    """
    atomic_feats, global_feats, targets, ids, s_atomic, s_global = extract_features(
        df, scaler_atomic, scaler_global, fit_scalers
    )

    # We can't easily save list of variable length arrays in npz directly without object=True
    # Instead, we'll flatten atomic feats and store split indices, or just pickle (but pickle prohibited).
    # We will use object array for atomic features list in npz, which numpy allows.
    # Alternatively, we can save as flat array + indices.
    # Let's use object array for simplicity if allowed, otherwise flat + counts.
    # "Do NOT use pickle" usually implies avoiding pickle files, but np.savez with allow_pickle=True is standard.
    # If strict no-pickle, we must flatten.

    # Flattening atomic features for storage
    flat_atomic = np.vstack(atomic_feats)
    # Store lengths to reconstruct
    lengths = np.array([len(x) for x in atomic_feats])

    np.savez(
        cache_path,
        flat_atomic=flat_atomic,
        lengths=lengths,
        global_feats=global_feats,
        targets=targets,
        ids=np.array(ids),
    )

    return atomic_feats, global_feats, targets, ids, s_atomic, s_global


def load_cached_dataset(cache_path):
    data = np.load(cache_path)
    flat_atomic = data["flat_atomic"]
    lengths = data["lengths"]
    global_feats = data["global_feats"]
    targets = data["targets"]
    ids = data["ids"]

    # Reconstruct atomic features list
    atomic_feats = []
    idx = 0
    for l in lengths:
        atomic_feats.append(flat_atomic[idx : idx + l])
        idx += l

    return atomic_feats, global_feats, targets, ids


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Main entry point to get dataloaders.
    """
    # Define cache paths
    train_cache = os.path.join(Config.CACHE_DIR, "train_data.npz")
    val_cache = os.path.join(Config.CACHE_DIR, "val_data.npz")
    test_cache = os.path.join(Config.CACHE_DIR, "test_data.npz")
    scaler_cache = os.path.join(Config.CACHE_DIR, "scalers.npz")

    # Load Metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # --- Training Data ---
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(scaler_cache)
    ):
        print("Loading cached training data...")
        train_atomic, train_global, train_targets, train_ids = load_cached_dataset(
            train_cache
        )

        # Load scalers (we need to reconstruct them to transform val/test)
        # Since sklearn scalers are python objects, we can't easily save/load without pickle.
        # We will save mean/scale params manually.
        scaler_data = np.load(scaler_cache)

        scaler_atomic = StandardScaler()
        scaler_atomic.mean_ = scaler_data["atomic_mean"]
        scaler_atomic.scale_ = scaler_data["atomic_scale"]
        scaler_atomic.var_ = scaler_data["atomic_var"]
        scaler_atomic.n_samples_seen_ = scaler_data["atomic_n"]

        scaler_global = StandardScaler()
        scaler_global.mean_ = scaler_data["global_mean"]
        scaler_global.scale_ = scaler_data["global_scale"]
        scaler_global.var_ = scaler_data["global_var"]
        scaler_global.n_samples_seen_ = scaler_data["global_n"]

    else:
        print("Processing training data...")
        (
            train_atomic,
            train_global,
            train_targets,
            train_ids,
            scaler_atomic,
            scaler_global,
        ) = process_and_cache_data(train_df, train_cache, fit_scalers=True)

        # Save scaler params
        np.savez(
            scaler_cache,
            atomic_mean=scaler_atomic.mean_,
            atomic_scale=scaler_atomic.scale_,
            atomic_var=scaler_atomic.var_,
            atomic_n=scaler_atomic.n_samples_seen_,
            global_mean=scaler_global.mean_,
            global_scale=scaler_global.scale_,
            global_var=scaler_global.var_,
            global_n=scaler_global.n_samples_seen_,
        )

    # --- Validation Data ---
    if load_cached_data and os.path.exists(val_cache):
        print("Loading cached validation data...")
        val_atomic, val_global, val_targets, val_ids = load_cached_dataset(val_cache)
    else:
        print("Processing validation data...")
        val_atomic, val_global, val_targets, val_ids, _, _ = process_and_cache_data(
            val_df, val_cache, scaler_atomic, scaler_global, fit_scalers=False
        )

    # --- Test Data ---
    if load_cached_data and os.path.exists(test_cache):
        print("Loading cached test data...")
        test_atomic, test_global, test_targets, test_ids = load_cached_dataset(
            test_cache
        )
    else:
        print("Processing test data...")
        test_atomic, test_global, test_targets, test_ids, _, _ = process_and_cache_data(
            test_df, test_cache, scaler_atomic, scaler_global, fit_scalers=False
        )

    # Create Datasets
    # batch_indices are handled in collate_fn, so we pass None here
    train_dataset = CrystalDataset(
        train_atomic, train_global, None, train_targets, train_ids
    )
    val_dataset = CrystalDataset(val_atomic, val_global, None, val_targets, val_ids)
    test_dataset = CrystalDataset(
        test_atomic, test_global, None, None, test_ids
    )  # No targets for test

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
