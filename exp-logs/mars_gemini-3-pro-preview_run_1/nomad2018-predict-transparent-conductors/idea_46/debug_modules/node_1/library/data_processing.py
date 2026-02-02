import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    ATOM_TO_IDX,
    ATOM_TYPES,
    COVALENT_RADII,
    ATOMIC_MASSES,
    ELECTRONEGATIVITY,
    K_NEIGHBORS_LIST,
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    SCALERS_CACHE_PATH,
    BATCH_SIZE,
    DEBUG_MODE,
    DEBUG_SUBSET_SIZE,
)
from library.physics_utils import (
    calculate_angular_distortion,
    get_pbc_neighbors,
    compute_weighted_context,
    calculate_bond_hardness,
)


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file to extract lattice vectors and atomic information.
    """
    full_path = os.path.join(INPUT_DIR, file_path)
    lattice = []
    atom_coords = []
    atom_types = []

    with open(full_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "lattice_vector":
                lattice.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                atom_coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    return np.array(lattice), np.array(atom_coords), atom_types


class MaterialDataset(Dataset):
    def __init__(self, metadata_path, cache_path, load_cached_data=True, is_test=False):
        self.metadata_path = metadata_path
        self.cache_path = cache_path
        self.is_test = is_test

        # Load metadata
        self.df = pd.read_csv(metadata_path)
        if DEBUG_MODE:
            self.df = self.df.iloc[:DEBUG_SUBSET_SIZE]

        # Try to load cached data
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            self.ids = data["ids"]
            self.atom_features_list = data["atom_features_list"]
            self.global_features = data["global_features"]
            self.targets = (
                data["targets"] if not is_test else np.zeros((len(self.ids), 2))
            )
            self.num_atoms_list = data["num_atoms_list"]
        else:
            print(f"Processing data from {metadata_path}...")
            self.process_data()

    def process_data(self):
        ids = []
        atom_features_list = []
        global_features_list = []
        targets_list = []
        num_atoms_list = []

        for idx, row in self.df.iterrows():
            # 1. Geometry Parsing
            lattice, coords, atom_types = parse_xyz(row["file_path"])
            num_atoms = len(atom_types)

            # Centering
            centroid = np.mean(coords, axis=0)
            centered_coords = coords - centroid

            # 2. Atomic Feature Engineering
            # One-hot encoding
            one_hot = np.zeros((num_atoms, 4))
            type_indices = [ATOM_TO_IDX[t] for t in atom_types]
            one_hot[np.arange(num_atoms), type_indices] = 1.0

            # Neighbor Search (PBC)
            # Find enough neighbors for the largest K context
            max_k = max(K_NEIGHBORS_LIST) + 1  # +1 for self/nearest calculation safety
            neighbor_dists, neighbor_type_indices = get_pbc_neighbors(
                coords, lattice, atom_types, max_k
            )

            # d_min (distance to nearest neighbor, exclude self which is usually index 0 in sorted list if dist~0)
            # get_pbc_neighbors returns sorted distances.
            # If the first distance is effectively 0, it's the atom itself.
            d_min = neighbor_dists[:, 1]  # Closest non-self neighbor

            # d_mean_12 (approximate coordination shell)
            # Use up to 12 neighbors or available
            k_packing = min(12, neighbor_dists.shape[1])
            d_mean_12 = np.mean(neighbor_dists[:, 1 : k_packing + 1], axis=1)

            # Packing Ratio
            packing_ratio = d_min / (d_mean_12 + 1e-8)

            # Bond Hardness Proxy
            # Context K=6 for radius averaging
            bond_hardness = calculate_bond_hardness(
                d_min,
                atom_types,
                neighbor_type_indices[:, 1:],
                neighbor_dists[:, 1:],
                k_context=6,
            )

            # Chemical Contexts
            contexts = []
            for k in K_NEIGHBORS_LIST:
                # Exclude self (index 0)
                ctx = compute_weighted_context(
                    neighbor_dists[:, 1 : k + 1],
                    neighbor_type_indices[:, 1 : k + 1],
                    num_types=4,
                )
                contexts.append(ctx)

            # Assemble Atomic Features
            # 4 (One-hot) + 3 (Coords) + 1 (d_min) + 1 (Packing) + 1 (Hardness) + 4 (Ctx1) + 4 (Ctx2) = 18
            atom_feats = np.concatenate(
                [
                    one_hot,
                    centered_coords,
                    d_min[:, np.newaxis],
                    packing_ratio[:, np.newaxis],
                    bond_hardness[:, np.newaxis],
                    contexts[0],
                    contexts[1],
                ],
                axis=1,
            )

            # 3. Global Feature Engineering
            # Lattice parameters
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

            # Volume & Density (Calculate volume from lattice vectors for consistency)
            # V = a . (b x c)
            vol = np.abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
            density = num_atoms / vol

            # Stoichiometry (Counts)
            stoich = np.array([atom_types.count(t) for t in ATOM_TYPES])

            # Aspect Ratios
            aspect_ratios = np.array(
                [
                    lat_lens[0] / (lat_lens[1] + 1e-8),
                    lat_lens[1] / (lat_lens[2] + 1e-8),
                    lat_lens[2] / (lat_lens[0] + 1e-8),
                ]
            )

            # Weighted Physics
            # Composition fractions
            fractions = stoich / num_atoms

            mean_mass = sum(
                fractions[i] * ATOMIC_MASSES[t] for i, t in enumerate(ATOM_TYPES)
            )
            mean_radius = sum(
                fractions[i] * COVALENT_RADII[t] for i, t in enumerate(ATOM_TYPES)
            )
            mean_en = sum(
                fractions[i] * ELECTRONEGATIVITY[t] for i, t in enumerate(ATOM_TYPES)
            )
            weighted_physics = np.array([mean_mass, mean_radius, mean_en])

            # Angular Distortion
            ang_distortion = calculate_angular_distortion(
                lat_angs[0], lat_angs[1], lat_angs[2]
            )

            # Assemble Global Features
            # 3+3+1+1+4+1+3+3+1 = 20
            global_feats = np.concatenate(
                [
                    lat_lens,
                    lat_angs,
                    np.array([vol, density]),
                    stoich,
                    np.array([num_atoms]),
                    aspect_ratios,
                    weighted_physics,
                    np.array([ang_distortion]),
                ]
            )

            # Targets
            if not self.is_test:
                target = np.array(
                    [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
                )
            else:
                target = np.zeros(2)

            ids.append(row["id"])
            atom_features_list.append(atom_feats.astype(np.float32))
            global_features_list.append(global_feats.astype(np.float32))
            targets_list.append(target.astype(np.float32))
            num_atoms_list.append(num_atoms)

        # Convert to numpy object array for variable length atom features
        self.ids = np.array(ids)
        self.atom_features_list = np.array(atom_features_list, dtype=object)
        self.global_features = np.array(global_features_list, dtype=np.float32)
        self.targets = np.array(targets_list, dtype=np.float32)
        self.num_atoms_list = np.array(num_atoms_list, dtype=np.int32)

        # Save to cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez(
            self.cache_path,
            ids=self.ids,
            atom_features_list=self.atom_features_list,
            global_features=self.global_features,
            targets=self.targets,
            num_atoms_list=self.num_atoms_list,
        )
        print(f"Data saved to {self.cache_path}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "atom_features": self.atom_features_list[idx],
            "global_features": self.global_features[idx],
            "target": self.targets[idx],
            "num_atoms": self.num_atoms_list[idx],
        }


def collate_sparse_batch(batch):
    """
    Collates a list of samples into a sparse batch.
    Flattens atom features and creates a batch index vector.
    """
    ids = []
    atom_features_list = []
    global_features_list = []
    targets_list = []
    batch_indices = []

    for i, sample in enumerate(batch):
        ids.append(sample["id"])
        atom_features_list.append(
            torch.tensor(sample["atom_features"], dtype=torch.float32)
        )
        global_features_list.append(
            torch.tensor(sample["global_features"], dtype=torch.float32)
        )
        targets_list.append(torch.tensor(sample["target"], dtype=torch.float32))

        # Create batch index for this sample's atoms
        num_atoms = sample["num_atoms"]
        batch_indices.append(torch.full((num_atoms,), i, dtype=torch.long))

    # Concatenate all
    atom_features = torch.cat(atom_features_list, dim=0)
    batch_index = torch.cat(batch_indices, dim=0)
    global_features = torch.stack(global_features_list, dim=0)
    targets = torch.stack(targets_list, dim=0)
    ids = torch.tensor(ids, dtype=torch.long)

    return {
        "atom_features": atom_features,
        "batch_index": batch_index,
        "global_features": global_features,
        "targets": targets,
        "ids": ids,
    }


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.scale = np.std(data, axis=0)
        # Avoid division by zero
        self.scale[self.scale < 1e-8] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.scale

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale)

    def load(self, path):
        data = np.load(path)
        self.mean = data["mean"]
        self.scale = data["scale"]


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    Handles scaling of continuous features.
    """
    # 1. Initialize Datasets
    train_dataset = MaterialDataset(
        TRAIN_METADATA_PATH, TRAIN_CACHE_PATH, load_cached_data, is_test=False
    )
    val_dataset = MaterialDataset(
        VAL_METADATA_PATH, VAL_CACHE_PATH, load_cached_data, is_test=False
    )
    test_dataset = MaterialDataset(
        TEST_METADATA_PATH, TEST_CACHE_PATH, load_cached_data, is_test=True
    )

    # 2. Fit/Load Scalers
    # Atomic continuous features: indices 4 to end (0-3 are one-hot)
    # Global continuous features: all indices 0 to end

    atom_scaler = StandardScaler()
    global_scaler = StandardScaler()

    if load_cached_data and os.path.exists(SCALERS_CACHE_PATH):
        print("Loading scalers...")
        data = np.load(SCALERS_CACHE_PATH)
        atom_scaler.mean = data["atom_mean"]
        atom_scaler.scale = data["atom_scale"]
        global_scaler.mean = data["global_mean"]
        global_scaler.scale = data["global_scale"]
    else:
        print("Fitting scalers on training data...")
        # Collect all atomic features for fitting
        all_train_atoms = np.concatenate(train_dataset.atom_features_list, axis=0)
        atom_scaler.fit(all_train_atoms[:, 4:])  # Fit only continuous part

        global_scaler.fit(train_dataset.global_features)

        # Save scalers
        np.savez(
            SCALERS_CACHE_PATH,
            atom_mean=atom_scaler.mean,
            atom_scale=atom_scaler.scale,
            global_mean=global_scaler.mean,
            global_scale=global_scaler.scale,
        )

    # 3. Apply Scaling (Transform data in-place in memory)
    def apply_scaling(dataset):
        # Scale Atomic Features
        for i in range(len(dataset.atom_features_list)):
            feats = dataset.atom_features_list[i]
            # Scale continuous part (cols 4+)
            feats[:, 4:] = atom_scaler.transform(feats[:, 4:])
            dataset.atom_features_list[i] = feats

        # Scale Global Features
        dataset.global_features = global_scaler.transform(dataset.global_features)

    print("Applying scaling...")
    apply_scaling(train_dataset)
    apply_scaling(val_dataset)
    apply_scaling(test_dataset)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_sparse_batch,
        num_workers=0,  # Avoid multiprocessing issues in some envs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sparse_batch,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sparse_batch,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
