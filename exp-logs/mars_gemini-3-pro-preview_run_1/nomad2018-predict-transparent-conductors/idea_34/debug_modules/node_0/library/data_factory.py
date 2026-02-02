import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
import library.geometry_utils as geom_utils


class MaterialsDataset(Dataset):
    """
    PyTorch Dataset for Materials.
    """

    def __init__(self, atomic_features_list, global_features, targets=None, ids=None):
        """
        Args:
            atomic_features_list (list of np.ndarray): List of (N_atoms, D_atom) arrays.
            global_features (np.ndarray): (N_samples, D_global) array.
            targets (np.ndarray, optional): (N_samples, 2) array of targets.
            ids (np.ndarray, optional): (N_samples,) array of IDs.
        """
        self.atomic_features_list = atomic_features_list
        self.global_features = global_features
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.global_features)

    def __getitem__(self, idx):
        sample = {
            "atomic_features": torch.tensor(
                self.atomic_features_list[idx], dtype=torch.float32
            ),
            "global_features": torch.tensor(
                self.global_features[idx], dtype=torch.float32
            ),
        }

        if self.targets is not None:
            sample["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            sample["id"] = torch.tensor(self.ids[idx], dtype=torch.long)

        return sample


def materials_collate_fn(batch):
    """
    Custom collate function to handle variable number of atoms.
    Concatenates atomic features and creates a batch index vector.
    """
    atomic_feats = []
    global_feats = []
    batch_indices = []
    targets = []
    ids = []

    for i, sample in enumerate(batch):
        # Atomic features
        a_feat = sample["atomic_features"]
        atomic_feats.append(a_feat)
        # Create batch index for these atoms (all have value 'i')
        batch_indices.append(torch.full((a_feat.shape[0],), i, dtype=torch.long))

        # Global features
        global_feats.append(sample["global_features"])

        # Targets
        if "targets" in sample:
            targets.append(sample["targets"])

        # IDs
        if "id" in sample:
            ids.append(sample["id"])

    # Concatenate everything
    batch_out = {
        "atomic_features": torch.cat(atomic_feats, dim=0),
        "batch_indices": torch.cat(batch_indices, dim=0),
        "global_features": torch.stack(global_feats, dim=0),
    }

    if targets:
        batch_out["targets"] = torch.stack(targets, dim=0)

    if ids:
        batch_out["ids"] = torch.stack(ids, dim=0)

    return batch_out


class DataProcessor:
    def __init__(self):
        self.atom_scaler = StandardScaler()
        self.global_scaler = StandardScaler()

    def _compute_split(self, meta_path, is_test=False):
        """
        Reads metadata, processes geometry files, and extracts features.
        """
        df = pd.read_csv(meta_path)

        atomic_feats_list = []
        global_feats_list = []
        targets_list = []
        ids_list = []

        # Mapping for reconstructing variable length arrays from flat cache
        # Not needed here, but conceptually we produce list of arrays

        print(f"Processing {len(df)} samples from {meta_path}...")

        for _, row in df.iterrows():
            file_path = row["file_path"]

            # Process geometry
            # atomic_f: (N_atoms, D_atom), global_f: (D_global,)
            try:
                atomic_f, global_f = geom_utils.process_geometry(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # Fallback to zeros or skip? Better to fail explicitly or provide dummy
                # Given requirements, we assume data integrity.
                raise e

            atomic_feats_list.append(atomic_f)
            global_feats_list.append(global_f)
            ids_list.append(row["id"])

            if not is_test:
                # Targets: formation_energy, bandgap_energy
                t = [row["formation_energy_ev_natom"], row["bandgap_energy_ev"]]
                targets_list.append(t)

        # Convert lists to numpy arrays where appropriate
        # atomic_feats_list remains a list of arrays
        global_feats = np.array(global_feats_list, dtype=np.float32)
        ids = np.array(ids_list, dtype=np.int64)

        if not is_test:
            targets = np.array(targets_list, dtype=np.float32)
        else:
            targets = None

        return atomic_feats_list, global_feats, targets, ids

    def _flatten_atomic_features(self, atomic_list):
        """
        Flattens list of atomic feature arrays into a single array + index array
        for saving without pickle.
        """
        if not atomic_list:
            return np.array([]), np.array([])

        flat_feats = np.concatenate(atomic_list, axis=0)
        # Create an index array mapping each atom to its sample index
        sample_indices = []
        for i, arr in enumerate(atomic_list):
            sample_indices.append(np.full((arr.shape[0],), i, dtype=np.int32))
        flat_indices = np.concatenate(sample_indices, axis=0)

        return flat_feats, flat_indices

    def _unflatten_atomic_features(self, flat_feats, flat_indices, num_samples):
        """
        Reconstructs list of atomic feature arrays from flat arrays.
        """
        atomic_list = []
        # We can use np.split or simple masking. Since indices are sorted 0..N-1,
        # we can find split points.

        # Find where indices change
        if len(flat_indices) == 0:
            return [np.zeros((0, flat_feats.shape[1])) for _ in range(num_samples)]

        # This assumes flat_indices is sorted, which it is by construction.
        # We need to handle cases where a sample might have 0 atoms (unlikely but possible).
        # A robust way is to iterate.

        current_idx = 0
        for i in range(num_samples):
            # Find count of atoms for sample i
            # Optimization: since sorted, we can just look ahead
            start = current_idx
            count = 0
            while current_idx < len(flat_indices) and flat_indices[current_idx] == i:
                count += 1
                current_idx += 1

            atomic_list.append(flat_feats[start : start + count])

        return atomic_list

    def _save_cache(self, path, atomic_list, global_feats, targets, ids):
        flat_atomic, flat_indices = self._flatten_atomic_features(atomic_list)

        save_dict = {
            "flat_atomic": flat_atomic,
            "flat_indices": flat_indices,
            "global_feats": global_feats,
            "ids": ids,
        }
        if targets is not None:
            save_dict["targets"] = targets

        np.savez(path, **save_dict)
        print(f"Saved cache to {path}")

    def _load_cache(self, path):
        print(f"Loading cache from {path}...")
        data = np.load(path)
        ids = data["ids"]
        num_samples = len(ids)

        atomic_list = self._unflatten_atomic_features(
            data["flat_atomic"], data["flat_indices"], num_samples
        )
        global_feats = data["global_feats"]

        targets = None
        if "targets" in data:
            targets = data["targets"]

        return atomic_list, global_feats, targets, ids

    def _save_scalers(self):
        # Save scalers using np.savez by saving mean and scale
        np.savez(
            Config.SCALERS_PATH,
            atom_mean=self.atom_scaler.mean_,
            atom_scale=self.atom_scaler.scale_,
            global_mean=self.global_scaler.mean_,
            global_scale=self.global_scaler.scale_,
        )

    def _load_scalers(self):
        if os.path.exists(Config.SCALERS_PATH):
            data = np.load(Config.SCALERS_PATH)
            self.atom_scaler.mean_ = data["atom_mean"]
            self.atom_scaler.scale_ = data["atom_scale"]
            self.atom_scaler.var_ = data["atom_scale"] ** 2  # var = scale^2
            self.global_scaler.mean_ = data["global_mean"]
            self.global_scaler.scale_ = data["global_scale"]
            self.global_scaler.var_ = data["global_scale"] ** 2
            return True
        return False

    def process_data(self, load_cached_data=True):
        """
        Main method to load, process, scale, and return data.
        """
        # Check if caches exist
        caches_exist = (
            os.path.exists(Config.PROCESSED_TRAIN_PATH)
            and os.path.exists(Config.PROCESSED_VAL_PATH)
            and os.path.exists(Config.PROCESSED_TEST_PATH)
            and os.path.exists(Config.SCALERS_PATH)
        )

        if load_cached_data and caches_exist:
            try:
                self._load_scalers()
                train_data = self._load_cache(Config.PROCESSED_TRAIN_PATH)
                val_data = self._load_cache(Config.PROCESSED_VAL_PATH)
                test_data = self._load_cache(Config.PROCESSED_TEST_PATH)
                return train_data, val_data, test_data
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # Compute from scratch
        print("Computing features from scratch...")

        # 1. Load and process raw geometry
        train_atomic, train_global, train_targets, train_ids = self._compute_split(
            Config.TRAIN_META_PATH, is_test=False
        )
        val_atomic, val_global, val_targets, val_ids = self._compute_split(
            Config.VAL_META_PATH, is_test=False
        )
        test_atomic, test_global, _, test_ids = self._compute_split(
            Config.TEST_META_PATH, is_test=True
        )

        # 2. Fit scalers on Training Data
        # Flatten training atomic features to fit scaler
        train_atomic_flat = np.concatenate(train_atomic, axis=0)
        self.atom_scaler.fit(train_atomic_flat)
        self.global_scaler.fit(train_global)

        # 3. Apply Scaling
        # Helper to scale list of arrays
        def scale_atomic_list(a_list, scaler):
            return [scaler.transform(a) for a in a_list]

        train_atomic = scale_atomic_list(train_atomic, self.atom_scaler)
        val_atomic = scale_atomic_list(val_atomic, self.atom_scaler)
        test_atomic = scale_atomic_list(test_atomic, self.atom_scaler)

        train_global = self.global_scaler.transform(train_global)
        val_global = self.global_scaler.transform(val_global)
        test_global = self.global_scaler.transform(test_global)

        # 4. Log Transform Targets
        # log(1 + y)
        train_targets = np.log1p(train_targets)
        val_targets = np.log1p(val_targets)

        # 5. Save Cache
        self._save_cache(
            Config.PROCESSED_TRAIN_PATH,
            train_atomic,
            train_global,
            train_targets,
            train_ids,
        )
        self._save_cache(
            Config.PROCESSED_VAL_PATH, val_atomic, val_global, val_targets, val_ids
        )
        self._save_cache(
            Config.PROCESSED_TEST_PATH, test_atomic, test_global, None, test_ids
        )
        self._save_scalers()

        return (
            (train_atomic, train_global, train_targets, train_ids),
            (val_atomic, val_global, val_targets, val_ids),
            (test_atomic, test_global, None, test_ids),
        )

    def get_dataloaders(self, load_cached_data=True):
        """
        Returns train, val, test dataloaders.
        """
        train_data, val_data, test_data = self.process_data(load_cached_data)

        train_dataset = MaterialsDataset(*train_data)
        val_dataset = MaterialsDataset(*val_data)
        test_dataset = MaterialsDataset(*test_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=materials_collate_fn,
            num_workers=0,  # Avoid multiprocessing issues in some envs
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=materials_collate_fn,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=materials_collate_fn,
            num_workers=0,
        )

        return train_loader, val_loader, test_loader
