import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.features import FeatureExtractor, SelectiveScaler


class MaterialDataset(Dataset):
    """
    PyTorch Dataset for the AMSP-DS strategy.
    Handles loading processed features, selective scaling, and target transformation.
    """

    def __init__(self, mode="train", scaler=None, load_cached=True):
        self.mode = mode

        # 1. Determine Metadata Path
        if mode == "train":
            csv_path = Config.TRAIN_CSV
        elif mode == "val":
            csv_path = Config.VAL_CSV
        else:
            csv_path = Config.TEST_CSV

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Debugging: Slice data if enabled to speed up development
        if Config.DEBUG:
            df = df.iloc[: Config.DEBUG_SIZE]
            cache_name = f"{mode}_debug_{Config.DEBUG_SIZE}"
        else:
            cache_name = mode

        # 2. Process Geometry and Features using FeatureExtractor
        # This handles parsing XYZ files, computing neighbors, and caching results
        extractor = FeatureExtractor()
        data = extractor.process_dataset(
            df, load_cached_data=load_cached, cache_name=cache_name
        )

        # Load raw numpy arrays
        self.atomic_features = data["atomic_features"].astype(np.float32)
        self.global_features = data["global_features"].astype(np.float32)
        self.targets = data["targets"].astype(np.float32)
        self.ids = data["ids"]
        batch_indices = data["batch_indices"]

        # 3. Precompute atom slices for O(1) access in __getitem__
        # batch_indices maps every atom to a crystal index (0..N-1)
        # We assume they are sorted and contiguous (guaranteed by FeatureExtractor)
        _, counts = np.unique(batch_indices, return_counts=True)
        cumulative = np.cumsum(np.concatenate(([0], counts)))
        self.slices = []
        for i in range(len(counts)):
            self.slices.append((cumulative[i], cumulative[i + 1]))

        # 4. Selective Scaling
        # We must scale continuous features but preserve one-hot encodings
        if mode == "train":
            if scaler is None:
                self.scaler = SelectiveScaler()
                self.scaler.fit(self.atomic_features, self.global_features)
                # Save scaler for inference later
                scaler_path = os.path.join(Config.WORKING_DIR, "scalers.npz")
                self.scaler.save(scaler_path)
            else:
                self.scaler = scaler
        else:
            if scaler is None:
                # Try to load existing scaler
                scaler_path = os.path.join(Config.WORKING_DIR, "scalers.npz")
                if os.path.exists(scaler_path):
                    self.scaler = SelectiveScaler()
                    self.scaler.load(scaler_path)
                else:
                    print(
                        "Warning: No scaler provided and no cached scaler found. Features will not be scaled."
                    )
                    self.scaler = None
            else:
                self.scaler = scaler

        if self.scaler is not None:
            self.atomic_features, self.global_features = self.scaler.transform(
                self.atomic_features, self.global_features
            )

        # 5. Target Transformation (Log1p)
        # Apply log(1+y) to stabilize regression and align with RMSLE metric
        # Only for train/val. Test targets are dummy zeros.
        if mode in ["train", "val"]:
            self.targets = np.log1p(self.targets)

        # Convert to PyTorch Tensors
        self.atomic_features = torch.from_numpy(self.atomic_features)
        self.global_features = torch.from_numpy(self.global_features)
        self.targets = torch.from_numpy(self.targets)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve the slice of atoms corresponding to this crystal
        start, end = self.slices[idx]

        sample = {
            "atomic_features": self.atomic_features[start:end],  # (N_atoms_i, 17)
            "global_features": self.global_features[idx],  # (19,)
            "target": self.targets[idx],  # (2,)
            "id": self.ids[idx],
        }
        return sample


def sparse_collate_fn(batch):
    """
    Collates a list of samples into a batch for sparse processing (Deep Sets).
    Flattens atomic features and creates a batch index vector.
    """
    atomic_features_list = []
    batch_indices_list = []
    global_features_list = []
    targets_list = []
    ids_list = []

    for i, sample in enumerate(batch):
        af = sample["atomic_features"]
        atomic_features_list.append(af)

        # Create batch index vector for this crystal (all atoms get index i)
        # i ranges from 0 to batch_size - 1
        n_atoms = af.shape[0]
        batch_indices_list.append(torch.full((n_atoms,), i, dtype=torch.long))

        global_features_list.append(sample["global_features"])
        targets_list.append(sample["target"])
        ids_list.append(sample["id"])

    return {
        "atomic_features": torch.cat(atomic_features_list, dim=0),
        "batch_indices": torch.cat(batch_indices_list, dim=0),
        "global_features": torch.stack(global_features_list),
        "targets": torch.stack(targets_list),
        "ids": ids_list,
    }


def get_dataloader(mode, batch_size=Config.BATCH_SIZE, shuffle=True, scaler=None):
    """
    Factory function to create a DataLoader with the correct configuration.
    Returns (loader, scaler) if mode is 'train', else just loader.
    """
    dataset = MaterialDataset(mode=mode, scaler=scaler)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=sparse_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    if mode == "train":
        return loader, dataset.scaler
    else:
        return loader
