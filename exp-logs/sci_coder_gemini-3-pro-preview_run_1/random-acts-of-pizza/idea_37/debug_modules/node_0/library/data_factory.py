import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import sparse
from library.features import get_features
from library.config import RANDOM_STATE, N_JOBS
from library.utils import set_seed


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Centroid-Augmented Dual-Attention MLP.
    Handles dictionary-based input from the feature generator.
    """

    def __init__(self, data_dict):
        # Convert numpy arrays from the NpzFile/dict to FloatTensors
        self.title_emb = torch.tensor(data_dict["title_emb"], dtype=torch.float32)
        self.body_emb = torch.tensor(data_dict["body_emb"], dtype=torch.float32)
        self.history_seq = torch.tensor(data_dict["history_seq"], dtype=torch.float32)
        self.history_mask = torch.tensor(data_dict["history_mask"], dtype=torch.float32)
        self.centroid = torch.tensor(data_dict["centroid"], dtype=torch.float32)
        self.meta = torch.tensor(data_dict["meta"], dtype=torch.float32)

        # Handle labels if they exist (Train/Val), else None (Test)
        if "labels" in data_dict:
            self.labels = torch.tensor(data_dict["labels"], dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        item = {
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "history_seq": self.history_seq[idx],
            "history_mask": self.history_mask[idx],
            "centroid": self.centroid[idx],
            "meta": self.meta[idx],
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item


class DataBuilder:
    """
    Orchestrates data loading and formatting for both RF and MLP streams.
    """

    def __init__(self):
        set_seed(RANDOM_STATE)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self, load_cached_data=True):
        """
        Loads data using the FeatureGenerator.
        If data is not in memory, it fetches it (which handles caching internally).
        """
        if self.train_data is None:
            self.train_data, self.val_data, self.test_data = get_features(
                load_cached_data=load_cached_data
            )

    def _reconstruct_sparse(self, npz_file):
        """
        Helper to reconstruct a scipy.sparse.csr_matrix from NpzFile components.
        """
        data = npz_file["rf_data"]
        indices = npz_file["rf_indices"]
        indptr = npz_file["rf_indptr"]
        shape = npz_file["rf_shape"]
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def get_rf_data(self):
        """
        Returns data formatted for the Random Forest (Stream A).

        Returns:
            tuple: ((X_train, y_train), (X_val, y_val), (X_test, None))
                   where X is a sparse CSR matrix and y is a numpy array.
        """
        self.load_data()

        X_train = self._reconstruct_sparse(self.train_data)
        y_train = self.train_data["labels"]

        X_val = self._reconstruct_sparse(self.val_data)
        y_val = self.val_data["labels"]

        X_test = self._reconstruct_sparse(self.test_data)
        y_test = None  # No labels for test set

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_mlp_loaders(self, batch_size=32):
        """
        Returns PyTorch DataLoaders for the MLP (Stream B).

        Args:
            batch_size (int): Batch size for the loaders.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        self.load_data()

        train_dataset = PizzaDataset(self.train_data)
        val_dataset = PizzaDataset(self.val_data)
        test_dataset = PizzaDataset(self.test_data)

        # Determine worker count (avoid overhead for small in-memory datasets, but allow some parallelism)
        num_workers = 2 if torch.cuda.is_available() else 0
        pin_memory = True if torch.cuda.is_available() else False

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader
