import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
from library.config import BATCH_SIZE, NUM_WORKERS, NUM_TAGS
from library.data_processor import prepare_data


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for Stack Exchange Tag Prediction.
    Handles Dense sequences for the Deep component and Sparse TF-IDF vectors for the Wide component.
    """

    def __init__(self, X_wide, X_deep, y=None):
        """
        Args:
            X_wide: scipy.sparse.csr_matrix of shape (n_samples, vocab_wide)
            X_deep: np.ndarray of shape (n_samples, max_len)
            y: scipy.sparse.csr_matrix of shape (n_samples, num_tags) or None
        """
        self.X_wide = X_wide
        self.X_deep = X_deep
        self.y = y

        # Ensure X_wide is csr for fast row slicing
        if not sp.isspmatrix_csr(self.X_wide):
            self.X_wide = self.X_wide.tocsr()

    def __len__(self):
        return len(self.X_deep)

    def __getitem__(self, idx):
        # Deep Component: Integer sequence
        deep_seq = torch.tensor(self.X_deep[idx], dtype=torch.long)

        # Wide Component: Sparse TF-IDF
        # Extract indices and data directly from the sparse matrix row
        # Slicing a CSR matrix returns a CSR matrix
        row = self.X_wide[idx]

        # We need indices (feature IDs) and values (TF-IDF weights)
        # row.indices and row.data give us exactly that for the non-zero elements
        wide_indices = torch.tensor(row.indices, dtype=torch.long)
        wide_values = torch.tensor(row.data, dtype=torch.float32)

        # Target
        if self.y is not None:
            # y is also sparse, convert to dense tensor for loss calculation
            # Slicing gives 1xNUM_TAGS sparse matrix
            target = torch.tensor(self.y[idx].toarray().flatten(), dtype=torch.float32)
        else:
            # Dummy target for test set
            target = torch.zeros(NUM_TAGS, dtype=torch.float32)

        return {
            "deep_seq": deep_seq,
            "wide_indices": wide_indices,
            "wide_values": wide_values,
            "target": target,
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sparse inputs for EmbeddingBag.

    Returns:
        dict containing:
            - deep_seq: (batch_size, max_len)
            - wide_indices: (total_nonzero_elements,)
            - wide_values: (total_nonzero_elements,)
            - wide_offsets: (batch_size,)
            - target: (batch_size, num_tags)
    """
    deep_seqs = []
    wide_indices_list = []
    wide_values_list = []
    targets = []

    # Offsets for EmbeddingBag:
    # offsets[i] is the starting index in the flattened indices/values tensor for the i-th sample.
    offsets = [0]

    for sample in batch:
        deep_seqs.append(sample["deep_seq"])
        targets.append(sample["target"])

        wide_indices_list.append(sample["wide_indices"])
        wide_values_list.append(sample["wide_values"])

        # The next offset is current offset + length of current sparse vector
        offsets.append(offsets[-1] + len(sample["wide_indices"]))

    # Remove the last entry from offsets (it points past the end of the last sample)
    offsets = torch.tensor(offsets[:-1], dtype=torch.long)

    # Concatenate all lists
    deep_seq_batch = torch.stack(deep_seqs)
    target_batch = torch.stack(targets)

    # Flatten wide features
    if wide_indices_list:
        wide_indices_batch = torch.cat(wide_indices_list)
        wide_values_batch = torch.cat(wide_values_list)
    else:
        # Handle empty batch edge case
        wide_indices_batch = torch.tensor([], dtype=torch.long)
        wide_values_batch = torch.tensor([], dtype=torch.float32)

    return {
        "deep_seq": deep_seq_batch,
        "wide_indices": wide_indices_batch,
        "wide_values": wide_values_batch,
        "wide_offsets": offsets,
        "target": target_batch,
    }


def get_dataloaders(load_cached_data=True):
    """
    Loads data and returns DataLoaders for train, val, and test.
    """
    # Load processed data
    # This calls the caching logic in data_processor
    data = prepare_data(load_cached_data=load_cached_data)

    (
        X_wide_train,
        X_deep_train,
        y_train,
        X_wide_val,
        X_deep_val,
        y_val,
        X_wide_test,
        X_deep_test,
        test_ids,
        preprocessor,
    ) = data

    # Create Datasets
    train_dataset = StackExchangeDataset(X_wide_train, X_deep_train, y_train)
    val_dataset = StackExchangeDataset(X_wide_val, X_deep_val, y_val)
    test_dataset = StackExchangeDataset(X_wide_test, X_deep_test, y=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids, preprocessor
