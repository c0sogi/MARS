import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.feature_engineering import run_feature_engineering
from library.config import TrainingConfig


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Triple-Branch MLP (Stream B).
    Handles SBERT embeddings, Community Sequences, and Numerical Metadata.
    """

    def __init__(self, data_dict, is_test=False):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays from feature engineering.
                              Expected keys: 'sbert', 'community', 'meta_num', 'y' (if not test), 'ids' (if test).
            is_test (bool): Flag indicating if this is the test set (no targets).
        """
        self.is_test = is_test

        # Convert inputs to appropriate tensors
        self.sbert = torch.tensor(data_dict["sbert"], dtype=torch.float32)
        self.community = torch.tensor(data_dict["community"], dtype=torch.long)
        self.meta_num = torch.tensor(data_dict["meta_num"], dtype=torch.float32)

        if not self.is_test:
            # Targets for binary classification (BCEWithLogitsLoss expects Float)
            # Ensure shape is (N, 1)
            self.y = torch.tensor(data_dict["y"], dtype=torch.float32).unsqueeze(1)
        else:
            # Store IDs for submission generation
            self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.sbert)

    def __getitem__(self, idx):
        item = {
            "semantic_input": self.sbert[idx],
            "community_input": self.community[idx],
            "meta_input": self.meta_num[idx],
        }

        if not self.is_test:
            item["target"] = self.y[idx]

        return item


def _convert_npz_to_dict(data_obj):
    """
    Helper to convert NpzFile object to a standard dictionary of in-memory arrays.
    Closes the NpzFile handle safely.
    """
    if isinstance(data_obj, dict):
        return data_obj

    # If it's a numpy NpzFile
    try:
        data_dict = {k: data_obj[k] for k in data_obj.files}
        data_obj.close()
        return data_dict
    except AttributeError:
        # Fallback if it's already some other object or doesn't support .files/.close
        return data_obj


def load_and_preprocess(load_cached_data: bool = True):
    """
    Orchestrates data loading and preprocessing.
    Calls the feature engineering pipeline and ensures data is loaded into memory.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (Stream A Data, Stream B Data)
               Stream A: (train_dict, val_dict, test_dict) for Random Forest
               Stream B: (train_dict, val_dict, test_dict) for MLP
    """
    # Run pipeline
    raw_output = run_feature_engineering(load_cached_data=load_cached_data)

    # Unpack tuple: (a_tr, a_val, a_te, b_tr, b_val, b_te)
    # Convert NpzFiles to dicts to avoid file handle issues in DataLoaders
    processed_data = tuple(_convert_npz_to_dict(item) for item in raw_output)

    # Group by stream
    stream_a = processed_data[0:3]
    stream_b = processed_data[3:6]

    return stream_a, stream_b


def get_pytorch_dataloaders(
    train_data, val_data, test_data, batch_size=None, num_workers=None
):
    """
    Creates PyTorch DataLoaders for Stream B (MLP) data.

    Args:
        train_data (dict): Stream B training dictionary.
        val_data (dict): Stream B validation dictionary.
        test_data (dict): Stream B test dictionary.
        batch_size (int, optional): Batch size. Defaults to TrainingConfig.BATCH_SIZE.
        num_workers (int, optional): Number of workers. Defaults to TrainingConfig.NUM_WORKERS.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = TrainingConfig.BATCH_SIZE
    if num_workers is None:
        num_workers = TrainingConfig.NUM_WORKERS

    # Create Datasets
    train_dataset = PizzaDataset(train_data, is_test=False)
    val_dataset = PizzaDataset(val_data, is_test=False)
    test_dataset = PizzaDataset(test_data, is_test=True)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
