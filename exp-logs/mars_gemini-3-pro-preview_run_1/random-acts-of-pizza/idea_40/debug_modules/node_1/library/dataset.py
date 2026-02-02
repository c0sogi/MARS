import torch
from torch.utils.data import Dataset
import numpy as np
from library import config
from library.features import FeatureEngineer


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request prediction task (MLP Stream).
    Handles multi-modal inputs: SBERT embeddings, User History Sequences, Metadata, and Consistency Scalars.
    """

    def __init__(
        self, title_emb, body_emb, hist_seq, hist_mask, meta, cons, labels=None
    ):
        """
        Args:
            title_emb (np.ndarray): SBERT embeddings of request titles (N, 384).
            body_emb (np.ndarray): SBERT embeddings of request bodies (N, 384).
            hist_seq (np.ndarray): User history embedding sequences (N, 20, 384).
            hist_mask (np.ndarray): Attention masks for history sequences (N, 20).
            meta (np.ndarray): Scaled metadata features (N, D_meta).
            cons (np.ndarray): Consistency scalars (N, 2).
            labels (np.ndarray, optional): Target labels (N,).
        """
        # Convert inputs to float32 tensors
        self.title_emb = torch.tensor(title_emb, dtype=torch.float32)
        self.body_emb = torch.tensor(body_emb, dtype=torch.float32)
        self.hist_seq = torch.tensor(hist_seq, dtype=torch.float32)
        self.hist_mask = torch.tensor(hist_mask, dtype=torch.float32)
        self.meta = torch.tensor(meta, dtype=torch.float32)
        self.cons = torch.tensor(cons, dtype=torch.float32)

        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing all features for a single sample.
        """
        sample = {
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "hist_seq": self.hist_seq[idx],
            "hist_mask": self.hist_mask[idx],
            "meta": self.meta[idx],
            "cons": self.cons[idx],
        }

        if self.labels is not None:
            sample["label"] = self.labels[idx]

        return sample


def get_mlp_datasets(load_cached_data=True, debug=config.DEBUG):
    """
    Generates and returns the training, validation, and test datasets for the MLP model.
    Interfaces with the FeatureEngineer to process or load data.

    Args:
        load_cached_data (bool): Whether to load features from cache if available.
        debug (bool): If True, uses a smaller subset of data for debugging.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Temporarily override config.DEBUG to control data sampling in utils.load_data
    original_debug = config.DEBUG
    config.DEBUG = debug

    # If debugging, force re-computation (load_cached_data=False) to avoid loading
    # the full dataset from cache, as cache filenames do not distinguish debug/full.
    if debug:
        load_cached_data = False

    try:
        fe = FeatureEngineer()
        # Retrieve MLP-specific features (stream B)
        # We ignore the first return value (rf_out) as this dataset is for the MLP
        _, mlp_out = fe.process_data(load_cached_data=load_cached_data)
    finally:
        # Restore original configuration
        config.DEBUG = original_debug

    # Helper function to initialize PizzaDataset for a specific split
    def create_dataset(split_prefix, has_labels=True):
        return PizzaDataset(
            title_emb=mlp_out[f"{split_prefix}_title_emb"],
            body_emb=mlp_out[f"{split_prefix}_body_emb"],
            hist_seq=mlp_out[f"{split_prefix}_hist_seq"],
            hist_mask=mlp_out[f"{split_prefix}_hist_mask"],
            meta=mlp_out[f"{split_prefix}_meta"],
            cons=mlp_out[f"{split_prefix}_cons"],
            labels=mlp_out[f"y_{split_prefix}"] if has_labels else None,
        )

    # Create datasets
    train_dataset = create_dataset("train", has_labels=True)
    val_dataset = create_dataset("val", has_labels=True)
    test_dataset = create_dataset("test", has_labels=False)

    return train_dataset, val_dataset, test_dataset
