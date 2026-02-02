import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Multi-Modal Pizza Request Predictor.
    Handles Title, Body, History Sequences, Centroids, Metadata, and Consistency scores.
    """

    def __init__(self, data_dict, is_test=False):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays for a specific split
                              (output from FeatureEngineer).
            is_test (bool): Whether this is the test set (no targets).
        """
        self.is_test = is_test

        # Unpack data
        self.title_emb = data_dict["X_mlp_title"]
        self.body_emb = data_dict["X_mlp_body"]
        self.history_seq = data_dict["X_mlp_history"]
        self.centroid = data_dict["X_mlp_centroid"]
        self.meta = data_dict["X_mlp_meta"]
        self.consistency = data_dict["X_mlp_consistency"]

        if not self.is_test:
            self.y = data_dict["y"]

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        # 1. Semantic Features
        title = torch.tensor(self.title_emb[idx], dtype=torch.float32)
        body = torch.tensor(self.body_emb[idx], dtype=torch.float32)

        # 2. History Sequence & Masking
        # Shape: (Seq_Len, Emb_Dim)
        history = torch.tensor(self.history_seq[idx], dtype=torch.float32)

        # Create Mask: True for padding (where vector is all zeros), False for valid data
        # Check if sum of absolute values across embedding dim is effectively 0
        # This is used for key_padding_mask in PyTorch Attention (True = ignore)
        history_mask = torch.abs(history).sum(dim=-1) == 0

        # 3. Global Context & Metadata
        centroid = torch.tensor(self.centroid[idx], dtype=torch.float32)
        meta = torch.tensor(self.meta[idx], dtype=torch.float32)
        consistency = torch.tensor(self.consistency[idx], dtype=torch.float32)

        sample = {
            "title": title,
            "body": body,
            "history": history,
            "history_mask": history_mask,
            "centroid": centroid,
            "meta": meta,
            "consistency": consistency,
        }

        # 4. Target
        if not self.is_test:
            # BCELoss expects float target
            target = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
            sample["target"] = target

        return sample


def create_dataloaders(processed_data, batch_size=None, num_workers=None):
    """
    Factory function to create DataLoaders for train, val, and test.

    Args:
        processed_data (dict): Output from FeatureEngineer.run(), containing 'train', 'val', 'test' dicts.
        batch_size (int, optional): Batch size override. Defaults to Config.MLP_BATCH_SIZE.
        num_workers (int, optional): Number of workers override. Defaults to Config.NUM_WORKERS.

    Returns:
        train_loader, val_loader, test_loader
    """
    bs = batch_size if batch_size is not None else Config.MLP_BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    # Initialize Datasets
    train_dataset = PizzaDataset(processed_data["train"], is_test=False)
    val_dataset = PizzaDataset(processed_data["val"], is_test=False)
    test_dataset = PizzaDataset(processed_data["test"], is_test=True)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
