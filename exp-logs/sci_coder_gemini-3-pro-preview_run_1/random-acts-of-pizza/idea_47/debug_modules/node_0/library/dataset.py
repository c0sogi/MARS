import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.feature_engineering import FeaturePipeline
from library.config import MLP_BATCH_SIZE, NUM_WORKERS


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request dataset (MLP Stream).
    """

    def __init__(
        self,
        title_emb,
        body_emb,
        history_emb,
        history_mask,
        centroid_emb,
        metadata,
        labels=None,
    ):
        """
        Args:
            title_emb (np.ndarray): SBERT embeddings of request titles (N, D).
            body_emb (np.ndarray): SBERT embeddings of request bodies (N, D).
            history_emb (np.ndarray): SBERT embeddings of user history (N, L, D).
            history_mask (np.ndarray): Attention mask for user history (N, L).
            centroid_emb (np.ndarray): Centroid of user history embeddings (N, D).
            metadata (np.ndarray): Preprocessed metadata features (N, M).
            labels (np.ndarray, optional): Target labels (N,).
        """
        self.title_emb = title_emb
        self.body_emb = body_emb
        self.history_emb = history_emb
        self.history_mask = history_mask
        self.centroid_emb = centroid_emb
        self.metadata = metadata
        self.labels = labels

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        item = {
            "title_emb": torch.tensor(self.title_emb[idx], dtype=torch.float32),
            "body_emb": torch.tensor(self.body_emb[idx], dtype=torch.float32),
            "history_emb": torch.tensor(self.history_emb[idx], dtype=torch.float32),
            "history_mask": torch.tensor(self.history_mask[idx], dtype=torch.float32),
            "centroid_emb": torch.tensor(self.centroid_emb[idx], dtype=torch.float32),
            "metadata": torch.tensor(self.metadata[idx], dtype=torch.float32),
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


def create_dataloaders(load_cached_data=True, batch_size=MLP_BATCH_SIZE):
    """
    Creates DataLoaders for train, validation, and test sets using FeaturePipeline.

    Args:
        load_cached_data (bool): Whether to load features from cache.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize and run feature pipeline
    pipeline = FeaturePipeline()
    # We only need mlp_out for this dataset
    _, mlp_out = pipeline.run(load_cached_data=load_cached_data)

    # Create Train Dataset
    train_dataset = PizzaDataset(
        title_emb=mlp_out["train_title"],
        body_emb=mlp_out["train_body"],
        history_emb=mlp_out["train_hist"],
        history_mask=mlp_out["train_hist_mask"],
        centroid_emb=mlp_out["train_centroid"],
        metadata=mlp_out["train_meta"],
        labels=mlp_out["train_y"],
    )

    # Create Validation Dataset
    val_dataset = PizzaDataset(
        title_emb=mlp_out["val_title"],
        body_emb=mlp_out["val_body"],
        history_emb=mlp_out["val_hist"],
        history_mask=mlp_out["val_hist_mask"],
        centroid_emb=mlp_out["val_centroid"],
        metadata=mlp_out["val_meta"],
        labels=mlp_out["val_y"],
    )

    # Create Test Dataset
    test_dataset = PizzaDataset(
        title_emb=mlp_out["test_title"],
        body_emb=mlp_out["test_body"],
        history_emb=mlp_out["test_hist"],
        history_mask=mlp_out["test_hist_mask"],
        centroid_emb=mlp_out["test_centroid"],
        metadata=mlp_out["test_meta"],
        labels=None,  # Test set has no labels
    )

    # Create DataLoaders
    # Pin memory is generally safe and recommended for CUDA
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
