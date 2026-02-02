import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library.config import BATCH_SIZE, NUM_WORKERS
from library.features import FeaturePipeline


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request Prediction task.
    Wraps pre-processed features for the MLP stream.
    """

    def __init__(self, title_emb, body_emb, hist_seq, dense, labels=None):
        """
        Args:
            title_emb (np.ndarray): SBERT embeddings of request titles.
            body_emb (np.ndarray): SBERT embeddings of request bodies.
            hist_seq (np.ndarray): Padded sequences of user history embeddings.
            dense (np.ndarray): Concatenated numerical metadata and Top-K flags.
            labels (np.ndarray, optional): Binary target labels.
        """
        # Convert numpy arrays to float tensors
        self.title_emb = torch.from_numpy(title_emb).float()
        self.body_emb = torch.from_numpy(body_emb).float()
        self.hist_seq = torch.from_numpy(hist_seq).float()
        self.dense = torch.from_numpy(dense).float()

        if labels is not None:
            self.labels = torch.from_numpy(labels).float()
        else:
            self.labels = None

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing the features for a single sample.
        Generates an attention mask for the history sequence on the fly.
        """
        hist = self.hist_seq[idx]

        # Generate mask: 1 if the embedding vector is not padding (all zeros), 0 otherwise.
        # We check if the sum of absolute values is significantly greater than 0.
        # hist shape: (Seq_Len, Emb_Dim) -> mask shape: (Seq_Len,)
        history_mask = (hist.abs().sum(dim=1) > 1e-6).float()

        item = {
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "history_seq": hist,
            "history_mask": history_mask,
            "dense_features": self.dense[idx],
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item


def custom_collate_fn(batch):
    """
    Custom collate function to stack dictionary items into batch tensors.
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        # Stack all tensors for the given key
        collated[key] = torch.stack([sample[key] for sample in batch])

    return collated


def get_dataloaders(load_cached_data=True, batch_size=BATCH_SIZE, verbose=True):
    """
    Initializes the FeaturePipeline, loads data, and returns DataLoaders.

    Args:
        load_cached_data (bool): Whether to load features from cache if available.
        batch_size (int): Batch size for DataLoaders.
        verbose (bool): Whether to print dataset statistics.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize and run pipeline
    # The pipeline handles the caching logic internally based on the flag
    pipeline = FeaturePipeline()
    _, mlp_data = pipeline.run(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = PizzaDataset(
        mlp_data["train_title_emb"],
        mlp_data["train_body_emb"],
        mlp_data["train_hist_seq"],
        mlp_data["train_dense"],
        mlp_data["y_train"],
    )

    val_dataset = PizzaDataset(
        mlp_data["val_title_emb"],
        mlp_data["val_body_emb"],
        mlp_data["val_hist_seq"],
        mlp_data["val_dense"],
        mlp_data["y_val"],
    )

    test_dataset = PizzaDataset(
        mlp_data["test_title_emb"],
        mlp_data["test_body_emb"],
        mlp_data["test_hist_seq"],
        mlp_data["test_dense"],
        labels=None,  # No labels for test set
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    if verbose:
        print(f"DataLoaders initialized:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples:   {len(val_dataset)}")
        print(f"  Test samples:  {len(test_dataset)}")

    return train_loader, val_loader, test_loader
