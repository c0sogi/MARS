import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from library.config import MLP_PARAMS, NUM_WORKERS, WORKING_DIR, RANDOM_STATE
from library.utils import load_from_cache, seed_everything


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Hybrid Ensemble MLP (Stream B).

    Inputs:
        - metadata: Normalized numeric features + Top-K binary indicators.
        - title_emb: SBERT embeddings of the request title.
        - body_emb: SBERT embeddings of the request body.
        - history_seq: Sequence of SBERT embeddings for user's past subreddits.
        - history_mask: Attention mask for the history sequence.
        - history_centroid: Mean embedding of user's history (Global Persona).
        - labels: Target variable (optional, for train/val).
    """

    def __init__(
        self,
        metadata,
        title_emb,
        body_emb,
        history_seq,
        history_mask,
        history_centroid,
        labels=None,
    ):
        self.metadata = torch.FloatTensor(metadata)
        self.title_emb = torch.FloatTensor(title_emb)
        self.body_emb = torch.FloatTensor(body_emb)
        self.history_seq = torch.FloatTensor(history_seq)
        self.history_mask = torch.FloatTensor(history_mask)
        self.history_centroid = torch.FloatTensor(history_centroid)

        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = {
            "metadata": self.metadata[idx],
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "history_seq": self.history_seq[idx],
            "history_mask": self.history_mask[idx],
            "history_centroid": self.history_centroid[idx],
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item


def get_dataloaders(batch_size=MLP_PARAMS["batch_size"]):
    """
    Loads cached features and constructs DataLoaders for Train, Val, and Test.

    Requires 'features_mlp.npz' and 'text_features.npz' to be present in the cache.
    """
    seed_everything(RANDOM_STATE)

    # 1. Load Cached Data
    # Tabular features (metadata + top-k)
    mlp_features = load_from_cache("features_mlp.npz")
    # Text embeddings (title, body, history)
    text_features = load_from_cache("text_features.npz")

    if mlp_features is None or text_features is None:
        raise FileNotFoundError(
            "Cached features not found. Please run feature_engineering.py first."
        )

    # 2. Construct Datasets

    # --- Train ---
    train_dataset = PizzaDataset(
        metadata=mlp_features["X_train"],
        title_emb=text_features["train_title_emb"],
        body_emb=text_features["train_body_emb"],
        history_seq=text_features["train_history_seq"],
        history_mask=text_features["train_history_mask"],
        history_centroid=text_features["train_centroid"],
        labels=mlp_features["y_train"],
    )

    # --- Validation ---
    val_dataset = PizzaDataset(
        metadata=mlp_features["X_val"],
        title_emb=text_features["val_title_emb"],
        body_emb=text_features["val_body_emb"],
        history_seq=text_features["val_history_seq"],
        history_mask=text_features["val_history_mask"],
        history_centroid=text_features["val_centroid"],
        labels=mlp_features["y_val"],
    )

    # --- Test ---
    test_dataset = PizzaDataset(
        metadata=mlp_features["X_test"],
        title_emb=text_features["test_title_emb"],
        body_emb=text_features["test_body_emb"],
        history_seq=text_features["test_history_seq"],
        history_mask=text_features["test_history_mask"],
        history_centroid=text_features["test_centroid"],
        labels=None,
    )

    # 3. Construct DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
