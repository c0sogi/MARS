import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import arcsinh_transform, set_seed
from library.data_loader import load_dataset
from library.feature_engineering import generate_features
from library.semantic_processing import SemanticEngine


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request Prediction task.
    Serves:
    1. Metadata (Dense vector: Tabular features + Topic Ratios + Consistency)
    2. Request Embedding (SBERT vector)
    3. History Embeddings (Sequence of SBERT vectors)
    4. History Mask (For attention mechanism)
    5. Target (Label)
    """

    def __init__(
        self,
        metadata,
        request_emb,
        history_emb,
        history_mask,
        labels=None,
    ):
        """
        Args:
            metadata (np.ndarray): Normalized dense features (N, meta_dim).
            request_emb (np.ndarray): SBERT embeddings of request text (N, 384).
            history_emb (np.ndarray): History embeddings (N, seq_len, 384).
            history_mask (np.ndarray): Mask for history sequence (N, seq_len).
            labels (np.ndarray, optional): Target labels (N,).
        """
        self.metadata = torch.FloatTensor(metadata)
        self.request_emb = torch.FloatTensor(request_emb)
        self.history_emb = torch.FloatTensor(history_emb)
        self.history_mask = torch.FloatTensor(history_mask)

        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = {
            "metadata": self.metadata[idx],
            "request_emb": self.request_emb[idx],
            "history_emb": self.history_emb[idx],
            "history_mask": self.history_mask[idx],
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item


def collate_fn(batch):
    """
    Custom collate function to stack batch items.
    Since history embeddings are already pre-padded in semantic_processing,
    this mainly handles stacking into tensors.
    """
    metadata = torch.stack([item["metadata"] for item in batch])
    request_emb = torch.stack([item["request_emb"] for item in batch])
    history_emb = torch.stack([item["history_emb"] for item in batch])
    history_mask = torch.stack([item["history_mask"] for item in batch])

    batch_dict = {
        "metadata": metadata,
        "request_emb": request_emb,
        "history_emb": history_emb,
        "history_mask": history_mask,
    }

    if "label" in batch[0]:
        labels = torch.stack([item["label"] for item in batch])
        batch_dict["label"] = labels

    return batch_dict


def get_dataloaders(batch_size=Config.MLP_PARAMS["batch_size"], load_cached_data=True):
    """
    Prepares data and returns PyTorch DataLoaders for Train, Val, and Test.

    Pipeline:
    1. Load Data (CSV -> DF)
    2. Feature Engineering (DF -> DF with extra cols)
    3. Semantic Processing (DF -> Dict of Embeddings/Topics)
    4. Preprocessing:
       - Merge Tabular + Topic Ratios + Consistency
       - Arcsinh Transform
       - StandardScaler
    5. Dataset & DataLoader creation

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
               input_dim is the size of the metadata vector.
    """
    set_seed(Config.SEED)

    # 1. Load Base Data
    train_df, val_df, test_df = load_dataset(load_cached_data=load_cached_data)

    # 2. Generate Tabular Features
    train_df, val_df, test_df = generate_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 3. Generate Semantic Features
    semantic_engine = SemanticEngine()
    semantic_data = semantic_engine.process(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 4. Prepare Metadata for MLP
    # We need to combine:
    #   a. Numerical columns from DataFrame (excluding IDs, targets, leakage)
    #   b. Topic Ratios (from semantic_data)
    #   c. Consistency Score (from semantic_data)

    # Identify numerical columns to keep
    drop_cols = set(Config.DROP_COLS)
    numeric_cols = [
        c
        for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in drop_cols
    ]

    print(f"Selected {len(numeric_cols)} numerical features from tabular data.")

    def prepare_metadata_matrix(df, semantic_subset):
        # Extract tabular part
        tabular = df[numeric_cols].values.astype(np.float32)

        # Extract semantic parts
        topic_ratios = semantic_subset["topic_ratios"]  # (N, K)
        consistency = semantic_subset["consistency"].reshape(-1, 1)  # (N, 1)

        # Concatenate
        # Result: [Tabular, Topics, Consistency]
        combined = np.hstack([tabular, topic_ratios, consistency])
        return combined

    X_train_raw = prepare_metadata_matrix(train_df, semantic_data["train"])
    X_val_raw = prepare_metadata_matrix(val_df, semantic_data["val"])
    X_test_raw = prepare_metadata_matrix(test_df, semantic_data["test"])

    # 5. Preprocessing (Arcsinh + Scaling)
    print("Applying Arcsinh transformation and StandardScaler to metadata...")

    # Arcsinh (handles heavy tails)
    X_train_asinh = np.arcsinh(X_train_raw)
    X_val_asinh = np.arcsinh(X_val_raw)
    X_test_asinh = np.arcsinh(X_test_raw)

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_asinh)
    X_val_scaled = scaler.transform(X_val_asinh)
    X_test_scaled = scaler.transform(X_test_asinh)

    metadata_dim = X_train_scaled.shape[1]
    print(f"Final Metadata Input Dimension: {metadata_dim}")

    # 6. Extract Targets
    y_train = train_df[Config.TARGET_COL].values.astype(np.float32)
    y_val = val_df[Config.TARGET_COL].values.astype(np.float32)
    # Test set has no target for prediction

    # 7. Create Datasets
    train_dataset = PizzaDataset(
        metadata=X_train_scaled,
        request_emb=semantic_data["train"]["sbert_request"],
        history_emb=semantic_data["train"]["history_emb"],
        history_mask=semantic_data["train"]["history_mask"],
        labels=y_train,
    )

    val_dataset = PizzaDataset(
        metadata=X_val_scaled,
        request_emb=semantic_data["val"]["sbert_request"],
        history_emb=semantic_data["val"]["history_emb"],
        history_mask=semantic_data["val"]["history_mask"],
        labels=y_val,
    )

    test_dataset = PizzaDataset(
        metadata=X_test_scaled,
        request_emb=semantic_data["test"]["sbert_request"],
        history_emb=semantic_data["test"]["history_emb"],
        history_mask=semantic_data["test"]["history_mask"],
        labels=None,
    )

    # 8. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, metadata_dim
