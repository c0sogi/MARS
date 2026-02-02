import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_metadata_splits
from library.semantic_engine import SemanticProcessor
from library.feature_engineering import FeatureEngineer


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Orthogonal Skip-Gated MLP.
    Aggregates semantic embeddings and processed metadata.
    """

    def __init__(self, semantic_features, tabular_features, labels=None):
        """
        Args:
            semantic_features (dict): Output from SemanticProcessor.
            tabular_features (dict): Output from FeatureEngineer.
            labels (pd.Series or np.array, optional): Target labels.
        """
        self.labels = None
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)

        # Extract Semantic Features
        # Convert to tensors
        self.title_embs = torch.tensor(
            semantic_features["title_embs"], dtype=torch.float32
        )
        self.body_embs = torch.tensor(
            semantic_features["body_embs"], dtype=torch.float32
        )
        self.centroid_embs = torch.tensor(
            semantic_features["centroid_embs"], dtype=torch.float32
        )
        self.history_seq_embs = torch.tensor(
            semantic_features["history_seq_embs"], dtype=torch.float32
        )
        self.history_mask = torch.tensor(
            semantic_features["history_mask"], dtype=torch.float32
        )
        self.topic_consistency = torch.tensor(
            semantic_features["topic_consistency"], dtype=torch.float32
        )
        self.narrative_consistency = torch.tensor(
            semantic_features["narrative_consistency"], dtype=torch.float32
        )

        # Extract MLP Metadata
        # FeatureEngineer provides 'metadata_mlp' which is Arcsinh + Scaled
        self.metadata_mlp = torch.tensor(
            tabular_features["metadata_mlp"], dtype=torch.float32
        )

        # Verification
        assert len(self.title_embs) == len(self.metadata_mlp), "Feature length mismatch"
        if self.labels is not None:
            assert len(self.labels) == len(self.title_embs), "Label length mismatch"

    def __len__(self):
        return len(self.title_embs)

    def __getitem__(self, idx):
        """
        Returns a dictionary of inputs matching the signature of OrthogonalSkipGatedMLP.forward
        and the label (if available).
        """
        inputs = {
            "title_embs": self.title_embs[idx],
            "body_embs": self.body_embs[idx],
            "history_seq_embs": self.history_seq_embs[idx],
            "history_mask": self.history_mask[idx],
            "centroid_embs": self.centroid_embs[idx],
            "topic_consistency": self.topic_consistency[idx],
            "narrative_consistency": self.narrative_consistency[idx],
            "metadata_mlp": self.metadata_mlp[idx],
        }

        if self.labels is not None:
            return inputs, self.labels[idx]
        else:
            return inputs


def get_dataloaders(
    batch_size=Config.MLP_BATCH_SIZE,
    load_cached_data=True,
    num_workers=0,
):
    """
    Orchestrates the creation of DataLoaders for Train, Val, and Test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to use cached features.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader, feature_dims)
            feature_dims is a dict containing input dimensions needed for model init.
    """
    # 1. Load Metadata
    train_df, val_df, test_df = load_metadata_splits()

    # 2. Initialize Processors
    semantic_processor = SemanticProcessor()
    feature_engineer = FeatureEngineer()

    # 3. Process Splits
    # We process all splits to ensure transformers are fitted on train and applied to others
    # Note: FeatureEngineer fits on train data internally when process_split is called or manually.
    # To be safe and follow the pipeline, we process train first.

    # --- Train ---
    print("Processing Train Split...")
    train_sem = semantic_processor.process_split(
        train_df, Config.TRAIN_JSON_PATH, "train", load_cached_data
    )
    train_tab = feature_engineer.process_split(
        train_df, "train", load_cached_data, semantic_features=train_sem
    )
    train_labels = train_df[Config.TARGET_COL].values.astype(np.float32)

    # --- Val ---
    print("Processing Val Split...")
    val_sem = semantic_processor.process_split(
        val_df, Config.TRAIN_JSON_PATH, "val", load_cached_data
    )
    val_tab = feature_engineer.process_split(
        val_df, "val", load_cached_data, semantic_features=val_sem
    )
    val_labels = val_df[Config.TARGET_COL].values.astype(np.float32)

    # --- Test ---
    print("Processing Test Split...")
    test_sem = semantic_processor.process_split(
        test_df, Config.TEST_JSON_PATH, "test", load_cached_data
    )
    test_tab = feature_engineer.process_split(
        test_df, "test", load_cached_data, semantic_features=test_sem
    )
    # No labels for test

    # 4. Create Datasets
    train_dataset = PizzaDataset(train_sem, train_tab, train_labels)
    val_dataset = PizzaDataset(val_sem, val_tab, val_labels)
    test_dataset = PizzaDataset(test_sem, test_tab, labels=None)

    # 5. Create DataLoaders
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

    # 6. Determine Input Dimensions for Model Initialization
    # We need the dimension of metadata_mlp
    metadata_dim = train_tab["metadata_mlp"].shape[1]
    feature_dims = {"metadata_dim": metadata_dim}

    return train_loader, val_loader, test_loader, feature_dims
