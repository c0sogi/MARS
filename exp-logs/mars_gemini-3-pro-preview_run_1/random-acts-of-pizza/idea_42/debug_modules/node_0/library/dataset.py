import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from library import config, utils, features_text, features_meta


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Persona-Aware Skip-Gated MLP.
    Handles multiple input branches: Title, Body, History Sequence, Persona Centroid, and Metadata.
    """

    def __init__(self, text_features, meta_features, split, labels=None):
        """
        Args:
            text_features (dict): Dictionary containing text embeddings (SBERT).
            meta_features (dict): Dictionary containing metadata and history sequences.
            split (str): One of 'train', 'val', 'test'.
            labels (array-like, optional): Target labels.
        """
        self.split = split
        self.labels = labels

        # Extract features based on split key prefix
        # 1. Semantic Features (SBERT)
        self.title_emb = text_features[f"{split}_title_emb"]
        self.body_emb = text_features[f"{split}_body_emb"]
        self.hist_centroid = text_features[f"{split}_hist_centroid"]

        # 2. Sequence Features
        self.hist_seq = meta_features[f"{split}_hist_seq"]

        # 3. Dense Metadata (Arcsinh + Scaled)
        self.metadata = meta_features[f"{split}_meta_mlp"]

        # Validation
        assert len(self.title_emb) == len(self.metadata), "Feature length mismatch"
        if self.labels is not None:
            assert len(self.labels) == len(self.title_emb), "Label length mismatch"

    def __len__(self):
        return len(self.title_emb)

    def __getitem__(self, idx):
        # Convert to torch tensors
        title_tensor = torch.tensor(self.title_emb[idx], dtype=torch.float32)
        body_tensor = torch.tensor(self.body_emb[idx], dtype=torch.float32)
        hist_seq_tensor = torch.tensor(self.hist_seq[idx], dtype=torch.float32)
        persona_tensor = torch.tensor(self.hist_centroid[idx], dtype=torch.float32)
        meta_tensor = torch.tensor(self.metadata[idx], dtype=torch.float32)

        # Create Padding Mask for History Sequence
        # PyTorch MultiheadAttention key_padding_mask expects True for positions to IGNORE (padding)
        # We assume zero-vectors in the sequence imply padding.
        # Shape: (Seq_Len,)
        # Check if the sum of absolute values in the embedding dimension is close to 0
        is_padding = torch.sum(torch.abs(hist_seq_tensor), dim=-1) < 1e-6

        item = {
            "title_emb": title_tensor,
            "body_emb": body_tensor,
            "history_seq": hist_seq_tensor,
            "history_mask": is_padding,  # BoolTensor: True where padding exists
            "persona_centroid": persona_tensor,
            "dense_metadata": meta_tensor,
        }

        if self.labels is not None:
            # Target needs to be float for BCEWithLogitsLoss usually, or long for CrossEntropy
            # Given binary classification and usually BCE, float is safer.
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return item, label_tensor

        return item


def create_dataloaders(
    debug_size=None, load_cached_data=True, batch_size=config.MLP_BATCH_SIZE
):
    """
    Orchestrates data loading, feature generation, and DataLoader creation.

    Args:
        debug_size (int, optional): If set, truncates data for debugging.
        load_cached_data (bool): Whether to use cached features.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Raw Data (Metadata CSVs)
    # We need the dataframes to pass to feature generators, even if features are cached,
    # because feature generators might need to check lengths or recompute if cache missing.
    # However, the feature functions handle cache checking internally.
    # We load DFs primarily to get the labels and ensure alignment.
    train_df, val_df, test_df = utils.load_data(
        return_val=True,
        parse_list_cols=["requester_subreddits_at_request"],
        debug_size=debug_size,
    )

    # 2. Generate/Load Features
    # Text Features (SBERT, TFIDF, Sentiment) - We only need SBERT related ones for MLP
    text_feats = features_text.generate_text_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # Meta Features (History Seq, Consistency, TopK, Dense Meta)
    meta_feats = features_meta.generate_meta_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 3. Extract Labels
    # Target column: 'requester_received_pizza'
    # Convert boolean to float (0.0, 1.0)
    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values
    # Test set has no labels for prediction

    # 4. Create Datasets
    train_dataset = PizzaDataset(text_feats, meta_feats, split="train", labels=y_train)
    val_dataset = PizzaDataset(text_feats, meta_feats, split="val", labels=y_val)
    test_dataset = PizzaDataset(text_feats, meta_feats, split="test", labels=None)

    # 5. Create DataLoaders
    # Shuffle training data
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues in some envs, set to >0 if safe
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
