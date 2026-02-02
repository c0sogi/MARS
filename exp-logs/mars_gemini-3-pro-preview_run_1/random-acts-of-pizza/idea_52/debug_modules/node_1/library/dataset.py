import torch
import numpy as np
from torch.utils.data import Dataset


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Pizza Request Success Prediction MLP.

    This dataset wraps the pre-computed SBERT embeddings and tabular metadata,
    providing a unified interface for the DataLoader. It handles the dual-query
    structure (Title, Body) and the user history sequence for the interaction-enhanced
    architecture.
    """

    def __init__(self, sbert_features, tabular_features, labels=None):
        """
        Initializes the dataset with feature dictionaries and optional labels.

        Args:
            sbert_features (dict): Dictionary containing SBERT embeddings from TextEncoder.
                Expected keys: 'title_emb', 'body_emb', 'hist_seq', 'hist_mask', 'hist_centroid'.
            tabular_features (dict): Dictionary containing tabular features from FeatureEngineer.
                Expected keys: 'mlp_metadata'.
            labels (array-like, optional): Target labels (0/1). Defaults to None for inference.
        """
        # 1. Store Text/Semantic Features
        self.title_emb = sbert_features["title_emb"]
        self.body_emb = sbert_features["body_emb"]
        self.hist_seq = sbert_features["hist_seq"]
        self.hist_mask = sbert_features["hist_mask"]
        self.hist_centroid = sbert_features["hist_centroid"]

        # 2. Store Metadata
        # We specifically use 'mlp_metadata' which is Arcsinh transformed and Scaled
        self.metadata = tabular_features["mlp_metadata"]

        # 3. Store Labels
        self.labels = labels

        # 4. Validation
        n_samples = len(self.title_emb)

        # Verify metadata alignment
        if len(self.metadata) != n_samples:
            raise ValueError(
                f"Sample count mismatch: Title embeddings ({n_samples}) vs Metadata ({len(self.metadata)})"
            )

        # Verify label alignment if provided
        if self.labels is not None:
            if len(self.labels) != n_samples:
                raise ValueError(
                    f"Sample count mismatch: Features ({n_samples}) vs Labels ({len(self.labels)})"
                )

            # Ensure labels are a numpy array for consistent indexing
            if not isinstance(self.labels, np.ndarray):
                self.labels = np.array(self.labels)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.title_emb)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index and converts features to PyTorch tensors.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the following FloatTensors:
                - 'title_emb': (384,)
                - 'body_emb': (384,)
                - 'hist_seq': (MAX_HISTORY_LENGTH, 384)
                - 'hist_mask': (MAX_HISTORY_LENGTH,) - 1.0 for valid, 0.0 for padding
                - 'hist_centroid': (384,)
                - 'metadata': (Num_Metadata_Features,)
                - 'label': (1,) - Only if labels were provided
        """
        # Convert features to FloatTensors
        # Note: Tensors are created on CPU. Transfer to GPU happens in the training loop.
        sample = {
            "title_emb": torch.tensor(self.title_emb[idx], dtype=torch.float32),
            "body_emb": torch.tensor(self.body_emb[idx], dtype=torch.float32),
            "hist_seq": torch.tensor(self.hist_seq[idx], dtype=torch.float32),
            "hist_mask": torch.tensor(self.hist_mask[idx], dtype=torch.float32),
            "hist_centroid": torch.tensor(self.hist_centroid[idx], dtype=torch.float32),
            "metadata": torch.tensor(self.metadata[idx], dtype=torch.float32),
        }

        # Add label if available
        if self.labels is not None:
            # Unsqueeze to shape (1,) for compatibility with BCEWithLogitsLoss (Batch, 1)
            sample["label"] = torch.tensor(
                self.labels[idx], dtype=torch.float32
            ).unsqueeze(0)

        return sample
