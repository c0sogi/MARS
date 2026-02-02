import torch
from torch.utils.data import Dataset
import numpy as np


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Centroid-Augmented Dual-Query MLP.
    Wraps the pre-computed feature dictionary from FeatureGenerator.
    """

    def __init__(self, features_dict, split="train"):
        """
        Args:
            features_dict (dict): Dictionary containing numpy arrays from FeatureGenerator.
                                  Keys are expected to be prefixed with the split name
                                  (e.g., 'train_title_emb', 'val_metadata').
            split (str): 'train', 'val', or 'test' to select the correct subset of data.
        """
        self.split = split
        prefix = f"{split}_"

        # Verify that the dictionary contains keys for this split
        if f"{prefix}title_emb" not in features_dict:
            raise ValueError(f"Features for split '{split}' not found in dictionary.")

        # Extract features using the prefix convention from features.py
        self.title_embs = features_dict[f"{prefix}title_emb"]
        self.body_embs = features_dict[f"{prefix}body_emb"]
        self.history_seqs = features_dict[f"{prefix}history_seqs"]
        self.history_masks = features_dict[f"{prefix}history_masks"]
        self.centroids = features_dict[f"{prefix}centroids"]
        self.metadata = features_dict[f"{prefix}metadata"]
        self.alignment_feats = features_dict[f"{prefix}alignment_feats"]

        # Extract labels if they exist (Train/Val sets)
        self.labels = None
        label_key = f"{prefix}labels"
        if label_key in features_dict:
            self.labels = features_dict[label_key]

        # Determine dataset size
        self.n_samples = self.title_embs.shape[0]

        # Consistency check
        assert self.body_embs.shape[0] == self.n_samples, "Mismatch in feature lengths"
        assert self.metadata.shape[0] == self.n_samples, "Mismatch in feature lengths"

    def __len__(self):
        """Returns the total number of samples."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        Returns:
            dict: Dictionary of tensors containing all inputs and optionally the label.
        """
        # Convert numpy arrays to float32 tensors
        item = {
            "title_emb": torch.tensor(self.title_embs[idx], dtype=torch.float32),
            "body_emb": torch.tensor(self.body_embs[idx], dtype=torch.float32),
            "history_seqs": torch.tensor(self.history_seqs[idx], dtype=torch.float32),
            "history_mask": torch.tensor(self.history_masks[idx], dtype=torch.float32),
            "centroid": torch.tensor(self.centroids[idx], dtype=torch.float32),
            "metadata": torch.tensor(self.metadata[idx], dtype=torch.float32),
            "alignment": torch.tensor(self.alignment_feats[idx], dtype=torch.float32),
        }

        # Add label if available
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item
