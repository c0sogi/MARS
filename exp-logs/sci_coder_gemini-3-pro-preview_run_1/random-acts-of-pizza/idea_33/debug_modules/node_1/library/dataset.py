import torch
import numpy as np
from torch.utils.data import Dataset


class PizzaDataset(Dataset):
    """
    PyTorch Dataset for the Dual-Query Alignment-Injected MLP.
    Handles multi-modal inputs: Text Embeddings (Title/Body), User History Sequences, and Metadata.
    """

    def __init__(self, data_dict):
        """
        Initializes the dataset with a dictionary of numpy arrays.

        Args:
            data_dict (dict): Dictionary containing the following keys:
                - 'title_emb': (N, 384) SBERT embeddings for request titles.
                - 'body_emb': (N, 384) SBERT embeddings for request bodies.
                - 'history_seq': (N, SeqLen, 384) Sequence of user history embeddings.
                - 'history_mask': (N, SeqLen) Attention mask for history sequence (1=real, 0=pad).
                - 'meta': (N, MetaDim) Processed numerical metadata.
                - 'y': (N,) Target labels (optional, for train/val).
        """
        # Convert numpy arrays to PyTorch tensors
        # We use torch.tensor to ensure a copy is made and types are enforced to float32
        self.title_emb = torch.tensor(data_dict["title_emb"], dtype=torch.float32)
        self.body_emb = torch.tensor(data_dict["body_emb"], dtype=torch.float32)
        self.history_seq = torch.tensor(data_dict["history_seq"], dtype=torch.float32)
        self.history_mask = torch.tensor(data_dict["history_mask"], dtype=torch.float32)
        self.meta = torch.tensor(data_dict["meta"], dtype=torch.float32)

        # Handle target variable if present
        if "y" in data_dict and data_dict["y"] is not None:
            # BCEWithLogitsLoss requires float targets
            self.y = torch.tensor(data_dict["y"], dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.title_emb)

    def __getitem__(self, idx):
        """
        Retrieves a single sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary of tensors representing the sample.
                  Keys: 'title_emb', 'body_emb', 'history_seq', 'history_mask', 'meta', 'target' (optional).
        """
        sample = {
            "title_emb": self.title_emb[idx],
            "body_emb": self.body_emb[idx],
            "history_seq": self.history_seq[idx],
            "history_mask": self.history_mask[idx],
            "meta": self.meta[idx],
        }

        if self.y is not None:
            sample["target"] = self.y[idx]

        return sample
